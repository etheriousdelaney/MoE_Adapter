from __future__ import annotations

import argparse
import concurrent.futures
import logging
import math
import multiprocessing
import os
import queue
import sys
import time
from pathlib import Path
from typing import Iterable

import torch

from fileio.sound_scp import soundfile_read
from model.encoder.kimi_audio_encoder import KimiAudioFrontend
from train.config import TrainConfig
from utils import config_argparse


logger = logging.getLogger(__name__)


class _SplitProgressBar:
    def __init__(self, split_name: str, total: int):
        self.split_name = split_name
        self.total = max(1, total)
        self.completed = 0
        self.start_time = time.time()
        self._last_render = 0.0
        self._tty = sys.stderr.isatty()

    def update(self, n: int) -> None:
        self.completed = min(self.total, self.completed + max(0, n))
        now = time.time()
        if self._tty:
            if now - self._last_render >= 0.2 or self.completed >= self.total:
                self._render(final=(self.completed >= self.total))
                self._last_render = now
        elif now - self._last_render >= 10.0 or self.completed >= self.total:
            self._render_log(final=(self.completed >= self.total))
            self._last_render = now

    def close(self) -> None:
        if self._tty:
            self._render(final=True)
            sys.stderr.write("\n")
            sys.stderr.flush()
        else:
            self._render_log(final=True)

    def _render(self, final: bool) -> None:
        ratio = self.completed / self.total
        width = 28
        filled = min(width, int(width * ratio))
        bar = "#" * filled + "-" * (width - filled)
        elapsed = max(1e-6, time.time() - self.start_time)
        rate = self.completed / elapsed
        remaining = max(0, self.total - self.completed)
        eta = remaining / rate if rate > 0 else float("inf")
        message = (
            f"\r[{self.split_name}] [{bar}] "
            f"{self.completed}/{self.total} "
            f"({ratio * 100:5.1f}%) "
            f"ETA {_format_seconds(eta)} "
            f"{rate:6.1f} utt/s"
        )
        if final:
            message += " done"
        sys.stderr.write(message)
        sys.stderr.flush()

    def _render_log(self, final: bool) -> None:
        ratio = self.completed / self.total
        elapsed = max(1e-6, time.time() - self.start_time)
        rate = self.completed / elapsed
        remaining = max(0, self.total - self.completed)
        eta = remaining / rate if rate > 0 else float("inf")
        logger.info(
            "[%s] progress %s/%s (%.1f%%) eta=%s rate=%.1f utt/s%s",
            self.split_name,
            self.completed,
            self.total,
            ratio * 100.0,
            _format_seconds(eta),
            rate,
            " done" if final else "",
        )


def _format_seconds(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "--:--:--"
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def extract_fused_features(
    config: TrainConfig,
    splits: list[str],
    output_root: str | Path = "dump/fused",
    nj: int = 1,
    log_dir: str | Path | None = None,
    force: bool = False,
    progress_every: int = 100,
) -> None:
    for split_name in splits:
        if split_name == "train":
            data_name = config.dataset.train_data
        elif split_name == "valid":
            data_name = config.dataset.valid_data
        else:
            raise ValueError(f"Unsupported split_name={split_name}")

        _extract_split(
            config=config,
            split_name=split_name,
            data_name=data_name,
            output_root=Path(output_root),
            nj=nj,
            log_dir=Path(log_dir) if log_dir else None,
            force=force,
            progress_every=progress_every,
        )


def _extract_split(
    config: TrainConfig,
    split_name: str,
    data_name: str,
    output_root: Path,
    nj: int,
    log_dir: Path | None,
    force: bool,
    progress_every: int,
) -> None:
    data_dir = Path("data") / data_name
    wav_scp_path = data_dir / "wav.scp"
    if not wav_scp_path.exists():
        raise FileNotFoundError(f"{wav_scp_path} not found")

    split_dump_dir = output_root / data_name
    split_dump_dir.mkdir(parents=True, exist_ok=True)
    fused_scp_path = data_dir / "fused.scp"
    shape_dir = data_dir / "shape"
    shape_dir.mkdir(parents=True, exist_ok=True)
    fused_shape_path = shape_dir / "fused_shape"

    lines = _read_manifest_lines(wav_scp_path)
    if not lines:
        raise RuntimeError(f"{wav_scp_path} is empty")

    if not force and fused_scp_path.exists() and fused_shape_path.exists():
        logger.info("[%s] reuse existing fused features for %s", split_name, data_name)
        return

    actual_nj = max(1, min(nj, len(lines)))
    logger.info(
        "[%s] extracting fused features with nj=%s from %s",
        split_name,
        actual_nj,
        wav_scp_path,
    )
    parts_dir = split_dump_dir / ".parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_previous_parts(parts_dir, split_name)
    chunk_size = math.ceil(len(lines) / actual_nj)

    frontend_kwargs = {
        "model_repo": config.model.model_repo,
        "tokenizer_repo": config.model.tokenizer_repo,
        "sample_rate": config.model.sample_rate,
    }

    futures: list[concurrent.futures.Future] = []
    progress_bar = _SplitProgressBar(split_name=split_name, total=len(lines))
    with multiprocessing.Manager() as manager:
        progress_queue = manager.Queue()
        with concurrent.futures.ProcessPoolExecutor(max_workers=actual_nj) as executor:
            for job_id, chunk in enumerate(_chunked(lines, chunk_size), start=1):
                scp_part = parts_dir / f"{split_name}.{job_id}.scp"
                shape_part = parts_dir / f"{split_name}.{job_id}.shape"
                worker_log = None
                if log_dir is not None:
                    worker_log = log_dir / f"{split_name}.{job_id}.log"
                futures.append(
                    executor.submit(
                        _extract_part,
                        frontend_kwargs,
                        chunk,
                        split_dump_dir,
                        scp_part,
                        shape_part,
                        split_name,
                        job_id,
                        force,
                        progress_every,
                        worker_log,
                        progress_queue,
                    )
                )

            completed_utts = 0
            finished_jobs = 0
            while finished_jobs < len(futures):
                try:
                    event = progress_queue.get(timeout=0.5)
                except queue.Empty:
                    event = None

                if event is not None:
                    if event["type"] == "progress":
                        progress_bar.update(int(event["delta"]))
                    elif event["type"] == "done":
                        finished_jobs += 1

                for future in futures:
                    if future.done() and not getattr(future, "_codex_result_collected", False):
                        job_id, num_utts = future.result()
                        future._codex_result_collected = True
                        completed_utts += num_utts
                        logger.info(
                            "[%s] job %s finished (%s/%s utterances)",
                            split_name,
                            job_id,
                            completed_utts,
                            len(lines),
                        )

        progress_bar.close()

    with fused_scp_path.open("w", encoding="utf-8") as scp_dst:
        for scp_part in sorted(parts_dir.glob(f"{split_name}.*.scp")):
            with scp_part.open("r", encoding="utf-8") as src:
                scp_dst.write(src.read())

    with fused_shape_path.open("w", encoding="utf-8") as shape_dst:
        for shape_part in sorted(parts_dir.glob(f"{split_name}.*.shape")):
            with shape_part.open("r", encoding="utf-8") as src:
                shape_dst.write(src.read())

    logger.info("[%s] wrote %s and %s", split_name, fused_scp_path, fused_shape_path)


def _extract_part(
    frontend_kwargs: dict,
    lines: list[str],
    split_dump_dir: Path,
    scp_part: Path,
    shape_part: Path,
    split_name: str,
    job_id: int,
    force: bool,
    progress_every: int,
    worker_log: Path | None,
    progress_queue,
) -> tuple[int, int]:
    if worker_log is not None:
        worker_log.parent.mkdir(parents=True, exist_ok=True)
    _worker_log(worker_log, f"[{split_name}] job {job_id} start: {len(lines)} utterances")

    frontend = KimiAudioFrontend(**frontend_kwargs)
    last_reported = 0
    with scp_part.open("w", encoding="utf-8") as scp_dst, shape_part.open(
        "w", encoding="utf-8"
    ) as shape_dst:
        for index, line in enumerate(lines, start=1):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            utt_id = parts[0]
            audio_paths = parts[1:]
            fused_path = split_dump_dir / f"{utt_id}.pt"
            if force or not fused_path.exists():
                audio, _ = soundfile_read(audio_paths, dtype="float32")
                fused_states, length = frontend.extract_fused_states(audio)
                fused_states = fused_states.float()
                torch.save(fused_states, fused_path)
            else:
                fused_states = torch.load(fused_path, map_location="cpu")
                length = int(fused_states.shape[0])

            feature_dim = int(fused_states.shape[-1])
            scp_dst.write(f"{utt_id} {fused_path.resolve()}\n")
            shape_dst.write(f"{utt_id} {length},{feature_dim}\n")
            if progress_every > 0 and (index == len(lines) or index % progress_every == 0):
                delta = index - last_reported
                if delta > 0:
                    progress_queue.put({"type": "progress", "delta": delta})
                    last_reported = index
                _worker_log(
                    worker_log,
                    f"[{split_name}] job {job_id} progress: {index}/{len(lines)} utterances",
                )

        if last_reported < len(lines):
            progress_queue.put({"type": "progress", "delta": len(lines) - last_reported})

    _worker_log(worker_log, f"[{split_name}] job {job_id} done")
    progress_queue.put({"type": "done", "job_id": job_id})
    return job_id, len(lines)


def _cleanup_previous_parts(parts_dir: Path, split_name: str) -> None:
    for old_part in parts_dir.glob(f"{split_name}.*.scp"):
        old_part.unlink(missing_ok=True)
    for old_part in parts_dir.glob(f"{split_name}.*.shape"):
        old_part.unlink(missing_ok=True)


def _read_manifest_lines(wav_scp_path: Path) -> list[str]:
    with wav_scp_path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def _chunked(lines: list[str], chunk_size: int) -> Iterable[list[str]]:
    for start in range(0, len(lines), chunk_size):
        yield lines[start : start + chunk_size]


def _worker_log(worker_log: Path | None, message: str) -> None:
    if worker_log is None:
        return
    with worker_log.open("a", encoding="utf-8") as f:
        f.write(message + os.linesep)


def build_parser():
    class ArgumentDefaultsRawTextHelpFormatter(
        argparse.RawTextHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
    ):
        pass

    parser = config_argparse.ArgumentParser(
        description="Extract offline fused frontend features from dataset_conf train_data/valid_data",
        formatter_class=ArgumentDefaultsRawTextHelpFormatter,
    )
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--ngpu", type=int, default=1)
    parser.add_argument("--exp_tag", type=str, default="")
    parser.add_argument("--model", type=str, default="moeclassifier")
    parser.add_argument("--model_conf", default=dict())
    parser.add_argument("--optimizer_conf", default=dict())
    parser.add_argument("--dataset_conf", default=dict())
    parser.add_argument("--token_type", type=str, default=None)
    parser.add_argument("--token_list", type=str, default="")
    parser.add_argument("--non_linguistic_symbols", type=str, default=None)
    parser.add_argument("--seed", type=int, default=314562)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="ddp_find_unused_parameters_true")
    parser.add_argument("--task", type=str, default="classify")
    parser.add_argument("--use_tensorboard", action="store_true", default=False)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_name", type=str, default="")
    parser.add_argument(
        "--best_model_criterion",
        action="append",
        nargs=3,
        metavar=("MONITOR", "MODE", "NBEST"),
        default=None,
    )
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    parser.add_argument("--output_root", type=str, default="dump/fused")
    parser.add_argument("--nj", type=int, default=1)
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress_every", type=int, default=100)
    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args, _ = build_parser().parse_known_args()
    config = TrainConfig.from_namespace(args)
    extract_fused_features(
        config=config,
        splits=args.splits,
        output_root=args.output_root,
        nj=args.nj,
        log_dir=args.log_dir or None,
        force=args.force,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
