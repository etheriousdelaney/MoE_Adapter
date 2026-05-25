from __future__ import annotations

import argparse
import concurrent.futures
import logging
import math
import os
from pathlib import Path
from typing import Iterable

import soundfile

from dataset.instruction_utils import compose_instruction_sample_id, iter_instruction_records
from fileio.read_text import load_num_sequence_text
from train.config import TrainConfig
from utils import config_argparse


logger = logging.getLogger(__name__)


def ensure_instruction_shape_files(
    config: TrainConfig,
    nj: int = 1,
    log_dir: str | Path | None = None,
    force: bool = False,
    progress_every: int = 500,
) -> None:
    dataset_config = config.dataset
    train_path = _ensure_split_instruction_shape(
        split_name="train",
        data_name=dataset_config.train_data,
        configured_paths=dataset_config.train_shape_file,
        data_type=dataset_config.data_type,
        instruction_source=dataset_config.instruction_source,
        instruction_tasks=dataset_config.instruction_tasks,
        nj=nj,
        log_dir=log_dir,
        force=force,
        progress_every=progress_every,
    )
    valid_path = _ensure_split_instruction_shape(
        split_name="valid",
        data_name=dataset_config.valid_data,
        configured_paths=dataset_config.valid_shape_file,
        data_type=dataset_config.data_type,
        instruction_source=dataset_config.instruction_source,
        instruction_tasks=dataset_config.instruction_tasks,
        nj=nj,
        log_dir=log_dir,
        force=force,
        progress_every=progress_every,
    )
    dataset_config.train_shape_file = [str(train_path)]
    dataset_config.valid_shape_file = [str(valid_path)]

    logger.info("train instruction_shape: %s", dataset_config.train_shape_file[0])
    logger.info("valid instruction_shape: %s", dataset_config.valid_shape_file[0])


def ensure_speech_shape_files(
    config: TrainConfig,
    nj: int = 1,
    log_dir: str | Path | None = None,
    force: bool = False,
    progress_every: int = 500,
) -> None:
    dataset_config = config.dataset
    train_path = _ensure_split_speech_shape(
        split_name="train",
        data_name=dataset_config.train_data,
        configured_paths=dataset_config.train_shape_file,
        nj=nj,
        log_dir=log_dir,
        force=force,
        progress_every=progress_every,
    )
    valid_path = _ensure_split_speech_shape(
        split_name="valid",
        data_name=dataset_config.valid_data,
        configured_paths=dataset_config.valid_shape_file,
        nj=nj,
        log_dir=log_dir,
        force=force,
        progress_every=progress_every,
    )
    dataset_config.train_shape_file = [str(train_path)]
    dataset_config.valid_shape_file = [str(valid_path)]

    logger.info("train speech_shape: %s", dataset_config.train_shape_file[0])
    logger.info("valid speech_shape: %s", dataset_config.valid_shape_file[0])


def _shape_files_ready(paths: list[str]) -> bool:
    return len(paths) > 0 and all(Path(path).exists() for path in paths)


def _ensure_split_instruction_shape(
    split_name: str,
    data_name: str,
    configured_paths: list[str],
    data_type: list[str],
    instruction_source: str,
    instruction_tasks: list[dict],
    nj: int,
    log_dir: str | Path | None,
    force: bool,
    progress_every: int,
) -> Path:
    data_dir = Path("data") / data_name
    if not data_dir.exists():
        raise FileNotFoundError(f"{split_name} data directory not found: {data_dir}")

    source_shape_path = data_dir / "shape" / "fused_shape"
    output_shape_name = "instruction_fused_shape"
    if "fused" not in data_type or not source_shape_path.exists():
        if "sound" not in data_type:
            raise FileNotFoundError(f"{split_name} fused_shape not found: {source_shape_path}")
        source_shape_path = _ensure_split_speech_shape(
            split_name=split_name,
            data_name=data_name,
            configured_paths=[],
            nj=nj,
            log_dir=log_dir,
            force=force,
            progress_every=progress_every,
        )
        output_shape_name = "instruction_speech_shape"

    instruction_shape_path = data_dir / "shape" / output_shape_name
    can_reuse_shape = instruction_source != "task_specs"
    if not force and can_reuse_shape:
        if _shape_files_ready(configured_paths):
            configured_path = Path(configured_paths[0]).resolve()
            if configured_path.exists() and configured_path == instruction_shape_path.resolve():
                logger.info("[%s] reuse configured instruction_shape: %s", split_name, configured_path)
                return configured_path
        if instruction_shape_path.exists():
            logger.info("[%s] reuse existing instruction_shape: %s", split_name, instruction_shape_path)
            return instruction_shape_path.resolve()

    utt2shape = load_num_sequence_text(source_shape_path, loader_type="csv_int")
    num_written = 0
    with instruction_shape_path.open("w", encoding="utf-8") as dst:
        for base_id, sample_id in _iter_instruction_shape_ids(
            data_dir=data_dir,
            instruction_source=instruction_source,
            instruction_tasks=instruction_tasks,
        ):
            if base_id not in utt2shape:
                raise KeyError(
                    f"{split_name} instruction sample id={base_id} not found in {source_shape_path}"
                )
            shape = ",".join(str(value) for value in utt2shape[base_id])
            dst.write(f"{sample_id} {shape}\n")
            num_written += 1

    if num_written == 0:
        raise RuntimeError(f"No instruction samples found for {data_dir}")

    logger.info(
        "[%s] wrote %s: %s (%s samples)",
        split_name,
        output_shape_name,
        instruction_shape_path.resolve(),
        num_written,
    )
    return instruction_shape_path.resolve()


def _iter_instruction_shape_ids(
    data_dir: Path,
    instruction_source: str,
    instruction_tasks: list[dict],
) -> Iterable[tuple[str, str]]:
    if instruction_source == "task_specs":
        if not instruction_tasks:
            raise ValueError("instruction_source=task_specs requires non-empty instruction_tasks")
        base_ids = _load_base_ids_for_task_specs(data_dir)
        for base_id in base_ids:
            for task_index in range(len(instruction_tasks)):
                yield base_id, compose_instruction_sample_id(base_id, task_index)
        return

    response_path = data_dir / "response.jsonl"
    if not response_path.exists():
        raise FileNotFoundError(f"response.jsonl not found: {response_path}")
    for record in iter_instruction_records(response_path):
        yield record["base_id"], record["sample_id"]


def _load_base_ids_for_task_specs(data_dir: Path) -> list[str]:
    text_path = data_dir / "text"
    if text_path.exists():
        return [line.split(maxsplit=1)[0] for line in _read_manifest_lines(text_path)]
    fused_scp_path = data_dir / "fused.scp"
    if fused_scp_path.exists():
        return [line.split(maxsplit=1)[0] for line in _read_manifest_lines(fused_scp_path)]
    wav_scp_path = data_dir / "wav.scp"
    if wav_scp_path.exists():
        return [line.split(maxsplit=1)[0] for line in _read_manifest_lines(wav_scp_path)]
    raise FileNotFoundError(f"Cannot infer task_specs base ids under {data_dir}")


def _ensure_split_speech_shape(
    split_name: str,
    data_name: str,
    configured_paths: list[str],
    nj: int,
    log_dir: str | Path | None,
    force: bool,
    progress_every: int,
) -> Path:
    data_dir = Path("data") / data_name
    if not data_dir.exists():
        raise FileNotFoundError(f"{split_name} data directory not found: {data_dir}")

    wav_scp_path = data_dir / "wav.scp"
    if not wav_scp_path.exists():
        raise FileNotFoundError(f"{split_name} wav.scp not found: {wav_scp_path}")

    split_dir = data_dir / "shape"
    split_dir.mkdir(parents=True, exist_ok=True)
    speech_shape_path = split_dir / "speech_shape"

    if not force:
        if _shape_files_ready(configured_paths):
            configured_path = Path(configured_paths[0]).resolve()
            if configured_path.exists() and configured_path == speech_shape_path.resolve():
                logger.info("[%s] reuse configured speech_shape: %s", split_name, configured_path)
                return configured_path
        if speech_shape_path.exists():
            logger.info("[%s] reuse existing speech_shape: %s", split_name, speech_shape_path)
            return speech_shape_path.resolve()

    lines = _read_manifest_lines(wav_scp_path)
    if not lines:
        raise RuntimeError(f"{wav_scp_path} is empty")

    actual_nj = max(1, min(nj, len(lines)))
    logger.info(
        "[%s] generating speech_shape with nj=%s from %s",
        split_name,
        actual_nj,
        wav_scp_path,
    )

    parts_dir = split_dir / ".parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_previous_parts(parts_dir, split_name)

    chunk_size = math.ceil(len(lines) / actual_nj)
    futures: list[concurrent.futures.Future] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=actual_nj) as executor:
        for job_id, chunk in enumerate(_chunked(lines, chunk_size), start=1):
            part_path = parts_dir / f"{split_name}.{job_id}.shape"
            worker_log = None
            if log_dir is not None:
                worker_log = Path(log_dir) / f"{split_name}.{job_id}.log"
            futures.append(
                executor.submit(
                    _write_shape_part,
                    chunk,
                    part_path,
                    split_name,
                    job_id,
                    worker_log,
                    progress_every,
                )
            )

        completed_utts = 0
        for future in concurrent.futures.as_completed(futures):
            job_id, num_utts, part_path = future.result()
            completed_utts += num_utts
            logger.info(
                "[%s] job %s finished: %s utterances -> %s (total %s/%s)",
                split_name,
                job_id,
                num_utts,
                part_path,
                completed_utts,
                len(lines),
            )

    part_paths = [parts_dir / f"{split_name}.{job_id}.shape" for job_id in range(1, len(futures) + 1)]
    with speech_shape_path.open("w", encoding="utf-8") as dst:
        for part_path in part_paths:
            with part_path.open("r", encoding="utf-8") as src:
                for line in src:
                    dst.write(line)

    logger.info("[%s] wrote merged speech_shape: %s", split_name, speech_shape_path.resolve())
    return speech_shape_path.resolve()


def _read_manifest_lines(wav_scp_path: Path) -> list[str]:
    with wav_scp_path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def _chunked(lines: list[str], chunk_size: int) -> Iterable[list[str]]:
    for start in range(0, len(lines), chunk_size):
        yield lines[start : start + chunk_size]


def _cleanup_previous_parts(parts_dir: Path, split_name: str) -> None:
    for old_part in parts_dir.glob(f"{split_name}.*.shape"):
        old_part.unlink(missing_ok=True)


def _write_shape_part(
    lines: list[str],
    part_path: Path,
    split_name: str,
    job_id: int,
    worker_log: Path | None,
    progress_every: int,
) -> tuple[int, int, str]:
    if worker_log is not None:
        worker_log.parent.mkdir(parents=True, exist_ok=True)
    _worker_log(worker_log, f"[{split_name}] job {job_id} start: {len(lines)} utterances")

    with part_path.open("w", encoding="utf-8") as dst:
        for index, line in enumerate(lines, start=1):
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            utt_id = parts[0]
            num_frames = 0
            for audio_path in parts[1:]:
                with soundfile.SoundFile(audio_path) as audio_file:
                    num_frames += len(audio_file)

            dst.write(f"{utt_id} {num_frames}\n")
            if progress_every > 0 and (index == len(lines) or index % progress_every == 0):
                _worker_log(
                    worker_log,
                    f"[{split_name}] job {job_id} progress: {index}/{len(lines)} utterances",
                )

    _worker_log(worker_log, f"[{split_name}] job {job_id} done: {part_path}")
    return job_id, len(lines), str(part_path)


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
        description="Generate speech_shape files from dataset_conf train_data/valid_data",
        formatter_class=ArgumentDefaultsRawTextHelpFormatter,
        ignore_unknown_config_keys=True,
    )
    parser.add_argument("--dataset_conf", default=dict())
    parser.add_argument("--nj", type=int, default=1)
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress_every", type=int, default=500)
    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args, _ = build_parser().parse_known_args()
    config = TrainConfig.from_namespace(args)
    if any(name in config.dataset.data_type for name in ("audio_context", "answer")):
        ensure_instruction_shape_files(
            config,
            nj=args.nj,
            log_dir=args.log_dir or None,
            force=args.force,
            progress_every=args.progress_every,
        )
    elif "sound" in config.dataset.data_type and "fused" not in config.dataset.data_type:
        ensure_speech_shape_files(
            config,
            nj=args.nj,
            log_dir=args.log_dir or None,
            force=args.force,
            progress_every=args.progress_every,
        )
    else:
        logging.info("No shape generation needed for data_type=%s", config.dataset.data_type)


if __name__ == "__main__":
    main()
