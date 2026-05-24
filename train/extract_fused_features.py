from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from fileio.sound_scp import soundfile_read
from model.encoder.kimi_audio_encoder import KimiAudioFrontend


logger = logging.getLogger(__name__)


def _format_seconds(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "--:--:--"
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def extract_fused_features(
    data_dirs: list[str],
    encoder: str = "moonshotai/Kimi-Audio-7B-Instruct",
    tokenizer: str = "THUDM/glm-4-voice-tokenizer",
    sample_rate: int = 16000,
    output_root: str | Path = "dump/fused",
    force: bool = False,
    resume: bool = False,
    progress_every: int = 100,
) -> None:
    frontend_kwargs = {
        "model_repo": encoder,
        "tokenizer_repo": tokenizer,
        "sample_rate": sample_rate,
    }
    for data_dir_text in data_dirs:
        data_dir = Path(data_dir_text)
        _extract_data_dir(
            data_dir=data_dir,
            output_root=Path(output_root),
            force=force,
            resume=resume,
            progress_every=progress_every,
            frontend_kwargs=frontend_kwargs,
        )


def _extract_data_dir(
    data_dir: Path,
    output_root: Path,
    force: bool,
    resume: bool,
    progress_every: int,
    frontend_kwargs: dict,
) -> None:
    wav_scp_path = data_dir / "wav.scp"
    if not wav_scp_path.exists():
        raise FileNotFoundError(f"{wav_scp_path} not found")

    split_name = data_dir.name
    output_name = _output_name_from_data_dir(data_dir)
    split_dump_dir = output_root / output_name
    split_dump_dir.mkdir(parents=True, exist_ok=True)
    fused_scp_path = data_dir / "fused.scp"
    shape_dir = data_dir / "shape"
    shape_dir.mkdir(parents=True, exist_ok=True)
    fused_shape_path = shape_dir / "fused_shape"

    lines = _read_manifest_lines(wav_scp_path)
    if not lines:
        raise RuntimeError(f"{wav_scp_path} is empty")

    if not force and not resume and fused_scp_path.exists() and fused_shape_path.exists():
        logger.info("[%s] reuse existing fused features for %s", split_name, output_name)
        return

    logger.info("[%s] extracting fused features from %s", split_name, wav_scp_path)
    frontend: KimiAudioFrontend | None = None
    start_time = time.time()
    scp_tmp_path = fused_scp_path.with_suffix(fused_scp_path.suffix + ".tmp")
    shape_tmp_path = fused_shape_path.with_suffix(fused_shape_path.suffix + ".tmp")

    with scp_tmp_path.open("w", encoding="utf-8") as scp_dst, shape_tmp_path.open(
        "w", encoding="utf-8"
    ) as shape_dst:
        for index, line in enumerate(lines, start=1):
            utt_id, audio_paths = _parse_wav_scp_line(line)
            fused_path = split_dump_dir / f"{utt_id}.pt"
            if force or not fused_path.exists():
                if frontend is None:
                    frontend = KimiAudioFrontend(**frontend_kwargs)
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
            if _should_report_progress(index, len(lines), progress_every):
                _log_progress(split_name, index, len(lines), start_time)

    scp_tmp_path.replace(fused_scp_path)
    shape_tmp_path.replace(fused_shape_path)

    logger.info("[%s] wrote %s and %s", split_name, fused_scp_path, fused_shape_path)


def _read_manifest_lines(wav_scp_path: Path) -> list[str]:
    with wav_scp_path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def _parse_wav_scp_line(line: str) -> tuple[str, list[str]]:
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Invalid wav.scp line: {line}")
    return parts[0], parts[1:]


def _should_report_progress(index: int, total: int, progress_every: int) -> bool:
    return index == total or (progress_every > 0 and index % progress_every == 0)


def _log_progress(split_name: str, completed: int, total: int, start_time: float) -> None:
    elapsed = max(1e-6, time.time() - start_time)
    rate = completed / elapsed
    remaining = max(0, total - completed)
    eta = remaining / rate if rate > 0 else float("inf")
    logger.info(
        "[%s] progress %s/%s (%.1f%%) eta=%s rate=%.1f utt/s",
        split_name,
        completed,
        total,
        completed / max(1, total) * 100.0,
        _format_seconds(eta),
        rate,
    )


def _output_name_from_data_dir(data_dir: Path) -> str:
    data_root = Path("data").resolve()
    resolved = data_dir.resolve()
    try:
        return str(resolved.relative_to(data_root))
    except ValueError:
        return data_dir.name


def build_parser():
    parser = argparse.ArgumentParser(
        description="Extract offline fused features from wav.scp.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        action="append",
        required=True,
        help="Dataset directory containing wav.scp. May be repeated.",
    )
    parser.add_argument(
        "--encoder",
        default="moonshotai/Kimi-Audio-7B-Instruct",
        help="Encoder model repo/path used by KimiAudioFrontend.",
    )
    parser.add_argument(
        "--tokenizer",
        default="THUDM/glm-4-voice-tokenizer",
        help="Audio tokenizer repo/path used by KimiAudioFrontend.",
    )
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--output_root", type=str, default="dump/fused")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing per-utterance .pt files and rebuild fused.scp/fused_shape.",
    )
    parser.add_argument("--progress_every", type=int, default=100)
    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    extract_fused_features(
        data_dirs=args.data_dir,
        encoder=args.encoder,
        tokenizer=args.tokenizer,
        sample_rate=args.sample_rate,
        output_root=args.output_root,
        force=args.force,
        resume=args.resume,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
