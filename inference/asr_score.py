from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from loguru import logger


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score ASR decode outputs with sclite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--decode_dir", required=True)
    parser.add_argument("--score_opts", default="")
    return parser


def read_text(path: Path) -> dict[str, str]:
    items: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 0:
                continue
            uttid = parts[0]
            text = parts[1] if len(parts) > 1 else ""
            items[uttid] = text
    return items


def read_utt2spk(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    speakers: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                speakers[parts[0]] = parts[1]
    return speakers


def normalize_cer_text(text: str) -> list[str]:
    compact = text.replace(" ", "")
    return list(compact)


def normalize_wer_text(text: str) -> list[str]:
    return text.split()


def write_trn(
    ref_text: dict[str, str],
    hyp_text: dict[str, str],
    utt2spk: dict[str, str],
    output_dir: Path,
    unit: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_path = output_dir / "ref.trn"
    hyp_path = output_dir / "hyp.trn"

    normalizer = normalize_cer_text if unit == "cer" else normalize_wer_text
    uttids = sorted(ref_text)

    with ref_path.open("w", encoding="utf-8") as ref_f, hyp_path.open(
        "w", encoding="utf-8"
    ) as hyp_f:
        for uttid in uttids:
            speaker = utt2spk.get(uttid, uttid)
            suffix = f"({speaker}-{uttid})"
            ref_units = " ".join(normalizer(ref_text[uttid]))
            hyp_units = " ".join(normalizer(hyp_text.get(uttid, "")))
            ref_f.write(f"{ref_units} {suffix}\n" if ref_units else f"{suffix}\n")
            hyp_f.write(f"{hyp_units} {suffix}\n" if hyp_units else f"{suffix}\n")

    return ref_path, hyp_path


def run_sclite(ref_path: Path, hyp_path: Path, result_path: Path, score_opts: str) -> None:
    cmd = ["sclite"]
    if score_opts:
        cmd.extend(score_opts.split())
    cmd.extend(["-r", str(ref_path), "trn", "-h", str(hyp_path), "trn", "-i", "rm", "-o", "all", "stdout"])
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    result_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"sclite failed: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")


def main() -> None:
    args = build_argparser().parse_args()
    dataset_dir = Path("data") / args.dataset
    decode_dir = Path(args.decode_dir)
    hyp_text_path = decode_dir / "text"
    ref_text_path = dataset_dir / "text"
    utt2spk_path = dataset_dir / "utt2spk"

    ref_text = read_text(ref_text_path)
    hyp_text = read_text(hyp_text_path)
    utt2spk = read_utt2spk(utt2spk_path)

    for unit, score_dir_name in (("cer", "score_cer"), ("wer", "score_wer")):
        score_dir = decode_dir / score_dir_name
        ref_path, hyp_path = write_trn(
            ref_text=ref_text,
            hyp_text=hyp_text,
            utt2spk=utt2spk,
            output_dir=score_dir,
            unit=unit,
        )
        result_path = score_dir / "result.txt"
        run_sclite(ref_path, hyp_path, result_path, score_opts=args.score_opts)
        logger.info("Write {} result in {}", unit, result_path)


if __name__ == "__main__":
    main()
