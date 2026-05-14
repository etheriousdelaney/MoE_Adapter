from __future__ import annotations

import argparse
import csv
from pathlib import Path

from loguru import logger

from dataset.instruction_utils import iter_instruction_records, strip_instruction_sample_id
from inference.asr_score import read_text, read_utt2spk, run_sclite, write_trn


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score instruction decode outputs with response.jsonl references",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--decode_dir", required=True)
    parser.add_argument("--score_opts", default="")
    return parser


def read_instruction_ref_text(path: Path) -> dict[str, str]:
    items: dict[str, str] = {}
    for record in iter_instruction_records(path):
        items[record["sample_id"]] = record["response"]
    return items


def read_instruction_prompts(path: Path) -> dict[str, str]:
    items: dict[str, str] = {}
    for record in iter_instruction_records(path):
        items[record["sample_id"]] = record["prompt"]
    return items


def expand_instruction_utt2spk(
    ref_text: dict[str, str],
    base_utt2spk: dict[str, str],
) -> dict[str, str]:
    speakers: dict[str, str] = {}
    for sample_id in ref_text:
        base_id = strip_instruction_sample_id(sample_id)
        speakers[sample_id] = base_utt2spk.get(base_id, base_id)
    return speakers


def write_instruction_details(
    score_dir: Path,
    prompts: dict[str, str],
    ref_text: dict[str, str],
    hyp_text: dict[str, str],
    utt2spk: dict[str, str],
) -> Path:
    details_path = score_dir / "details.tsv"
    with details_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["sample_id", "speaker", "prompt", "reference", "hypothesis"])
        for sample_id in sorted(ref_text):
            writer.writerow(
                [
                    sample_id,
                    utt2spk.get(sample_id, strip_instruction_sample_id(sample_id)),
                    prompts.get(sample_id, ""),
                    ref_text.get(sample_id, ""),
                    hyp_text.get(sample_id, ""),
                ]
            )
    return details_path


def append_prompt_section(
    result_path: Path,
    details_path: Path,
    prompts: dict[str, str],
) -> None:
    with result_path.open("a", encoding="utf-8") as f:
        f.write("\n\nINSTRUCTION PROMPTS\n")
        f.write(f"details_tsv\t{details_path.name}\n")
        f.write("sample_id\tprompt\n")
        for sample_id in sorted(prompts):
            prompt = prompts[sample_id].replace("\t", " ").replace("\n", " ").strip()
            f.write(f"{sample_id}\t{prompt}\n")


def main() -> None:
    args = build_argparser().parse_args()
    dataset_dir = Path("data") / args.dataset
    decode_dir = Path(args.decode_dir)
    hyp_text_path = decode_dir / "text"
    ref_response_path = dataset_dir / "response.jsonl"
    utt2spk_path = dataset_dir / "utt2spk"

    ref_text = read_instruction_ref_text(ref_response_path)
    prompt_text = read_instruction_prompts(ref_response_path)
    hyp_text = read_text(hyp_text_path)
    utt2spk = expand_instruction_utt2spk(ref_text, read_utt2spk(utt2spk_path))

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
        details_path = write_instruction_details(
            score_dir=score_dir,
            prompts=prompt_text,
            ref_text=ref_text,
            hyp_text=hyp_text,
            utt2spk=utt2spk,
        )
        append_prompt_section(
            result_path=result_path,
            details_path=details_path,
            prompts=prompt_text,
        )
        logger.info("Write {} result in {}", unit, result_path)


if __name__ == "__main__":
    main()
