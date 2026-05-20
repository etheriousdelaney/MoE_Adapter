from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from loguru import logger

from dataset.instruction_utils import iter_instruction_records, strip_instruction_sample_id
from inference.asr_score import read_text, read_utt2spk, run_sclite, write_trn


ENV_ALIASES = {
    "BUS": "BUS",
    "CAFE": "CAFE",
    "CAF": "CAFE",
    "PEDESTRIAN": "PEDESTRIAN",
    "PED": "PEDESTRIAN",
    "STREET": "STREET",
    "STR": "STREET",
}
ENV_ORDER = ["BUS", "CAFE", "PEDESTRIAN", "STREET"]
GENDER_ORDER = ["female", "male"]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score multitask instruction decode outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--decode_dir", required=True)
    parser.add_argument("--score_opts", default="")
    return parser


def sample_index(sample_id: str) -> int:
    match = re.search(r"__sample__(\d+)$", sample_id)
    if not match:
        return 0
    return int(match.group(1))


def read_instruction_prompts(path: Path) -> dict[str, str]:
    return {record["sample_id"]: record["prompt"] for record in iter_instruction_records(path)}


def read_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.items()
    elif isinstance(payload, list):
        items = ((item.get("id"), item) for item in payload if isinstance(item, dict))
    else:
        raise ValueError(f"Unsupported metadata format: {path}")
    metadata: dict[str, dict[str, str]] = {}
    for key, item in items:
        if key is None or not isinstance(item, dict):
            continue
        metadata[str(key)] = {str(k): str(v) for k, v in item.items()}
    return metadata


def clean_asr_reference(base_id: str, metadata: dict[str, dict[str, str]], text_refs: dict[str, str]) -> str:
    if base_id in metadata and metadata[base_id].get("text"):
        return metadata[base_id]["text"]
    return text_refs.get(base_id, "")


def normalize_environment(text: str) -> str:
    normalized = text.strip().upper()
    for key, value in ENV_ALIASES.items():
        if normalized == key or key in normalized:
            return value
    return normalized


def normalize_gender(text: str) -> str:
    normalized = text.strip().lower()
    if "female" in normalized:
        return "female"
    if "male" in normalized:
        return "male"
    return normalized


def environment_reference(base_id: str, metadata: dict[str, dict[str, str]], response: str) -> str:
    if base_id in metadata and metadata[base_id].get("environment"):
        return normalize_environment(metadata[base_id]["environment"])
    return normalize_environment(response)


def gender_reference(base_id: str, metadata: dict[str, dict[str, str]], response: str) -> str:
    if base_id in metadata:
        for key in ("Gender", "gender"):
            if metadata[base_id].get(key):
                return normalize_gender(metadata[base_id][key])
    return normalize_gender(response)


def expand_instruction_utt2spk(
    sample_ids: list[str],
    base_utt2spk: dict[str, str],
) -> dict[str, str]:
    speakers: dict[str, str] = {}
    for sample_id in sample_ids:
        base_id = strip_instruction_sample_id(sample_id)
        speakers[sample_id] = base_utt2spk.get(base_id, base_id)
    return speakers


def split_references(
    dataset_dir: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    metadata = read_metadata(dataset_dir / "metadata.json")
    text_refs = read_text(dataset_dir / "text") if (dataset_dir / "text").exists() else {}
    asr_refs: dict[str, str] = {}
    env_refs: dict[str, str] = {}
    gender_refs: dict[str, str] = {}
    prompts: dict[str, str] = {}
    for record in iter_instruction_records(dataset_dir / "response.jsonl"):
        sample_id = record["sample_id"]
        base_id = record["base_id"]
        prompts[sample_id] = record["prompt"]
        index = sample_index(sample_id)
        if index == 0:
            asr_refs[sample_id] = clean_asr_reference(base_id, metadata, text_refs)
        elif index == 1:
            env_refs[sample_id] = environment_reference(base_id, metadata, record["response"])
        elif index == 2:
            gender_refs[sample_id] = gender_reference(base_id, metadata, record["response"])
    return asr_refs, env_refs, gender_refs, prompts


def write_instruction_details(
    score_dir: Path,
    prompts: dict[str, str],
    ref_text: dict[str, str],
    hyp_text: dict[str, str],
    utt2spk: dict[str, str],
) -> Path:
    details_path = score_dir / "details.tsv"
    score_dir.mkdir(parents=True, exist_ok=True)
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


def score_asr(
    decode_dir: Path,
    hyp_text: dict[str, str],
    asr_refs: dict[str, str],
    prompts: dict[str, str],
    utt2spk: dict[str, str],
    score_opts: str,
) -> None:
    for unit, score_dir_name in (("cer", "score_asr_cer"), ("wer", "score_asr_wer")):
        score_dir = decode_dir / score_dir_name
        ref_path, hyp_path = write_trn(
            ref_text=asr_refs,
            hyp_text=hyp_text,
            utt2spk=utt2spk,
            output_dir=score_dir,
            unit=unit,
        )
        result_path = score_dir / "result.txt"
        run_sclite(ref_path, hyp_path, result_path, score_opts=score_opts)
        write_instruction_details(score_dir, prompts, asr_refs, hyp_text, utt2spk)
        logger.info("Write ASR {} result in {}", unit, result_path)


def score_accuracy(
    decode_dir: Path,
    task_name: str,
    hyp_text: dict[str, str],
    refs: dict[str, str],
    prompts: dict[str, str],
    label_order: list[str],
    normalizer,
) -> None:
    score_dir = decode_dir / f"score_{task_name}_accuracy"
    score_dir.mkdir(parents=True, exist_ok=True)
    labels = list(label_order)
    confusion = {ref: {hyp: 0 for hyp in labels + ["OTHER"]} for ref in labels + ["OTHER"]}
    correct = 0
    total = 0
    details_path = score_dir / "details.tsv"
    with details_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["sample_id", "prompt", "reference", "hypothesis", "correct"])
        for sample_id in sorted(refs):
            ref = normalizer(refs[sample_id])
            hyp = normalizer(hyp_text.get(sample_id, ""))
            if ref not in labels:
                ref = "OTHER"
            if hyp not in labels:
                hyp = "OTHER"
            is_correct = ref == hyp
            correct += int(is_correct)
            total += 1
            confusion.setdefault(ref, {name: 0 for name in labels + ["OTHER"]})
            confusion[ref][hyp] = confusion[ref].get(hyp, 0) + 1
            writer.writerow([sample_id, prompts.get(sample_id, ""), ref, hyp, int(is_correct)])

    accuracy = correct / total if total else 0.0
    result_path = score_dir / "result.txt"
    with result_path.open("w", encoding="utf-8") as f:
        f.write(f"task\t{task_name}\n")
        f.write(f"correct\t{correct}\n")
        f.write(f"total\t{total}\n")
        f.write(f"accuracy\t{accuracy:.6f}\n")
        f.write(f"accuracy_percent\t{accuracy * 100.0:.2f}\n")
        f.write(f"details_tsv\t{details_path.name}\n")

    matrix_path = score_dir / "confusion.tsv"
    columns = labels + ["OTHER"]
    with matrix_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["ref\\hyp", *columns])
        for ref in columns:
            writer.writerow([ref, *[confusion.get(ref, {}).get(hyp, 0) for hyp in columns]])
    logger.info("Write {} accuracy result in {}", task_name, result_path)


def main() -> None:
    args = build_argparser().parse_args()
    dataset_dir = Path("data") / args.dataset
    decode_dir = Path(args.decode_dir)
    hyp_text = read_text(decode_dir / "text")
    base_utt2spk = read_utt2spk(dataset_dir / "utt2spk")

    asr_refs, env_refs, gender_refs, prompts = split_references(dataset_dir)
    all_sample_ids = list(asr_refs) + list(env_refs) + list(gender_refs)
    utt2spk = expand_instruction_utt2spk(all_sample_ids, base_utt2spk)

    score_asr(
        decode_dir=decode_dir,
        hyp_text=hyp_text,
        asr_refs=asr_refs,
        prompts=prompts,
        utt2spk=utt2spk,
        score_opts=args.score_opts,
    )
    score_accuracy(
        decode_dir=decode_dir,
        task_name="environment",
        hyp_text=hyp_text,
        refs=env_refs,
        prompts=prompts,
        label_order=ENV_ORDER,
        normalizer=normalize_environment,
    )
    score_accuracy(
        decode_dir=decode_dir,
        task_name="gender",
        hyp_text=hyp_text,
        refs=gender_refs,
        prompts=prompts,
        label_order=GENDER_ORDER,
        normalizer=normalize_gender,
    )


if __name__ == "__main__":
    main()
