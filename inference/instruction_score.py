from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

from loguru import logger

from dataset.instruction_utils import iter_instruction_records, strip_instruction_sample_id
from inference.instruction_inference import load_train_config


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


def read_text(path: Path) -> dict[str, str]:
    items: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            uttid = parts[0]
            items[uttid] = parts[1] if len(parts) > 1 else ""
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
    return list(text.replace(" ", ""))


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

    with ref_path.open("w", encoding="utf-8") as ref_f, hyp_path.open(
        "w", encoding="utf-8"
    ) as hyp_f:
        for uttid in sorted(ref_text):
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
    cmd.extend(
        [
            "-r",
            str(ref_path),
            "trn",
            "-h",
            str(hyp_path),
            "trn",
            "-i",
            "rm",
            "-o",
            "all",
            "stdout",
        ]
    )
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    result_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"sclite failed: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score multitask instruction decode outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--decode_dir", required=True)
    parser.add_argument("--train_config", default="")
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


def default_instruction_tasks() -> list[dict[str, object]]:
    return [
        {
            "name": "asr",
            "answer_field": "text",
            "scoring": ["wer", "cer"],
        },
        {
            "name": "environment",
            "answer_field": "environment",
            "scoring": ["acc"],
            "labels": ENV_ORDER,
        },
        {
            "name": "gender",
            "answer_field": "Gender",
            "scoring": ["acc"],
            "labels": GENDER_ORDER,
        },
    ]


def load_instruction_tasks(train_config_path: str) -> list[dict[str, object]]:
    if not train_config_path:
        return default_instruction_tasks()
    config = load_train_config(train_config_path)
    tasks = [dict(task) for task in getattr(config.dataset, "instruction_tasks", [])]
    return tasks or default_instruction_tasks()


def task_name(task: dict[str, object], index: int) -> str:
    return str(task.get("name") or f"task_{index}")


def task_metrics(task: dict[str, object]) -> list[str]:
    metrics = task.get("scoring", [])
    if isinstance(metrics, str):
        metrics = [metrics]
    return [str(metric).lower() for metric in metrics]


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


def metadata_reference(
    base_id: str,
    metadata: dict[str, dict[str, str]],
    answer_field: str,
    response: str,
    text_refs: dict[str, str],
    field_refs: dict[str, str],
) -> str:
    if answer_field == "text":
        return clean_asr_reference(base_id, metadata, text_refs)
    if answer_field == "environment":
        if base_id in field_refs:
            return normalize_environment(field_refs[base_id])
        return environment_reference(base_id, metadata, response)
    if answer_field in {"Gender", "gender"}:
        if base_id in field_refs:
            return normalize_gender(field_refs[base_id])
        return gender_reference(base_id, metadata, response)
    if base_id in metadata and metadata[base_id].get(answer_field):
        return metadata[base_id][answer_field]
    if base_id in field_refs:
        return field_refs[base_id]
    return response


def normalize_by_labels(text: str, labels: list[str]) -> str:
    if not labels:
        return text.strip()
    exact = {label.lower(): label for label in labels}
    normalized = text.strip()
    lower = normalized.lower()
    if lower in exact:
        return exact[lower]
    for label in sorted(labels, key=len, reverse=True):
        if label.lower() in lower:
            return label
    return normalized


def build_task_references(
    dataset_dir: Path,
    tasks: list[dict[str, object]],
    sample_ids: list[str],
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    metadata = read_metadata(dataset_dir / "metadata.json")
    text_refs = read_text(dataset_dir / "text") if (dataset_dir / "text").exists() else {}
    field_cache: dict[str, dict[str, str]] = {}
    refs_by_task = {task_name(task, index): {} for index, task in enumerate(tasks)}
    prompts: dict[str, str] = {}

    response_path = dataset_dir / "response.jsonl"
    if response_path.exists():
        records = list(iter_instruction_records(response_path))
    else:
        records = [
            {
                "sample_id": sample_id,
                "base_id": strip_instruction_sample_id(sample_id),
                "prompt": "",
                "response": "",
            }
            for sample_id in sample_ids
        ]

    for record in records:
        sample_id = record["sample_id"]
        if sample_ids and sample_id not in sample_ids:
            continue
        base_id = record["base_id"]
        index = sample_index(sample_id)
        if index >= len(tasks):
            continue
        task = tasks[index]
        name = task_name(task, index)
        prompts[sample_id] = record.get("prompt") or str(task.get("prompt", ""))
        answer_field = str(task.get("answer_field", "text"))
        if answer_field not in field_cache:
            field_path = dataset_dir / answer_field
            field_cache[answer_field] = read_text(field_path) if field_path.exists() else {}
        reference = metadata_reference(
            base_id=base_id,
            metadata=metadata,
            answer_field=answer_field,
            response=record.get("response", ""),
            text_refs=text_refs,
            field_refs=field_cache[answer_field],
        )
        label_map = task.get("label_map") or {}
        if isinstance(label_map, dict):
            reference = str(label_map.get(reference, label_map.get(str(reference), reference)))
        refs_by_task[name][sample_id] = reference
    return refs_by_task, prompts


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
    tasks = load_instruction_tasks(args.train_config)
    refs_by_task, prompts = build_task_references(
        dataset_dir=dataset_dir,
        tasks=tasks,
        sample_ids=list(hyp_text),
    )
    all_sample_ids = [
        sample_id
        for refs in refs_by_task.values()
        for sample_id in refs
    ]
    utt2spk = expand_instruction_utt2spk(all_sample_ids, base_utt2spk)

    for index, task in enumerate(tasks):
        name = task_name(task, index)
        refs = refs_by_task.get(name, {})
        if not refs:
            logger.warning("Skip task={} because no references were found", name)
            continue
        metrics = task_metrics(task)
        if "wer" in metrics or "cer" in metrics:
            asr_metrics = [metric for metric in ("cer", "wer") if metric in metrics]
            for unit in asr_metrics:
                score_dir = decode_dir / f"score_{name}_{unit}"
                ref_path, hyp_path = write_trn(
                    ref_text=refs,
                    hyp_text=hyp_text,
                    utt2spk=utt2spk,
                    output_dir=score_dir,
                    unit=unit,
                )
                result_path = score_dir / "result.txt"
                run_sclite(ref_path, hyp_path, result_path, score_opts=args.score_opts)
                write_instruction_details(score_dir, prompts, refs, hyp_text, utt2spk)
                logger.info("Write {} {} result in {}", name, unit, result_path)
        if "acc" in metrics or "accuracy" in metrics:
            labels = [str(label) for label in task.get("labels", [])]
            normalizer = lambda text, labels=labels: normalize_by_labels(text, labels)
            if not labels:
                labels = sorted({normalizer(value) for value in refs.values()})
            score_accuracy(
                decode_dir=decode_dir,
                task_name=name,
                hyp_text=hyp_text,
                refs=refs,
                prompts=prompts,
                label_order=labels,
                normalizer=normalizer,
            )


if __name__ == "__main__":
    main()
