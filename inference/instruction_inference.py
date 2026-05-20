from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import torch
from loguru import logger

from dataset.instruction_utils import strip_instruction_sample_id
from task.instruction_asr import InstructionAsrTask
from inference.asr_inference import (
    DecodeConfig,
    build_dataloader,
    decode_batch,
    finalize_heatmap_matrix,
    load_keys,
    load_model_for_inference,
    load_train_config,
    normalize_heatmap_policy,
    render_heatmap,
    save_accumulator,
    create_accumulator,
    accumulate_expert_usage,
    write_results,
)

TASK_HEATMAP_ORDER = ["ASR", "environment", "gender"]
ENV_HEATMAP_ORDER = ["BUS", "CAFE", "PEDESTRIAN", "STREET"]
GENDER_HEATMAP_ORDER = ["female", "male"]
ENV_ALIASES = {
    "BUS": "BUS",
    "CAFE": "CAFE",
    "CAF": "CAFE",
    "PEDESTRIAN": "PEDESTRIAN",
    "PED": "PEDESTRIAN",
    "STREET": "STREET",
    "STR": "STREET",
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Instruction/multitask inference for MoE_Adapter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--decode_config", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--key_file", type=str, required=True)
    parser.add_argument("--ngpu", type=int, default=1)
    parser.add_argument(
        "--expert_heatmap",
        choices=("auto", "true", "false"),
        default=None,
        help="Override decode_config expert_heatmap policy.",
    )
    return parser


def resolve_instruction_data_type(train_config, dataset: str) -> list[str]:
    data_dir = Path("data") / dataset
    preferred = list(train_config.dataset.data_type)
    text_types = [name for name in preferred if name in {"audio_context", "prompt", "response"}]
    if (
        "audio_context" in preferred
        and train_config.dataset.instruction_source != "task_specs"
        and not (data_dir / "message.jsonl").exists()
    ):
        text_types = [name for name in text_types if name != "audio_context"]
    if (data_dir / "fused.scp").exists():
        return ["fused", *text_types]
    if (data_dir / "wav.scp").exists():
        return ["sound", *text_types]
    raise FileNotFoundError(
        f"Could not infer instruction inference input type under {data_dir}: "
        "expected fused.scp or wav.scp"
    )


def sample_index(sample_id: str) -> int:
    match = re.search(r"__sample__(\d+)$", sample_id)
    if not match:
        return 0
    return int(match.group(1))


def task_name_from_sample_id(sample_id: str) -> str:
    index = sample_index(sample_id)
    if index == 0:
        return "ASR"
    if index == 1:
        return "environment"
    if index == 2:
        return "gender"
    return "ASR"


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


def read_metadata(dataset: str) -> dict[str, dict[str, str]]:
    path = Path("data") / dataset / "metadata.json"
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


def environment_from_sample_id(sample_id: str, metadata: dict[str, dict[str, str]]) -> str:
    base_id = strip_instruction_sample_id(sample_id)
    if base_id in metadata and metadata[base_id].get("environment"):
        return normalize_environment(metadata[base_id]["environment"])
    return normalize_environment(base_id)


def gender_from_sample_id(sample_id: str, metadata: dict[str, dict[str, str]]) -> str:
    base_id = strip_instruction_sample_id(sample_id)
    if base_id in metadata:
        for key in ("Gender", "gender"):
            if metadata[base_id].get(key):
                return normalize_gender(metadata[base_id][key])
    return ""


def label_ids_from_names(names: list[str], row_order: list[str]) -> torch.Tensor:
    name_to_idx = {name: idx for idx, name in enumerate(row_order)}
    return torch.tensor([name_to_idx[name] for name in names], dtype=torch.long)


def ensure_accumulator(
    accumulators: dict[str, tuple[torch.Tensor, torch.Tensor]],
    name: str,
    row_order: list[str],
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if name not in accumulators:
        accumulators[name] = create_accumulator(
            num_experts=num_experts,
            device="cpu",
            row_order=row_order,
        )
    return accumulators[name]


def accumulate_named_usage(
    accumulators: dict[str, tuple[torch.Tensor, torch.Tensor]],
    name: str,
    row_order: list[str],
    expert_usage: torch.Tensor,
    label_names: list[str],
) -> None:
    if not label_names:
        return
    sums, counts = ensure_accumulator(accumulators, name, row_order, expert_usage.shape[-1])
    label_ids = label_ids_from_names(label_names, row_order)
    accumulate_expert_usage(
        sums=sums,
        counts=counts,
        expert_usage=expert_usage,
        label_ids=label_ids,
        row_order=row_order,
    )


def save_heatmaps(
    output_dir: Path,
    accumulators: dict[str, tuple[torch.Tensor, torch.Tensor]],
    adapter_top_k: int,
    dataset: str,
) -> None:
    specs = {
        "task": (TASK_HEATMAP_ORDER, "Instruction Task Expert Usage"),
        "environment": (ENV_HEATMAP_ORDER, "Environment Expert Usage"),
        "gender": (GENDER_HEATMAP_ORDER, "Gender Expert Usage"),
    }
    for name, (row_order, title) in specs.items():
        if name not in accumulators:
            continue
        sums, counts = accumulators[name]
        if not torch.sum(counts).item() > 0:
            continue
        save_accumulator(
            output_dir / f"expert_heatmap_{name}_stats.pt",
            sums,
            counts,
            row_order=row_order,
            highlight_top_k=adapter_top_k,
        )
        render_heatmap(
            matrix=finalize_heatmap_matrix(sums, counts),
            output_path=output_dir / f"expert_heatmap_{name}.png",
            title=f"{title}: {dataset}",
            row_order=row_order,
            highlight_top_k=adapter_top_k,
        )


def main() -> None:
    args = build_argparser().parse_args()
    train_config = load_train_config(args.train_config)
    decode_config = DecodeConfig.from_yaml(args.decode_config)
    if args.expert_heatmap is not None:
        decode_config.expert_heatmap = normalize_heatmap_policy(args.expert_heatmap)

    keys = load_keys(args.key_file)
    data_type = resolve_instruction_data_type(train_config, args.dataset)
    device = torch.device("cuda" if args.ngpu > 0 and torch.cuda.is_available() else "cpu")
    preprocess = InstructionAsrTask.build_preprocess_fn(train_config, train=False)
    metadata = read_metadata(args.dataset)

    dataloader = build_dataloader(
        config=train_config,
        dataset_name=args.dataset,
        keys=keys,
        decode_config=decode_config,
        data_type=data_type,
        preprocess=preprocess,
    )
    model = load_model_for_inference(
        config=train_config,
        model_file=args.model_file,
        device=device,
        data_type=data_type,
    )
    supports_expert_heatmap = bool(getattr(model, "supports_expert_heatmap", False))
    collect_expert_heatmap = decode_config.expert_heatmap != "false" and supports_expert_heatmap
    if decode_config.expert_heatmap == "true" and not supports_expert_heatmap:
        raise ValueError(
            f"expert_heatmap=true was requested, but {type(model).__name__} does not support expert heatmap"
        )

    logger.info(
        "Start instruction inference dataset={} num_utts={} data_type={} decode_mode={} batch_size={} max_new_tokens={} expert_heatmap={} collect_expert_heatmap={} device={}",
        args.dataset,
        len(keys),
        ",".join(data_type),
        decode_config.decode_mode,
        decode_config.batch_size,
        decode_config.max_new_tokens,
        decode_config.expert_heatmap,
        collect_expert_heatmap,
        device,
    )

    results: list[dict[str, str]] = []
    processed_count = 0
    heatmap_accumulators: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.inference_mode():
        for batch in dataloader:
            uttids = list(batch[0]) if isinstance(batch, (tuple, list)) and len(batch) >= 1 else []
            batch_results, expert_usage = decode_batch(
                model=model,
                batch=batch,
                decode_config=decode_config,
                token_converter=None,
                processed_count=processed_count,
                total_count=len(keys),
            )
            results.extend(batch_results)
            processed_count += len(batch_results)
            if (
                collect_expert_heatmap
                and expert_usage is not None
                and isinstance(batch, (tuple, list))
                and len(batch) >= 2
            ):
                expert_usage = expert_usage.detach().cpu()
                task_names = [task_name_from_sample_id(uttid) for uttid in uttids]
                accumulate_named_usage(
                    heatmap_accumulators,
                    name="task",
                    row_order=TASK_HEATMAP_ORDER,
                    expert_usage=expert_usage,
                    label_names=task_names,
                )

                env_indices = [idx for idx, task_name in enumerate(task_names) if task_name == "environment"]
                if env_indices:
                    env_pairs = [
                        (idx, environment_from_sample_id(uttids[idx], metadata))
                        for idx in env_indices
                    ]
                    env_pairs = [(idx, name) for idx, name in env_pairs if name in ENV_HEATMAP_ORDER]
                    if env_pairs:
                        env_usage = expert_usage[[idx for idx, _ in env_pairs]]
                        accumulate_named_usage(
                            heatmap_accumulators,
                            name="environment",
                            row_order=ENV_HEATMAP_ORDER,
                            expert_usage=env_usage,
                            label_names=[name for _, name in env_pairs],
                        )

                gender_indices = [idx for idx, task_name in enumerate(task_names) if task_name == "gender"]
                if gender_indices:
                    gender_pairs = [
                        (idx, gender_from_sample_id(uttids[idx], metadata))
                        for idx in gender_indices
                    ]
                    gender_pairs = [
                        (idx, name) for idx, name in gender_pairs if name in GENDER_HEATMAP_ORDER
                    ]
                    if gender_pairs:
                        gender_usage = expert_usage[[idx for idx, _ in gender_pairs]]
                        accumulate_named_usage(
                            heatmap_accumulators,
                            name="gender",
                            row_order=GENDER_HEATMAP_ORDER,
                            expert_usage=gender_usage,
                            label_names=[name for _, name in gender_pairs],
                        )

    write_results(args.output_dir, results)
    if heatmap_accumulators:
        adapter_top_k = max(1, int(getattr(getattr(model, "adapter", None), "top_k", 2)))
        save_heatmaps(
            output_dir=Path(args.output_dir),
            accumulators=heatmap_accumulators,
            adapter_top_k=adapter_top_k,
            dataset=args.dataset,
        )
    elif decode_config.expert_heatmap == "true":
        raise ValueError(
            "expert_heatmap=true was requested, but no heatmap stats were collected. "
            "Check that the model returns expert_usage and instruction ids are available."
        )

    logger.info(
        "Finished instruction inference dataset={} model_file={} output_dir={} num_utts={}",
        args.dataset,
        args.model_file,
        args.output_dir,
        len(results),
    )


if __name__ == "__main__":
    main()
