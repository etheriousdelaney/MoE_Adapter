from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import re

import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader

from dataset.collate_fn import CommonCollateFn
from dataset.dataset import Dataset
from dataset.instruction_utils import strip_instruction_sample_id
from inference.expert_heatmap_utils import (
    accumulate_expert_usage,
    create_accumulator,
    finalize_heatmap_matrix,
    render_heatmap,
    save_accumulator,
)
from task.instruction_asr import InstructionTask
from train.config import (
    AdapterConfig,
    DatasetConfig,
    LlmDecoderConfig,
    ModelConfig,
    OptimizerConfig,
    QFormerConfig,
    TrainConfig,
)
from train.model_factory import build_model_from_config
from train.trainer import build_parser

DEFAULT_TASK_HEATMAP_ORDER = ["asr", "environment", "gender"]
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


@dataclass
class DecodeConfig:
    decode_mode: str = "greedy"
    batch_size: int = 1
    beam_size: int = 1
    nbest: int = 1
    num_workers: int = 0
    max_new_tokens: int = 128
    expert_heatmap: str = "auto"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DecodeConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        allowed = {
            "decode_mode",
            "batch_size",
            "beam_size",
            "nbest",
            "num_workers",
            "max_new_tokens",
            "expert_heatmap",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown decode config keys: {unknown}")
        decode_mode = str(payload.get("decode_mode", "greedy"))
        if decode_mode != "greedy":
            raise ValueError(f"instruction inference only supports decode_mode=greedy: {decode_mode}")
        nbest = int(payload.get("nbest", 1))
        if nbest != 1:
            raise ValueError("Only nbest=1 is supported")
        return cls(
            decode_mode=decode_mode,
            batch_size=int(payload.get("batch_size", 1)),
            beam_size=int(payload.get("beam_size", 1)),
            nbest=nbest,
            num_workers=int(payload.get("num_workers", 0)),
            max_new_tokens=int(payload.get("max_new_tokens", 128)),
            expert_heatmap=normalize_heatmap_policy(payload.get("expert_heatmap", "auto")),
        )


def normalize_heatmap_policy(value: str | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    normalized = str(value).lower()
    if normalized in {"auto", "true", "false"}:
        return normalized
    if normalized in {"1", "yes", "y"}:
        return "true"
    if normalized in {"0", "no", "n"}:
        return "false"
    raise ValueError(f"expert_heatmap must be auto, true, or false: {value}")


def load_train_config(config_path: str | Path) -> TrainConfig:
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals(
            [
                TrainConfig,
                ModelConfig,
                OptimizerConfig,
                QFormerConfig,
                DatasetConfig,
                AdapterConfig,
                LlmDecoderConfig,
            ]
        )
    parser = build_parser()
    args, _ = parser.parse_known_args(["--config", str(config_path)])
    return TrainConfig.from_namespace(args)


def load_keys(key_file: str | Path) -> list[str]:
    with Path(key_file).open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_dataloader(
    config: TrainConfig,
    dataset_name: str,
    keys: list[str],
    decode_config: DecodeConfig,
    data_type: list[str],
    preprocess=None,
) -> DataLoader:
    dataset = Dataset(
        dataset_name,
        float_dtype=config.dataset.train_dtype,
        preprocess=preprocess,
        max_cache_size=0.0,
        max_cache_fd=config.dataset.max_cache_fd,
        allow_multi_rates=config.dataset.allow_multi_rates,
        keys_to_load=set(keys),
        data_type=data_type,
        message_file=(
            config.dataset.train_message_file
            if dataset_name == config.dataset.train_data
            else config.dataset.valid_message_file
        ),
        instruction_source=config.dataset.instruction_source,
        instruction_tasks=config.dataset.instruction_tasks,
    )
    batch_sampler = [
        keys[start : start + decode_config.batch_size]
        for start in range(0, len(keys), decode_config.batch_size)
    ]
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=CommonCollateFn(float_pad_value=0.0, int_pad_value=-1),
        num_workers=decode_config.num_workers,
        pin_memory=config.ngpu > 0,
    )


def load_model_for_inference(
    config: TrainConfig,
    model_file: str | Path,
    device: torch.device,
    data_type: list[str],
) -> torch.nn.Module:
    model_file = Path(model_file)
    inference_config = copy.deepcopy(config)
    inference_config.dataset.data_type = list(data_type)
    model = build_model_from_config(config=inference_config, data_type=data_type)
    state_dict, checkpoint_format = load_inference_state_dict(model_file)
    load_hook = getattr(model, "load_for_inference", None)
    if callable(load_hook):
        load_hook(
            state_dict=state_dict,
            checkpoint_format=checkpoint_format,
            data_type=list(data_type),
        )
    else:
        model.load_state_dict(state_dict, strict=True)
    if not callable(getattr(model, "generate_greedy", None)) or not callable(getattr(model, "ids_to_text", None)):
        raise TypeError(f"Model {type(model).__name__} does not support instruction generation inference")
    model = model.to(device)
    model.eval()
    return model


def load_inference_state_dict(model_file: str | Path) -> tuple[dict[str, torch.Tensor], str]:
    model_file = Path(model_file)
    if model_file.suffix == ".ckpt":
        checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
        return {
            key[len("model.") :] if key.startswith("model.") else key: value
            for key, value in state_dict.items()
        }, "ckpt"
    if model_file.suffix == ".pth":
        return torch.load(model_file, map_location="cpu", weights_only=False), "pth"
    raise ValueError(f"Unsupported model file suffix: {model_file}")


def decode_batch(
    model: torch.nn.Module,
    batch,
    decode_config: DecodeConfig,
    processed_count: int,
    total_count: int,
) -> tuple[list[dict[str, str]], torch.Tensor | None]:
    uttids, _ = batch
    generated_ids, scores, expert_usage = model.generate_greedy(
        batch,
        max_new_tokens=decode_config.max_new_tokens,
    )
    if expert_usage is not None:
        expert_usage = expert_usage.detach().cpu()

    results = []
    for batch_idx, uttid in enumerate(uttids):
        token_ids = generated_ids[batch_idx]
        score = float(scores[batch_idx])
        token_items, text = model.ids_to_text(token_ids)
        current_index = processed_count + batch_idx + 1
        logger.info(
            "decode progress {}/{} uttid={} output_tokens={} mode={} score={:.4f}",
            current_index,
            total_count,
            uttid,
            len(token_ids),
            decode_config.decode_mode,
            score,
        )
        results.append(
            {
                "uttid": uttid,
                "text": text,
                "prompt": "",
                "token": " ".join(token_items),
                "token_int": " ".join(str(token_id) for token_id in token_ids),
                "score": f"{score:.6f}",
            }
        )
    return results, expert_usage


def write_results(output_dir: str | Path, results: list[dict[str, str]]) -> None:
    recog_dir = Path(output_dir) / "1best_recog"
    recog_dir.mkdir(parents=True, exist_ok=True)
    field_to_path = {
        "text": recog_dir / "text",
        "prompt": recog_dir / "prompt",
        "token": recog_dir / "token",
        "token_int": recog_dir / "token_int",
        "score": recog_dir / "score",
    }
    sorted_results = sorted(results, key=lambda item: item["uttid"])
    for field, path in field_to_path.items():
        with path.open("w", encoding="utf-8") as f:
            for item in sorted_results:
                value = item[field]
                if value:
                    f.write(f"{item['uttid']} {value}\n")
                else:
                    f.write(f"{item['uttid']}\n")


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
    text_types = [name for name in preferred if name == "audio_context"]
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


def instruction_task_names(train_config) -> list[str]:
    tasks = list(getattr(train_config.dataset, "instruction_tasks", None) or [])
    names = [str(task.get("name") or f"task_{idx}") for idx, task in enumerate(tasks)]
    return names or DEFAULT_TASK_HEATMAP_ORDER


def task_name_from_sample_id(sample_id: str, task_names: list[str]) -> str:
    index = sample_index(sample_id)
    if 0 <= index < len(task_names):
        return task_names[index]
    return task_names[0] if task_names else "task_0"


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
    task_row_order: list[str],
) -> None:
    specs = {
        "task": (task_row_order, "Instruction Task Expert Usage"),
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
    preprocess = InstructionTask.build_preprocess_fn(train_config, train=False)
    metadata = read_metadata(args.dataset)
    task_row_order = instruction_task_names(train_config)

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
                task_names = [task_name_from_sample_id(uttid, task_row_order) for uttid in uttids]
                accumulate_named_usage(
                    heatmap_accumulators,
                    name="task",
                    row_order=task_row_order,
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
            task_row_order=task_row_order,
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
