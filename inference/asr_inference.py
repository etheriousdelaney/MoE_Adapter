from __future__ import annotations

import argparse
import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader

from inference.expert_heatmap_utils import (
    accumulate_expert_usage,
    create_accumulator,
    finalize_heatmap_matrix,
    maybe_append_chime4_label,
    render_heatmap,
    save_accumulator,
)
from dataset.collate_fn import CommonCollateFn
from dataset.dataset import Dataset
from text.token_id_converter import TokenIDConverter
from train.config import (
    AdapterConfig,
    AsrDecoderConfig,
    ClassifierConfig,
    DatasetConfig,
    LlmDecoderConfig,
    ModelConfig,
    OptimizerConfig,
    QFormerConfig,
    TrainConfig,
)
from train.model_factory import build_model_from_config
from train.trainer import build_parser


@dataclass
class DecodeConfig:
    decode_mode: str = "greedy"
    batch_size: int = 1
    beam_size: int = 8
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
        if decode_mode not in {"greedy", "beam"}:
            raise ValueError(f"decode_mode must be 'greedy' or 'beam': {decode_mode}")
        expert_heatmap = normalize_heatmap_policy(payload.get("expert_heatmap", "auto"))
        nbest = int(payload.get("nbest", 1))
        if nbest != 1:
            raise ValueError("Only nbest=1 is supported in the current implementation")
        return cls(
            decode_mode=decode_mode,
            batch_size=int(payload.get("batch_size", 1)),
            beam_size=int(payload.get("beam_size", 8)),
            nbest=nbest,
            num_workers=int(payload.get("num_workers", 0)),
            max_new_tokens=int(payload.get("max_new_tokens", 128)),
            expert_heatmap=expert_heatmap,
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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ASR inference for MoE_Adapter",
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


def load_train_config(config_path: str | Path) -> TrainConfig:
    torch.serialization.add_safe_globals(
        [
            TrainConfig,
            ModelConfig,
            OptimizerConfig,
            QFormerConfig,
            DatasetConfig,
            AdapterConfig,
            ClassifierConfig,
            AsrDecoderConfig,
            LlmDecoderConfig,
        ]
    )
    parser = build_parser()
    args, _ = parser.parse_known_args(["--config", str(config_path)])
    return TrainConfig.from_namespace(args)


def resolve_inference_data_type(
    config: TrainConfig,
    dataset: str,
    expert_heatmap: str = "auto",
) -> list[str]:
    data_dir = Path("data") / dataset
    preferred = list(config.dataset.data_type)

    def maybe_with_heatmap_label(data_type: list[str]) -> list[str]:
        if expert_heatmap == "false":
            return data_type
        return maybe_append_chime4_label(data_type, dataset)

    is_instruction_task = (
        config.task == "instruction_asr"
        or "prompt" in preferred
        or "response" in preferred
    )
    if is_instruction_task and (data_dir / "fused.scp").exists() and (data_dir / "response.jsonl").exists():
        instruction_types = [name for name in preferred if name in {"fused", "prompt", "response"}]
        if "fused" not in instruction_types:
            instruction_types.insert(0, "fused")
        return maybe_with_heatmap_label(instruction_types)

    if "fused" in preferred and (data_dir / "fused.scp").exists():
        return maybe_with_heatmap_label(["fused"])
    if "sound" in preferred and (data_dir / "wav.scp").exists():
        return maybe_with_heatmap_label(["sound"])
    if (data_dir / "fused.scp").exists():
        return maybe_with_heatmap_label(["fused"])
    if (data_dir / "wav.scp").exists():
        return maybe_with_heatmap_label(["sound"])
    raise FileNotFoundError(f"Could not infer inference input type under {data_dir}")


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
    load_state_for_inference(
        model=model,
        state_dict=state_dict,
        checkpoint_format=checkpoint_format,
        data_type=data_type,
    )
    validate_asr_inference_model(model)
    model = model.to(device)
    model.eval()
    return model


def load_inference_state_dict(
    model_file: str | Path,
) -> tuple[dict[str, torch.Tensor], str]:
    model_file = Path(model_file)
    if model_file.suffix == ".ckpt":
        checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
        model_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                model_state_dict[key[len("model.") :]] = value
            else:
                model_state_dict[key] = value
        return model_state_dict, "ckpt"
    if model_file.suffix == ".pth":
        state_dict = torch.load(model_file, map_location="cpu", weights_only=False)
        return state_dict, "pth"
    raise ValueError(f"Unsupported model file suffix: {model_file}")


def load_state_for_inference(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    checkpoint_format: str,
    data_type: list[str],
) -> None:
    load_hook = getattr(model, "load_for_inference", None)
    if callable(load_hook):
        load_hook(
            state_dict=state_dict,
            checkpoint_format=checkpoint_format,
            data_type=list(data_type),
        )
        return
    model.load_state_dict(state_dict, strict=True)


def validate_asr_inference_model(model: torch.nn.Module) -> None:
    supports_ctc = callable(getattr(model, "compute_log_probs", None)) and hasattr(model, "blank_id")
    supports_generation = callable(getattr(model, "generate_greedy", None)) and callable(
        getattr(model, "ids_to_text", None)
    )
    if not supports_ctc and not supports_generation:
        raise TypeError(
            f"Model {type(model).__name__} does not support ASR inference: "
            "missing either CTC decode hooks or generation decode hooks"
        )


def get_token_converter(config: TrainConfig, model: torch.nn.Module) -> TokenIDConverter:
    tokens = getattr(model, "tokens", None)
    if tokens is not None:
        return TokenIDConverter(tokens)
    if config.token_list:
        return TokenIDConverter(config.token_list)
    raise ValueError(
        f"Model {type(model).__name__} does not provide token metadata for ASR inference"
    )


def supports_ctc_inference(model: torch.nn.Module) -> bool:
    return callable(getattr(model, "compute_log_probs", None)) and hasattr(model, "blank_id")


def supports_generation_inference(model: torch.nn.Module) -> bool:
    return callable(getattr(model, "generate_greedy", None)) and callable(
        getattr(model, "ids_to_text", None)
    )


def ids_to_text(token_converter: TokenIDConverter, ids: Iterable[int]) -> tuple[list[str], str]:
    tokens = []
    for token in token_converter.ids2tokens(ids):
        if token == "<space>":
            tokens.append(" ")
        elif token.startswith("<") and token.endswith(">"):
            continue
        else:
            tokens.append(token)
    text = "".join(tokens).strip()
    token_items = ["<space>" if token == " " else token for token in tokens]
    return token_items, text


def ctc_greedy_decode(
    log_probs: torch.Tensor,
    input_length: int,
    blank_id: int,
) -> tuple[list[int], float]:
    frame_ids = log_probs[:input_length].argmax(dim=-1).tolist()
    collapsed: list[int] = []
    score = 0.0
    prev = None
    for frame_idx, token_id in enumerate(frame_ids):
        score += float(log_probs[frame_idx, token_id].item())
        if token_id == blank_id:
            prev = None
            continue
        if prev == token_id:
            continue
        collapsed.append(int(token_id))
        prev = token_id
    return collapsed, score


def log_add(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def ctc_prefix_beam_decode(
    log_probs: torch.Tensor,
    input_length: int,
    blank_id: int,
    beam_size: int,
) -> tuple[list[int], float]:
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -math.inf)}
    for t in range(input_length):
        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}
        step = log_probs[t]
        topk = torch.topk(step, k=min(beam_size, step.shape[-1]))
        candidate_ids = topk.indices.tolist()
        if blank_id not in candidate_ids:
            candidate_ids.append(blank_id)

        for prefix, (pb, pnb) in beams.items():
            prefix_total = log_add(pb, pnb)
            for token_id in candidate_ids:
                token_logp = float(step[token_id].item())
                if token_id == blank_id:
                    cur_pb, cur_pnb = next_beams.get(prefix, (-math.inf, -math.inf))
                    next_beams[prefix] = (log_add(cur_pb, prefix_total + token_logp), cur_pnb)
                    continue

                last = prefix[-1] if prefix else None
                new_prefix = prefix + (token_id,)
                if token_id == last:
                    cur_pb, cur_pnb = next_beams.get(prefix, (-math.inf, -math.inf))
                    next_beams[prefix] = (cur_pb, log_add(cur_pnb, pnb + token_logp))

                    cur_pb, cur_pnb = next_beams.get(new_prefix, (-math.inf, -math.inf))
                    next_beams[new_prefix] = (cur_pb, log_add(cur_pnb, pb + token_logp))
                else:
                    cur_pb, cur_pnb = next_beams.get(new_prefix, (-math.inf, -math.inf))
                    next_beams[new_prefix] = (cur_pb, log_add(cur_pnb, prefix_total + token_logp))

        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda item: log_add(item[1][0], item[1][1]),
                reverse=True,
            )[:beam_size]
        )

    best_prefix, (best_pb, best_pnb) = max(
        beams.items(),
        key=lambda item: log_add(item[1][0], item[1][1]),
    )
    return list(best_prefix), log_add(best_pb, best_pnb)


def decode_batch(
    model: torch.nn.Module,
    batch,
    decode_config: DecodeConfig,
    token_converter: TokenIDConverter | None,
    processed_count: int,
    total_count: int,
) -> tuple[list[dict[str, str]], torch.Tensor | None]:
    uttids, batch_data = batch
    expert_usage = None

    prompt_texts: list[str] | None = None
    prompt_ids = batch_data.get("prompt") if isinstance(batch_data, dict) else None
    prompt_lengths = batch_data.get("prompt_lengths") if isinstance(batch_data, dict) else None
    if prompt_ids is not None and prompt_lengths is not None:
        tokenizer = getattr(getattr(model, "decoder", None), "tokenizer", None)
        if tokenizer is not None:
            prompt_texts = []
            special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
            for row, length in zip(prompt_ids, prompt_lengths):
                seq = [int(token_id) for token_id in row[: int(length.item())].tolist()]
                seq = [token_id for token_id in seq if token_id not in special_ids]
                prompt_texts.append(tokenizer.decode(seq, skip_special_tokens=True).strip())
    elif hasattr(model, "inference_prompt_text"):
        prompt_texts = [str(getattr(model, "inference_prompt_text")) for _ in uttids]

    results = []
    if supports_ctc_inference(model):
        assert token_converter is not None
        log_probs, input_lengths = model.compute_log_probs(batch)
        get_last_expert_usage = getattr(model, "get_last_expert_usage", None)
        if callable(get_last_expert_usage):
            expert_usage = get_last_expert_usage()
            if expert_usage is not None:
                expert_usage = expert_usage.detach().cpu()
        log_probs = log_probs.detach().cpu()
        input_lengths = input_lengths.detach().cpu().tolist()
        for batch_idx, uttid in enumerate(uttids):
            sample_log_probs = log_probs[batch_idx]
            sample_length = int(input_lengths[batch_idx])
            current_index = processed_count + batch_idx + 1
            if decode_config.decode_mode == "greedy":
                token_ids, score = ctc_greedy_decode(
                    sample_log_probs,
                    input_length=sample_length,
                    blank_id=model.blank_id,
                )
            else:
                token_ids, score = ctc_prefix_beam_decode(
                    sample_log_probs,
                    input_length=sample_length,
                    blank_id=model.blank_id,
                    beam_size=decode_config.beam_size,
                )

            token_items, text = ids_to_text(token_converter, token_ids)
            logger.info(
                "decode progress {}/{} uttid={} input_length={} output_tokens={} mode={} score={:.4f}",
                current_index,
                total_count,
                uttid,
                sample_length,
                len(token_ids),
                decode_config.decode_mode,
                score,
            )
            results.append(
                {
                    "uttid": uttid,
                    "text": text,
                    "prompt": prompt_texts[batch_idx] if prompt_texts is not None else "",
                    "token": " ".join(token_items),
                    "token_int": " ".join(str(token_id) for token_id in token_ids),
                    "score": f"{score:.6f}",
                }
            )
        return results, expert_usage

    if not supports_generation_inference(model):
        raise TypeError(f"Unsupported inference model type: {type(model).__name__}")
    if decode_config.decode_mode != "greedy":
        raise ValueError("Qwen decoder inference currently supports only decode_mode=greedy")

    generated_ids, scores, expert_usage = model.generate_greedy(
        batch,
        max_new_tokens=decode_config.max_new_tokens,
    )
    if expert_usage is not None:
        expert_usage = expert_usage.detach().cpu()
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
                "prompt": prompt_texts[batch_idx] if prompt_texts is not None else "",
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


def main() -> None:
    args = build_argparser().parse_args()
    train_config = load_train_config(args.train_config)
    decode_config = DecodeConfig.from_yaml(args.decode_config)
    if args.expert_heatmap is not None:
        decode_config.expert_heatmap = normalize_heatmap_policy(args.expert_heatmap)
    keys = load_keys(args.key_file)
    data_type = resolve_inference_data_type(
        train_config,
        args.dataset,
        expert_heatmap=decode_config.expert_heatmap,
    )
    device = torch.device("cuda" if args.ngpu > 0 and torch.cuda.is_available() else "cpu")

    dataloader = build_dataloader(
        config=train_config,
        dataset_name=args.dataset,
        keys=keys,
        decode_config=decode_config,
        data_type=data_type,
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
    token_converter = get_token_converter(train_config, model) if supports_ctc_inference(model) else None
    logger.info(
        "Start ASR inference dataset={} num_utts={} data_type={} decode_mode={} batch_size={} beam_size={} max_new_tokens={} expert_heatmap={} collect_expert_heatmap={} device={}",
        args.dataset,
        len(keys),
        ",".join(data_type),
        decode_config.decode_mode,
        decode_config.batch_size,
        decode_config.beam_size,
        decode_config.max_new_tokens,
        decode_config.expert_heatmap,
        collect_expert_heatmap,
        device,
    )

    results: list[dict[str, str]] = []
    processed_count = 0
    heatmap_sums = None
    heatmap_counts = None
    with torch.inference_mode():
        for batch in dataloader:
            batch_results, expert_usage = decode_batch(
                model=model,
                batch=batch,
                decode_config=decode_config,
                token_converter=token_converter,
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
                batch_data = batch[1]
                labels = batch_data.get("chime4_label")
                if labels is not None:
                    if heatmap_sums is None or heatmap_counts is None:
                        heatmap_sums, heatmap_counts = create_accumulator(
                            num_experts=expert_usage.shape[-1],
                            device="cpu",
                        )
                    accumulate_expert_usage(
                        sums=heatmap_sums,
                        counts=heatmap_counts,
                        expert_usage=expert_usage,
                        label_ids=labels.detach().cpu(),
                    )

    write_results(args.output_dir, results)
    if heatmap_sums is not None and heatmap_counts is not None and torch.sum(heatmap_counts).item() > 0:
        stats_path = Path(args.output_dir) / "expert_heatmap_stats.pt"
        adapter_top_k = max(1, int(getattr(getattr(model, "adapter", None), "top_k", 2)))
        save_accumulator(
            stats_path,
            heatmap_sums,
            heatmap_counts,
            highlight_top_k=adapter_top_k,
        )
        render_heatmap(
            matrix=finalize_heatmap_matrix(heatmap_sums, heatmap_counts),
            output_path=Path(args.output_dir) / "expert_heatmap.png",
            title=f"ASR Inference Expert Usage: {args.dataset}",
            highlight_top_k=adapter_top_k,
        )
    elif decode_config.expert_heatmap == "true":
        raise ValueError(
            "expert_heatmap=true was requested, but no heatmap stats were collected. "
            "Check that the dataset provides chime4_label and the model returns expert_usage."
        )
    logger.info(
        "Finished ASR inference dataset={} model_file={} output_dir={} num_utts={}",
        args.dataset,
        args.model_file,
        args.output_dir,
        len(results),
    )


if __name__ == "__main__":
    main()
