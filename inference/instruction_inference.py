from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger

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
    text_types = [name for name in preferred if name in {"prompt", "response"}]
    if (data_dir / "fused.scp").exists():
        return ["fused", *text_types]
    if (data_dir / "wav.scp").exists():
        return ["sound", *text_types]
    raise FileNotFoundError(
        f"Could not infer instruction inference input type under {data_dir}: "
        "expected fused.scp or wav.scp"
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
    heatmap_sums = None
    heatmap_counts = None
    with torch.inference_mode():
        for batch in dataloader:
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
            title=f"Instruction Inference Expert Usage: {args.dataset}",
            highlight_top_k=adapter_top_k,
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
