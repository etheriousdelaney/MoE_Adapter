from __future__ import annotations

from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from inference.expert_heatmap_utils import (
    INSTRUCTION_LABEL_ORDER,
    LABEL_ORDER,
    PROMPT_ORDER,
    accumulate_expert_usage,
    classify_instruction_label,
    classify_prompt_text,
    create_accumulator,
    finalize_heatmap_matrix,
    instruction_label_texts_to_ids,
    prompt_texts_to_ids,
    render_heatmap,
)


class AverageCheckpointsCallback(Callback):
    def __init__(self, output_dir: str | Path, best_ckpt_callbacks):
        self.output_dir = Path(output_dir)
        self.best_ckpt_callbacks = list(best_ckpt_callbacks)

    def on_fit_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return

        for ckpt_callback in self.best_ckpt_callbacks:
            checkpoints = list(ckpt_callback.best_k_models.keys())
            if len(checkpoints) == 0:
                continue

            avg_state_dict = None
            for ckpt_path in checkpoints:
                # These checkpoints are generated locally by this training run, so
                # loading the full checkpoint payload is acceptable here.
                state_dict = torch.load(
                    ckpt_path,
                    map_location="cpu",
                    weights_only=False,
                )["state_dict"]
                if avg_state_dict is None:
                    avg_state_dict = state_dict
                else:
                    for key in avg_state_dict:
                        avg_state_dict[key] = avg_state_dict[key] + state_dict[key]

            for key in avg_state_dict:
                if not str(avg_state_dict[key].dtype).startswith("torch.int"):
                    avg_state_dict[key] = avg_state_dict[key] / len(checkpoints)

            new_avg_state_dict = {
                key.removeprefix("model."): value
                for key, value in avg_state_dict.items()
                if key.startswith("model.")
            }

            avg_ckpt_path = self.output_dir / (
                ckpt_callback.monitor.replace("/", ".") + f".ave_{len(checkpoints)}best.pth"
            )
            torch.save(new_avg_state_dict, avg_ckpt_path)


class ExpertHeatmapCallback(Callback):
    def __init__(
        self,
        output_dir: str | Path,
        highlight_top_k: int = 2,
    ):
        self.output_dir = Path(output_dir)
        self.highlight_top_k = highlight_top_k
        self._states: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]] | None] = {
            "train": None,
            "valid": None,
        }

    def _reset(self, split: str, pl_module) -> None:
        num_experts = int(pl_module.model.adapter.num_experts)
        self._states[split] = {
            "label": create_accumulator(num_experts, device=pl_module.device, row_order=LABEL_ORDER),
            "prompt": create_accumulator(num_experts, device=pl_module.device, row_order=PROMPT_ORDER),
            "instruction_label": create_accumulator(
                num_experts,
                device=pl_module.device,
                row_order=INSTRUCTION_LABEL_ORDER,
            ),
        }

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if not getattr(pl_module.model, "supports_expert_heatmap", False):
            return
        self._reset("train", pl_module)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        if not getattr(pl_module.model, "supports_expert_heatmap", False):
            return
        self._reset("valid", pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._accumulate("train", outputs, batch, pl_module)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if trainer.sanity_checking:
            return
        self._accumulate("valid", outputs, batch, pl_module)

    def _accumulate(self, split: str, outputs, batch, pl_module) -> None:
        if self._states[split] is None or not isinstance(outputs, dict):
            return
        expert_usage = outputs.get("expert_usage")
        if expert_usage is None:
            return
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            return
        batch_data = batch[1]
        labels = batch_data.get("chime4_label")
        if labels is not None:
            sums, counts = self._states[split]["label"]
            accumulate_expert_usage(
                sums=sums,
                counts=counts,
                expert_usage=expert_usage,
                label_ids=labels,
                row_order=LABEL_ORDER,
            )

        prompt_ids = batch_data.get("prompt")
        prompt_lengths = batch_data.get("prompt_lengths")
        response_ids = batch_data.get("response")
        response_lengths = batch_data.get("response_lengths")
        if (
            prompt_ids is None
            or prompt_lengths is None
            or response_ids is None
            or response_lengths is None
        ):
            return

        prompt_texts = self._decode_batch_texts(
            tokenizer=pl_module.model.decoder.tokenizer,
            token_ids=prompt_ids,
            lengths=prompt_lengths,
        )
        response_texts = self._decode_batch_texts(
            tokenizer=pl_module.model.decoder.tokenizer,
            token_ids=response_ids,
            lengths=response_lengths,
        )

        prompt_category_ids = prompt_texts_to_ids(prompt_texts).to(expert_usage.device)
        prompt_sums, prompt_counts = self._states[split]["prompt"]
        accumulate_expert_usage(
            sums=prompt_sums,
            counts=prompt_counts,
            expert_usage=expert_usage,
            label_ids=prompt_category_ids,
            row_order=PROMPT_ORDER,
        )

        instruction_label_ids = instruction_label_texts_to_ids(prompt_texts, response_texts).to(
            expert_usage.device
        )
        instruction_sums, instruction_counts = self._states[split]["instruction_label"]
        accumulate_expert_usage(
            sums=instruction_sums,
            counts=instruction_counts,
            expert_usage=expert_usage,
            label_ids=instruction_label_ids,
            row_order=INSTRUCTION_LABEL_ORDER,
        )

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._render("train", trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        self._render("valid", trainer, pl_module)

    def _reduce_state(self, sums: torch.Tensor, counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        return sums, counts

    @rank_zero_only
    def _write_heatmap(
        self,
        matrix: torch.Tensor,
        suffix: str,
        row_order: list[str] | tuple[str, ...],
        split: str,
        current_epoch: int,
        highlight_top_k: int,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{split}_expert_heatmap_{suffix}_epoch{current_epoch + 1:03d}.png"
        render_heatmap(
            matrix=matrix,
            output_path=output_path,
            title=f"{split.capitalize()} Expert Usage {suffix.replace('_', ' ').title()} Epoch {current_epoch + 1}",
            row_order=row_order,
            highlight_top_k=highlight_top_k,
        )

    def _render(self, split: str, trainer, pl_module) -> None:
        state = self._states.get(split)
        if state is None:
            return
        adapter_top_k = getattr(pl_module.model.adapter, "top_k", self.highlight_top_k)
        highlight_top_k = max(1, int(adapter_top_k))
        for suffix, row_order in (
            ("label", LABEL_ORDER),
            ("prompt", PROMPT_ORDER),
            ("instruction_label", INSTRUCTION_LABEL_ORDER),
        ):
            sums, counts = state[suffix]
            sums, counts = self._reduce_state(sums, counts)
            if torch.sum(counts).item() == 0:
                continue
            matrix = finalize_heatmap_matrix(sums, counts)
            self._write_heatmap(
                matrix=matrix,
                suffix=suffix,
                row_order=row_order,
                split=split,
                current_epoch=trainer.current_epoch,
                highlight_top_k=highlight_top_k,
            )

    @staticmethod
    def _decode_batch_texts(tokenizer, token_ids: torch.Tensor, lengths: torch.Tensor) -> list[str]:
        decoded: list[str] = []
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        for row, length in zip(token_ids, lengths):
            seq = [int(token_id) for token_id in row[: int(length.item())].tolist()]
            seq = [token_id for token_id in seq if token_id not in special_ids]
            decoded.append(tokenizer.decode(seq, skip_special_tokens=True).strip())
        return decoded
