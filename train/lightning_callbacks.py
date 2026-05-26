from __future__ import annotations

import json
from pathlib import Path
import re

import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from dataset.instruction_utils import strip_instruction_sample_id
from inference.expert_heatmap_utils import (
    accumulate_expert_usage,
    create_accumulator,
    finalize_heatmap_matrix,
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
        train_config=None,
        highlight_top_k: int = 2,
    ):
        self.output_dir = Path(output_dir)
        self.train_config = train_config
        self.highlight_top_k = highlight_top_k
        self._states: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]] | None] = {
            "train": None,
            "valid": None,
        }
        self._metadata: dict[str, dict[str, dict[str, str]]] = {"train": {}, "valid": {}}
        self.task_row_order = self._task_row_order(train_config)

    @staticmethod
    def _task_row_order(train_config) -> list[str]:
        dataset_config = getattr(train_config, "dataset", None)
        tasks = list(getattr(dataset_config, "instruction_tasks", None) or [])
        names = [str(task.get("name") or f"task_{idx}") for idx, task in enumerate(tasks)]
        return names or ["asr", "environment", "gender"]

    def _reset(self, split: str, pl_module) -> None:
        num_experts = int(pl_module.model.adapter.num_experts)
        self._states[split] = {
            "task": create_accumulator(num_experts, device=pl_module.device, row_order=self.task_row_order),
            "environment": create_accumulator(
                num_experts,
                device=pl_module.device,
                row_order=["BUS", "CAFE", "PEDESTRIAN", "STREET"],
            ),
            "gender": create_accumulator(num_experts, device=pl_module.device, row_order=["female", "male"]),
        }
        self._metadata[split] = self._read_split_metadata(split, pl_module)

    def _read_split_metadata(self, split: str, pl_module) -> dict[str, dict[str, str]]:
        dataset_config = getattr(getattr(pl_module, "config", None), "dataset", None)
        if dataset_config is None:
            return {}
        data_name = dataset_config.train_data if split == "train" else dataset_config.valid_data
        metadata_path = Path("data") / data_name / "metadata.json"
        if not metadata_path.exists():
            return {}
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            items = payload.items()
        elif isinstance(payload, list):
            items = ((item.get("id"), item) for item in payload if isinstance(item, dict))
        else:
            return {}
        metadata: dict[str, dict[str, str]] = {}
        for key, item in items:
            if key is None or not isinstance(item, dict):
                continue
            metadata[str(key)] = {str(k): str(v) for k, v in item.items()}
        return metadata

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
        uttids = [str(uttid) for uttid in batch[0]]
        task_names = [self._task_name_from_sample_id(uttid) for uttid in uttids]
        self._accumulate_named_usage(
            split=split,
            name="task",
            expert_usage=expert_usage,
            label_names=task_names,
        )

        env_pairs = [
            (idx, self._environment_from_sample_id(uttid, split))
            for idx, uttid in enumerate(uttids)
        ]
        env_pairs = [(idx, name) for idx, name in env_pairs if name in {"BUS", "CAFE", "PEDESTRIAN", "STREET"}]
        if env_pairs:
            self._accumulate_named_usage(
                split=split,
                name="environment",
                expert_usage=expert_usage[[idx for idx, _ in env_pairs]],
                label_names=[name for _, name in env_pairs],
            )

        gender_pairs = [
            (idx, self._gender_from_sample_id(uttid, split))
            for idx, uttid in enumerate(uttids)
        ]
        gender_pairs = [(idx, name) for idx, name in gender_pairs if name in {"female", "male"}]
        if gender_pairs:
            self._accumulate_named_usage(
                split=split,
                name="gender",
                expert_usage=expert_usage[[idx for idx, _ in gender_pairs]],
                label_names=[name for _, name in gender_pairs],
            )

    def _accumulate_named_usage(
        self,
        split: str,
        name: str,
        expert_usage: torch.Tensor,
        label_names: list[str],
    ) -> None:
        if not label_names or self._states[split] is None:
            return
        sums, counts = self._states[split][name]
        row_order = self._row_order(name)
        name_to_idx = {label_name: idx for idx, label_name in enumerate(row_order)}
        label_ids = torch.tensor(
            [name_to_idx[label_name] for label_name in label_names],
            dtype=torch.long,
            device=expert_usage.device,
        )
        accumulate_expert_usage(
            sums=sums,
            counts=counts,
            expert_usage=expert_usage,
            label_ids=label_ids,
            row_order=row_order,
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
            ("task", self.task_row_order),
            ("environment", ["BUS", "CAFE", "PEDESTRIAN", "STREET"]),
            ("gender", ["female", "male"]),
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

    def _row_order(self, name: str) -> list[str]:
        if name == "task":
            return self.task_row_order
        if name == "environment":
            return ["BUS", "CAFE", "PEDESTRIAN", "STREET"]
        if name == "gender":
            return ["female", "male"]
        raise KeyError(name)

    def _task_name_from_sample_id(self, sample_id: str) -> str:
        match = re.search(r"__sample__(\d+)$", sample_id)
        sample_index = int(match.group(1)) if match else 0
        if 0 <= sample_index < len(self.task_row_order):
            return self.task_row_order[sample_index]
        return self.task_row_order[0] if self.task_row_order else "task_0"

    def _environment_from_sample_id(self, sample_id: str, split: str) -> str:
        base_id = strip_instruction_sample_id(sample_id)
        metadata = self._metadata.get(split, {})
        if base_id in metadata and metadata[base_id].get("environment"):
            return self._normalize_environment(metadata[base_id]["environment"])
        return self._normalize_environment(base_id)

    def _gender_from_sample_id(self, sample_id: str, split: str) -> str:
        base_id = strip_instruction_sample_id(sample_id)
        metadata = self._metadata.get(split, {})
        if base_id in metadata:
            for key in ("Gender", "gender"):
                if metadata[base_id].get(key):
                    return self._normalize_gender(metadata[base_id][key])
        return ""

    @staticmethod
    def _normalize_environment(text: str) -> str:
        normalized = text.strip().upper()
        aliases = {
            "BUS": "BUS",
            "CAFE": "CAFE",
            "CAF": "CAFE",
            "PEDESTRIAN": "PEDESTRIAN",
            "PED": "PEDESTRIAN",
            "STREET": "STREET",
            "STR": "STREET",
        }
        for key, value in aliases.items():
            if normalized == key or key in normalized:
                return value
        return normalized

    @staticmethod
    def _normalize_gender(text: str) -> str:
        normalized = text.strip().lower()
        if "female" in normalized:
            return "female"
        if "male" in normalized:
            return "male"
        return normalized
