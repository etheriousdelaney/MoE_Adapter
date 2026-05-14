from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch

from dataset.dataset import CHIME4_ENV_MAP


matplotlib.use("Agg")

LABEL_ORDER = ["BUS", "CAF", "PED", "STR"]
PROMPT_ORDER = ["TRANSCRIBE", "ENVIRONMENT", "GENDER", "OTHER"]
INSTRUCTION_LABEL_ORDER = ["TRANSCRIPT", "BUS", "CAF", "PED", "STR", "MALE", "FEMALE", "OTHER"]


def create_accumulator(
    num_experts: int,
    device: torch.device | str | None = None,
    row_order: list[str] | tuple[str, ...] = LABEL_ORDER,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_labels = len(row_order)
    sums = torch.zeros(num_labels, num_experts, dtype=torch.float32, device=device)
    counts = torch.zeros(num_labels, dtype=torch.float32, device=device)
    return sums, counts


def expert_usage_from_selected_experts(
    selected_experts: torch.Tensor,
    padding_mask: torch.Tensor | None,
    top_k: int,
) -> torch.Tensor:
    if selected_experts.ndim != 3:
        raise ValueError(
            f"selected_experts must have shape (B, T, E), but got {tuple(selected_experts.shape)}"
        )

    if padding_mask is None:
        mask = torch.ones(
            selected_experts.shape[:2],
            dtype=selected_experts.dtype,
            device=selected_experts.device,
        )
    else:
        mask = padding_mask.to(dtype=selected_experts.dtype)

    valid_counts = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    usage = (selected_experts * mask.unsqueeze(-1)).sum(dim=1) / valid_counts
    return usage


def accumulate_expert_usage(
    sums: torch.Tensor,
    counts: torch.Tensor,
    expert_usage: torch.Tensor,
    label_ids: torch.Tensor,
    row_order: list[str] | tuple[str, ...] = LABEL_ORDER,
) -> None:
    usage = expert_usage.detach().to(device=sums.device, dtype=torch.float32)
    labels = label_ids.detach().to(device=sums.device).view(-1).long()
    if usage.ndim != 2:
        raise ValueError(f"expert_usage must have shape (B, E), but got {tuple(usage.shape)}")
    if usage.shape[0] != labels.shape[0]:
        raise ValueError(
            f"expert_usage batch size {usage.shape[0]} != label batch size {labels.shape[0]}"
        )

    for label_idx in range(len(row_order)):
        mask = labels == label_idx
        if torch.any(mask):
            sums[label_idx] += usage[mask].sum(dim=0)
            counts[label_idx] += mask.sum()


def finalize_heatmap_matrix(
    sums: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    matrix = torch.zeros_like(sums, dtype=torch.float32)
    valid = counts > 0
    if torch.any(valid):
        matrix[valid] = sums[valid] / counts[valid].unsqueeze(-1)
    return matrix


def save_accumulator(
    path: str | Path,
    sums: torch.Tensor,
    counts: torch.Tensor,
    row_order: list[str] | tuple[str, ...] = LABEL_ORDER,
    highlight_top_k: int = 2,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sums": sums.detach().cpu(),
            "counts": counts.detach().cpu(),
            "label_order": list(row_order),
            "highlight_top_k": int(highlight_top_k),
        },
        output_path,
    )


def load_accumulator(path: str | Path) -> tuple[torch.Tensor, torch.Tensor, list[str], int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return (
        payload["sums"].float(),
        payload["counts"].float(),
        list(payload["label_order"]),
        int(payload.get("highlight_top_k", 2)),
    )


def merge_accumulators(paths: list[str | Path]) -> tuple[torch.Tensor, torch.Tensor, list[str], int]:
    total_sums = None
    total_counts = None
    row_order = None
    highlight_top_k = None
    for path in paths:
        sums, counts, current_row_order, current_highlight_top_k = load_accumulator(path)
        if total_sums is None:
            total_sums = sums
            total_counts = counts
            row_order = current_row_order
            highlight_top_k = current_highlight_top_k
        else:
            if current_row_order != row_order:
                raise ValueError(f"Accumulator label order mismatch: {current_row_order} != {row_order}")
            total_sums += sums
            total_counts += counts
            highlight_top_k = max(int(highlight_top_k), int(current_highlight_top_k))
    if total_sums is None or total_counts is None or row_order is None or highlight_top_k is None:
        raise ValueError("No accumulator files were provided")
    return total_sums, total_counts, row_order, int(highlight_top_k)


def render_heatmap(
    matrix: torch.Tensor,
    output_path: str | Path,
    title: str,
    row_order: list[str] | tuple[str, ...] = LABEL_ORDER,
    highlight_top_k: int = 2,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_cpu = matrix.detach().cpu()

    plt.figure(figsize=(11, 4.5))
    plt.imshow(matrix_cpu.numpy() * 100.0, cmap="viridis", aspect="auto", vmin=0.0, vmax=100.0)
    plt.title(title)
    plt.xlabel("Expert ID")
    plt.ylabel("Label")
    plt.xticks(range(matrix_cpu.shape[1]), [str(i) for i in range(matrix_cpu.shape[1])])
    plt.yticks(range(matrix_cpu.shape[0]), list(row_order))

    for row in range(matrix_cpu.shape[0]):
        top_indices = torch.argsort(matrix_cpu[row])[-max(1, highlight_top_k) :].tolist()
        for col in range(matrix_cpu.shape[1]):
            text_color = "red" if col in top_indices else "white"
            plt.text(
                col,
                row,
                f"{matrix_cpu[row, col].item() * 100.0:.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold",
            )

    cbar = plt.colorbar()
    cbar.set_label("Usage Rate (%)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def maybe_append_chime4_label(data_type: list[str], dataset_name: str) -> list[str]:
    updated = list(data_type)
    if dataset_name.startswith("chime4/") and "chime4_label" not in updated:
        updated.append("chime4_label")
    return updated


def label_id_to_name(label_id: int) -> str:
    for label_name, idx in CHIME4_ENV_MAP.items():
        if idx == label_id:
            return label_name
    raise KeyError(label_id)


def classify_prompt_text(prompt_text: str) -> str:
    normalized = prompt_text.strip().upper()
    if "TRANSCRIBE" in normalized:
        return "TRANSCRIBE"
    if "ENVIRONMENT" in normalized:
        return "ENVIRONMENT"
    if "GENDER" in normalized:
        return "GENDER"
    return "OTHER"


def classify_instruction_label(prompt_text: str, response_text: str) -> str:
    prompt_type = classify_prompt_text(prompt_text)
    normalized_response = response_text.strip().upper()
    if prompt_type == "TRANSCRIBE":
        return "TRANSCRIPT"
    if prompt_type == "ENVIRONMENT":
        aliases = {
            "BUS": "BUS",
            "BUSY": "BUS",
            "CAFE": "CAF",
            "CAF": "CAF",
            "PEDESTRIAN": "PED",
            "PED": "PED",
            "STREET": "STR",
            "STR": "STR",
        }
        for key, value in aliases.items():
            if key in normalized_response:
                return value
        return "OTHER"
    if prompt_type == "GENDER":
        if "FEMALE" in normalized_response:
            return "FEMALE"
        if "MALE" in normalized_response:
            return "MALE"
    return "OTHER"


def texts_to_category_ids(
    texts: list[str],
    row_order: list[str] | tuple[str, ...],
    classifier,
) -> torch.Tensor:
    name_to_idx = {name: idx for idx, name in enumerate(row_order)}
    category_ids = [name_to_idx[classifier(text)] for text in texts]
    return torch.tensor(category_ids, dtype=torch.long)


def prompt_texts_to_ids(prompt_texts: list[str]) -> torch.Tensor:
    return texts_to_category_ids(
        texts=prompt_texts,
        row_order=PROMPT_ORDER,
        classifier=classify_prompt_text,
    )


def instruction_label_texts_to_ids(prompt_texts: list[str], response_texts: list[str]) -> torch.Tensor:
    name_to_idx = {name: idx for idx, name in enumerate(INSTRUCTION_LABEL_ORDER)}
    label_ids = [
        name_to_idx[classify_instruction_label(prompt_text, response_text)]
        for prompt_text, response_text in zip(prompt_texts, response_texts)
    ]
    return torch.tensor(label_ids, dtype=torch.long)
