from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize ASR scoring results into Markdown",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("exp_dir", type=str, help="Decode root directory")
    return parser


def safe_command_output(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def find_result_files(exp_dir: Path, score_type: str) -> list[Path]:
    pattern = f"**/score_{score_type}/result.txt"
    return sorted(exp_dir.glob(pattern))


def find_metric_result_files(exp_dir: Path, metric: str) -> list[Path]:
    return sorted(exp_dir.glob(f"**/score_*_{metric}/result.txt"))


def parse_avg_row(result_path: Path) -> list[str] | None:
    for line in result_path.read_text(encoding="utf-8").splitlines():
        if "Sum/Avg" not in line:
            continue
        cells = [cell.strip() for cell in line.split("|")]
        values = [cell for cell in cells if cell]
        if not values:
            continue
        if values[0] != "Sum/Avg":
            continue
        parsed: list[str] = []
        for cell in values[1:]:
            parsed.extend(cell.split())
        if len(parsed) >= 8:
            return parsed[:8]
    return None


def format_result_table(exp_dir: Path, score_type: str) -> list[str]:
    result_paths = find_result_files(exp_dir, score_type)
    if not result_paths:
        return []

    lines = [
        f"### {score_type.upper()}",
        "",
        "|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for result_path in result_paths:
        avg_values = parse_avg_row(result_path)
        if avg_values is None:
            continue
        dataset = result_path.parent.parent.relative_to(exp_dir).as_posix()
        row = "|".join([dataset] + avg_values)
        lines.append(f"|{row}|")
    lines.append("")
    return lines


def format_metric_table(exp_dir: Path, metric: str) -> list[str]:
    result_paths = find_metric_result_files(exp_dir, metric)
    if not result_paths:
        return []

    lines = [
        f"### {metric.upper()}",
        "",
        "|dataset|task|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for result_path in result_paths:
        avg_values = parse_avg_row(result_path)
        if avg_values is None:
            continue
        dataset = result_path.parent.parent.relative_to(exp_dir).as_posix()
        score_dir = result_path.parent.name
        task = score_dir.removeprefix("score_").removesuffix(f"_{metric}")
        row = "|".join([dataset, task] + avg_values)
        lines.append(f"|{row}|")
    lines.append("")
    return lines


def parse_accuracy_result(result_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in result_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "\t" not in line:
            continue
        key, value = line.split("\t", 1)
        values[key] = value
    return values


def format_accuracy_table(exp_dir: Path, task_name: str) -> list[str]:
    result_paths = sorted(exp_dir.glob(f"**/score_{task_name}_accuracy/result.txt"))
    if not result_paths:
        return []

    lines = [
        f"### {task_name.capitalize()} Accuracy",
        "",
        "|dataset|correct|total|accuracy %|",
        "|---|---|---|---|",
    ]
    for result_path in result_paths:
        values = parse_accuracy_result(result_path)
        dataset = result_path.parent.parent.relative_to(exp_dir).as_posix()
        lines.append(
            "|{}|{}|{}|{}|".format(
                dataset,
                values.get("correct", ""),
                values.get("total", ""),
                values.get("accuracy_percent", ""),
            )
        )
    lines.append("")
    return lines


def format_all_accuracy_tables(exp_dir: Path) -> list[str]:
    result_paths = sorted(exp_dir.glob("**/score_*_accuracy/result.txt"))
    if not result_paths:
        return []

    lines = [
        "### Accuracy",
        "",
        "|dataset|task|correct|total|accuracy %|",
        "|---|---|---|---|---|",
    ]
    for result_path in result_paths:
        values = parse_accuracy_result(result_path)
        dataset = result_path.parent.parent.relative_to(exp_dir).as_posix()
        score_dir = result_path.parent.name
        task = score_dir.removeprefix("score_").removesuffix("_accuracy")
        lines.append(
            "|{}|{}|{}|{}|{}|".format(
                dataset,
                task,
                values.get("correct", ""),
                values.get("total", ""),
                values.get("accuracy_percent", ""),
            )
        )
    lines.append("")
    return lines


def format_heatmap_paths(exp_dir: Path) -> list[str]:
    heatmaps = sorted(exp_dir.glob("**/expert_heatmap_*.png"))
    if not heatmaps:
        return []
    lines = ["### Expert Heatmaps", ""]
    for heatmap in heatmaps:
        lines.append(f"- `{heatmap.relative_to(exp_dir).as_posix()}`")
    lines.append("")
    return lines


def render_results(exp_dir: Path) -> str:
    pyversion = sys.version.replace("\n", " ")
    git_hash = safe_command_output(["git", "rev-parse", "HEAD"])
    git_date = safe_command_output(["git", "log", "-1", "--format=%cd"])

    lines = [
        "<!-- Generated by inference.show_asr_result -->",
        "# RESULTS",
        "## Environments",
        f"- date: `{datetime.now().astimezone().strftime('%a %b %d %H:%M:%S %Z %Y')}`",
        f"- python version: `{pyversion}`",
        f"- pytorch version: `pytorch {torch.__version__}`",
        f"- Git hash: `{git_hash}`",
        f"  - Commit date: `{git_date}`",
        "",
        f"## {exp_dir}",
        "",
    ]
    for metric in ("wer", "cer"):
        metric_lines = format_metric_table(exp_dir, metric)
        if metric_lines:
            lines.extend(metric_lines)
        else:
            lines.extend(format_result_table(exp_dir, metric))
    lines.extend(format_all_accuracy_tables(exp_dir))
    lines.extend(format_heatmap_paths(exp_dir))
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = build_argparser().parse_args()
    exp_dir = Path(args.exp_dir)
    print(render_results(exp_dir), end="")


if __name__ == "__main__":
    main()
