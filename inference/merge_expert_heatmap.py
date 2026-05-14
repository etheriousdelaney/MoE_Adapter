from __future__ import annotations

import argparse
from pathlib import Path

from inference.expert_heatmap_utils import finalize_heatmap_matrix, merge_accumulators, render_heatmap


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge expert heatmap stats from multiple inference jobs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stats", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="ASR Inference Expert Usage")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    sums, counts, row_order, highlight_top_k = merge_accumulators(args.stats)
    render_heatmap(
        matrix=finalize_heatmap_matrix(sums, counts),
        output_path=Path(args.output),
        title=args.title,
        row_order=row_order,
        highlight_top_k=highlight_top_k,
    )


if __name__ == "__main__":
    main()
