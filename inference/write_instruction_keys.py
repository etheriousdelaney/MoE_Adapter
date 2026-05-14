from __future__ import annotations

import argparse
from pathlib import Path

from dataset.instruction_utils import iter_instruction_records


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write instruction sample ids from response.jsonl",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--response_jsonl", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in iter_instruction_records(args.response_jsonl):
            f.write(record["sample_id"] + "\n")


if __name__ == "__main__":
    main()
