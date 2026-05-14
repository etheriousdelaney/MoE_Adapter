from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator


INSTRUCTION_SAMPLE_SEPARATOR = "__sample__"


def compose_instruction_sample_id(base_id: str, sample_index: int) -> str:
    return f"{base_id}{INSTRUCTION_SAMPLE_SEPARATOR}{sample_index}"


def strip_instruction_sample_id(sample_id: str) -> str:
    if INSTRUCTION_SAMPLE_SEPARATOR not in sample_id:
        return sample_id
    return sample_id.split(INSTRUCTION_SAMPLE_SEPARATOR, 1)[0]


def is_instruction_sample_id(sample_id: str) -> bool:
    return INSTRUCTION_SAMPLE_SEPARATOR in sample_id


def iter_instruction_records(response_jsonl_path: str | Path) -> Iterator[dict[str, str]]:
    response_path = Path(response_jsonl_path)
    counters: dict[str, int] = defaultdict(int)
    with response_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            payload = json.loads(line)
            base_id = str(payload["id"])
            sample_index = counters[base_id]
            counters[base_id] += 1
            yield {
                "sample_id": compose_instruction_sample_id(base_id, sample_index),
                "base_id": base_id,
                "prompt": str(payload["prompt"]),
                "response": str(payload["response"]),
                "line_num": str(line_num),
            }
