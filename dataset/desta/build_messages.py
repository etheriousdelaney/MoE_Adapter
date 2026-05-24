#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "You are an audio language model.\n\nThe audio clip is provided between <start_audio> and <end_audio>.\n\nFormat rules:\n- Square brackets contain timestamps only, such as [00:00.00-00:07.70].\n- Parentheses contain metadata such as environment, gender, and duration.\n- Text outside square brackets and parentheses is the transcribed speech content from the audio.\n- The transcribed speech content is not the target answer unless the user explicitly asks for transcription.\n- Use the metadata only to identify the target label when relevant.\n- Do not output timestamps, gender, duration, transcript, explanations, or any extra text."
    )


REQUIRED_METADATA_KEYS = ("id", "text", "environment", "Gender", "Duration")


def main() -> int:
    args = build_parser().parse_args()

    data_dir = Path(args.data_dir)
    metadata_path = Path(args.metadata) if args.metadata else data_dir / "metadata.json"
    prompt_path = Path(args.prompt)
    output_path = Path(args.output) if args.output else data_dir / "message.jsonl"

    metadata = read_json_list(metadata_path, "metadata")
    prompts = read_json_list(prompt_path, "prompt")
    prompts = [str(prompt) for prompt in prompts]

    errors = validate_metadata(metadata)
    if errors:
        print("Invalid metadata; message JSONL was not written.")
        for error in errors[:20]:
            print(f"  {error}")
        if len(errors) > 20:
            print(f"  ... {len(errors) - 20} more")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in metadata:
            for prompt in prompts:
                item = build_message_item(
                    row,
                    prompt,
                    args.system_prompt,
                    default_max_new_tokens=args.default_max_new_tokens,
                    auto_max_new_tokens=not args.disable_auto_max_new_tokens,
                )
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1

    print(f"Loaded metadata: {len(metadata)}")
    print(f"Loaded prompts: {len(prompts)}")
    print(f"Written messages: {count}")
    print(f"Output: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build DeSTA/Qwen text-only message JSONL from CHiME4 metadata.json."
    )
    parser.add_argument("--data-dir", required=True, help="Dataset directory containing metadata.json.")
    parser.add_argument(
        "--metadata",
        default="",
        help="Metadata JSON path. Defaults to <data-dir>/metadata.json.",
    )
    parser.add_argument(
        "--prompt",
        default="/mnt/disk2/m11315045/MoE_Adapter/data/desta/prompt",
        help="Prompt JSON list path.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt inserted as the first chat message.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSONL path. Defaults to <data-dir>/message.jsonl.",
    )
    parser.add_argument(
        "--default-max-new-tokens",
        type=int,
        default=256,
        help="Fallback max_new_tokens stored per message when no task-specific rule matches.",
    )
    parser.add_argument(
        "--disable-auto-max-new-tokens",
        action="store_true",
        help="Do not derive per-message max_new_tokens from prompt/metadata.",
    )
    return parser


def read_json_list(path: Path, label: str) -> list[Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{label} file must contain a JSON list: {path}")
    return data


def validate_metadata(metadata: list[Any]) -> list[str]:
    errors: list[str] = []
    for index, row in enumerate(metadata):
        if not isinstance(row, dict):
            errors.append(f"row {index}: expected object, got {type(row).__name__}")
            continue
        for key in REQUIRED_METADATA_KEYS:
            if key not in row:
                errors.append(f"row {index} ({row.get('id', '<no id>')}): missing {key}")
        try:
            float(row.get("Duration", 0.0))
        except (TypeError, ValueError):
            errors.append(f"row {index} ({row.get('id', '<no id>')}): invalid Duration {row.get('Duration')}")
    return errors


def build_message_item(
    row: dict[str, Any],
    prompt: str,
    system_prompt: str,
    default_max_new_tokens: int,
    auto_max_new_tokens: bool,
) -> dict[str, Any]:
    duration = float(row["Duration"])
    end_time = format_timestamp(duration)
    max_new_tokens = estimate_max_new_tokens(row, prompt, default_max_new_tokens) if auto_max_new_tokens else default_max_new_tokens
    content = (
        f"{prompt}<start_audio>[00:00.00-{end_time}] {row['text']} "
        f"(environment: {row['environment']}, gender:{row['Gender']}, Duration: {row['Duration']})"
        "<end_audio>"
    )
    return {
        "id": row["id"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "prompt": prompt,
        # "max_new_tokens": max_new_tokens,
    }


def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        raise ValueError(f"Duration must be non-negative: {seconds}")
    total_centiseconds = int(round(seconds * 100))
    minutes, centiseconds = divmod(total_centiseconds, 60 * 100)
    secs, centis = divmod(centiseconds, 100)
    return f"{minutes:02d}:{secs:02d}.{centis:02d}"


def estimate_max_new_tokens(row: dict[str, Any], prompt: str, default_max_new_tokens: int) -> int:
    prompt_lower = prompt.lower()
    if "environment" in prompt_lower or "environmental" in prompt_lower:
        return 4
    if "gender" in prompt_lower or "male or female" in prompt_lower:
        return 4
    if "transcribe" in prompt_lower or "transcription" in prompt_lower:
        word_count = len(str(row["text"]).split())
        return max(8, min(default_max_new_tokens, int(math.ceil(word_count * 1.35)) + 8))
    if "describe" in prompt_lower:
        return min(default_max_new_tokens, 64)
    return default_max_new_tokens


if __name__ == "__main__":
    raise SystemExit(main())
