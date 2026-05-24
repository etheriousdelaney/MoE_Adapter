#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - depends on runtime environment
    tqdm = None


def main() -> int:
    args = build_parser().parse_args()

    message_path = Path(args.messages)
    output_path = Path(args.output)
    full_output_path = Path(args.full_output) if args.full_output else None
    decoder = load_decoder(Path(args.decoder))

    messages = read_jsonl(message_path)
    if args.limit > 0:
        messages = messages[: args.limit]
    metadata_by_id = read_metadata_map(Path(args.metadata)) if args.metadata else {}

    done_keys = read_done_keys(output_path) if args.resume else set()
    pending = [item for item in messages if response_key(item) not in done_keys]

    print(f"Loaded messages: {len(messages)}")
    print(f"Pending messages: {len(pending)}")
    print(f"Output: {output_path}")
    if full_output_path is not None:
        print(f"Full output: {full_output_path}")

    if not pending:
        return 0

    model_bundle = decoder.load_model(
        args.model,
        device=args.device,
        torch_dtype=args.torch_dtype,
        trust_remote_code=not args.no_trust_remote_code,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if full_output_path is not None:
        full_output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with output_path.open(mode, encoding="utf-8") as f:
        full_f = full_output_path.open(mode, encoding="utf-8") if full_output_path is not None else None
        try:
            for item in progress_iter(pending, desc="Generating responses"):
                item_max_new_tokens = args.max_new_tokens
                if not args.ignore_message_max_tokens:
                    item_max_new_tokens = int(item.get("max_new_tokens", args.max_new_tokens))
                response = decoder.generate_response(
                    model_bundle,
                    item["messages"],
                    max_new_tokens=item_max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=args.do_sample,
                    enable_thinking=args.enable_thinking,
                )
                prompt = item.get("prompt", "")
                out = {
                    "id": item["id"],
                    "prompt": prompt,
                    "response": response,
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()

                if full_f is not None:
                    user_message = next(
                        (message["content"] for message in item["messages"] if message.get("role") == "user"),
                        "",
                    )
                    full_out = {
                        "id": item["id"],
                        "prompt": prompt,
                        "message": user_message,
                        "max_new_tokens": item_max_new_tokens,
                        "response": response,
                        "metadata": item.get("metadata") or metadata_by_id.get(str(item["id"]), {}),
                    }
                    full_f.write(json.dumps(full_out, ensure_ascii=False) + "\n")
                    full_f.flush()
        finally:
            if full_f is not None:
                full_f.close()

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate response JSONL from DeSTA/Qwen message JSONL.")
    parser.add_argument("--messages", required=True, help="Input message JSONL path.")
    parser.add_argument("--decoder", required=True, help="Python decoder file with load_model/generate_response.")
    parser.add_argument("--model", required=True, help="Model repo or local model path passed to decoder.load_model.")
    parser.add_argument("--output", required=True, help="Output response JSONL path.")
    parser.add_argument(
        "--full-output",
        default="",
        help="Optional full JSONL output containing id, prompt, message, response, and metadata.",
    )
    parser.add_argument(
        "--metadata",
        default="",
        help="Optional metadata.json used to fill metadata in --full-output when messages do not contain metadata.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved for compatibility; generation is sequential.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--ignore-message-max-tokens",
        action="store_true",
        default=False,
        help="Ignore per-message max_new_tokens and always use --max-new-tokens.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Enable Qwen thinking mode. Default is disabled for direct answers.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--torch-dtype", default="auto", help="auto, float32, float16, bfloat16.")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N messages; 0 means all.")
    parser.add_argument("--resume", action="store_true", help="Skip responses already present in output.")
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=False to the decoder.",
    )
    return parser


def load_decoder(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"decoder not found: {path}")
    spec = importlib.util.spec_from_file_location("desta_decoder_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load decoder module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ("load_model", "generate_response"):
        if not hasattr(module, name):
            raise AttributeError(f"decoder {path} must define {name}()")
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"message JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_no}: expected JSON object")
            if "id" not in row or "messages" not in row:
                raise ValueError(f"line {line_no}: missing id or messages")
            rows.append(row)
    return rows


def read_done_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add((str(row.get("id", "")), str(row.get("prompt", ""))))
    return done


def read_metadata_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"metadata JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"metadata JSON must contain a list: {path}")

    metadata_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or "id" not in row:
            raise ValueError(f"metadata row {index} must be an object with id")
        metadata_by_id[str(row["id"])] = row
    return metadata_by_id


def response_key(item: dict[str, Any]) -> tuple[str, str]:
    return str(item.get("id", "")), str(item.get("prompt", ""))


def progress_iter(items: list[dict[str, Any]], desc: str) -> Iterable[dict[str, Any]]:
    if tqdm is not None:
        yield from tqdm(items, desc=desc, unit="msg")
        return
    for index, item in enumerate(items, start=1):
        yield item
        if index == len(items) or index % 100 == 0:
            print(f"{desc}: {index}/{len(items)}")


if __name__ == "__main__":
    raise SystemExit(main())
