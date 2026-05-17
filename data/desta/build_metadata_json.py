#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import sys
import wave
from pathlib import Path
from typing import Iterable

try:
    import soundfile
except Exception:  # pragma: no cover - depends on runtime environment
    soundfile = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - depends on runtime environment
    tqdm = None


CHANNEL_RE = re.compile(r"\.CH[1-6]$", re.IGNORECASE)
KIND_RE = re.compile(r"_(REAL|SIMU)$", re.IGNORECASE)


def main() -> int:
    args = build_parser().parse_args()

    data_dir = Path(args.data_dir)
    wav_scp_path = data_dir / "wav.scp"
    text_path = data_dir / "text"
    output_path = Path(args.output) if args.output else data_dir / "metadata.json"

    wav_entries = read_wav_scp(wav_scp_path)
    text_by_id = read_text(text_path)
    gender_by_speaker = read_speaker_gender(Path(args.speaker_gender))
    annotations = read_annotations([Path(path) for path in args.annotations])

    records, errors = build_records(wav_entries, text_by_id, gender_by_speaker, annotations)

    duration_errors = fill_durations(records, args.nj, args.progress_every)
    errors["duration_errors"].extend(duration_errors)

    print_summary(len(wav_entries), len(records), errors, output_path)

    if any(errors.values()):
        print_error_examples(errors)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build CHiME4 metadata JSON from wav.scp, text, and CHiME3 annotation JSON files."
    )
    parser.add_argument("--data-dir", required=True, help="Dataset directory containing wav.scp and text.")
    parser.add_argument(
        "--annotations",
        required=True,
        nargs="+",
        help="Annotation JSON files. Filenames must contain _real or _simu.",
    )
    parser.add_argument(
        "--speaker-gender",
        default="/mnt/disk2/m11315045/MoE_Adapter/data/chime4/all_speaker/speaker",
        help="Speaker gender file with lines: <speaker> <0-or-1>, where 1=male and 0=female.",
    )
    parser.add_argument("--output", default="", help="Output JSON path. Defaults to <data-dir>/metadata.json.")
    parser.add_argument("--nj", type=int, default=1, help="Number of worker processes for reading wav durations.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Fallback progress print frequency when tqdm is unavailable.",
    )
    return parser


def read_wav_scp(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"wav.scp not found: {path}")

    entries: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Invalid wav.scp line {line_no} in {path}: {line}")
            entries.append((parts[0], parts[1]))
    if not entries:
        raise RuntimeError(f"wav.scp is empty: {path}")
    return entries


def read_text(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"text not found: {path}")

    text_by_id: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Invalid text line {line_no} in {path}: {line}")
            text_by_id[parts[0]] = parts[1]
    return text_by_id


def read_speaker_gender(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"speaker gender file not found: {path}")

    gender_by_speaker: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid speaker gender line {line_no} in {path}: {line}")

            speaker = parts[0].upper()
            if parts[1] == "1":
                gender = "male"
            elif parts[1] == "0":
                gender = "female"
            else:
                raise ValueError(f"Invalid gender id on line {line_no} in {path}: {parts[1]}")
            gender_by_speaker[speaker] = gender
    return gender_by_speaker


def read_annotations(paths: list[Path]) -> dict[tuple[str, str], dict]:
    annotations: dict[tuple[str, str], dict] = {}
    for path in paths:
        kind = annotation_kind(path)
        with path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"Annotation JSON must be a list: {path}")

        for index, row in enumerate(rows):
            try:
                speaker = str(row["speaker"]).upper()
                wsj_name = str(row["wsj_name"]).upper()
                environment = str(row["environment"]).upper()
            except KeyError as exc:
                raise ValueError(f"Missing annotation key {exc} in {path} row {index}") from exc

            base_id = f"{speaker}_{wsj_name}_{environment}"
            annotations[(base_id, kind)] = row
    return annotations


def annotation_kind(path: Path) -> str:
    name = path.name.lower()
    if "_real" in name:
        return "REAL"
    if "_simu" in name:
        return "SIMU"
    raise ValueError(f"Cannot infer annotation type from filename, expected _real or _simu: {path}")


def build_records(
    wav_entries: list[tuple[str, str]],
    text_by_id: dict[str, str],
    gender_by_speaker: dict[str, str],
    annotations: dict[tuple[str, str], dict],
) -> tuple[list[dict], dict[str, list[str]]]:
    records: list[dict] = []
    errors: dict[str, list[str]] = {
        "missing_annotation": [],
        "missing_text": [],
        "missing_gender": [],
        "duration_errors": [],
    }

    for utt_id, wav_path in wav_entries:
        annotation_key = annotation_lookup_key(utt_id)
        annotation = annotations.get(annotation_key) if annotation_key is not None else None
        if annotation is None:
            errors["missing_annotation"].append(utt_id)
            continue

        text = text_by_id.get(utt_id)
        if text is None:
            errors["missing_text"].append(utt_id)
            continue

        speaker = str(annotation["speaker"]).upper()
        gender = gender_by_speaker.get(speaker)
        if gender is None:
            errors["missing_gender"].append(speaker)
            continue

        records.append(
            {
                "id": utt_id,
                "text": text,
                "environment": str(annotation["environment"]).upper(),
                "speaker": speaker,
                "Gender": gender,
                "Duration": None,
                "_wav_path": wav_path,
            }
        )

    return records, errors


def annotation_lookup_key(utt_id: str) -> tuple[str, str] | None:
    match = KIND_RE.search(utt_id)
    if match is None:
        return None

    kind = match.group(1).upper()
    base_id = utt_id[: match.start()]
    base_id = CHANNEL_RE.sub("", base_id)
    return base_id.upper(), kind


def fill_durations(records: list[dict], nj: int, progress_every: int) -> list[str]:
    if not records:
        return []

    actual_nj = max(1, min(nj, len(records)))
    duration_errors: list[str] = []

    if actual_nj == 1:
        iterator = progress_iter(records, total=len(records), desc="Reading durations", progress_every=progress_every)
        for record in iterator:
            utt_id, duration, error = read_duration_item((record["id"], record["_wav_path"]))
            if error:
                duration_errors.append(error)
            else:
                record["Duration"] = duration
        cleanup_internal_fields(records)
        return duration_errors

    chunk_size = math.ceil(len(records) / actual_nj)
    chunks = [records[start : start + chunk_size] for start in range(0, len(records), chunk_size)]
    future_to_chunk: dict[concurrent.futures.Future, list[dict]] = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=actual_nj) as executor:
        for chunk in chunks:
            items = [(record["id"], record["_wav_path"]) for record in chunk]
            future = executor.submit(read_duration_chunk, items)
            future_to_chunk[future] = chunk

        completed = 0
        progress = progress_bar(total=len(records), desc="Reading durations")
        try:
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                results = future.result()
                for record, (utt_id, duration, error) in zip(chunk, results):
                    if record["id"] != utt_id:
                        duration_errors.append(f"{record['id']}: worker returned mismatched utt_id {utt_id}")
                    elif error:
                        duration_errors.append(error)
                    else:
                        record["Duration"] = duration

                completed += len(chunk)
                if progress is not None:
                    progress.update(len(chunk))
                elif progress_every > 0 and (completed == len(records) or completed % progress_every == 0):
                    print(f"Reading durations: {completed}/{len(records)}", file=sys.stderr)
        finally:
            if progress is not None:
                progress.close()

    cleanup_internal_fields(records)
    return duration_errors


def read_duration_chunk(items: list[tuple[str, str]]) -> list[tuple[str, float | None, str | None]]:
    return [read_duration_item(item) for item in items]


def read_duration_item(item: tuple[str, str]) -> tuple[str, float | None, str | None]:
    utt_id, wav_path = item
    try:
        return utt_id, read_wav_duration(wav_path), None
    except Exception as exc:  # pragma: no cover - data dependent
        return utt_id, None, f"{utt_id}: {wav_path}: {exc}"


def read_wav_duration(wav_path: str) -> float:
    if soundfile is not None:
        with soundfile.SoundFile(wav_path) as audio_file:
            return round(len(audio_file) / float(audio_file.samplerate), 6)

    with wave.open(wav_path, "rb") as audio_file:
        return round(audio_file.getnframes() / float(audio_file.getframerate()), 6)


def progress_bar(total: int, desc: str):
    if tqdm is None:
        return None
    return tqdm(total=total, desc=desc, unit="wav")


def progress_iter(items: list[dict], total: int, desc: str, progress_every: int) -> Iterable[dict]:
    if tqdm is not None:
        yield from tqdm(items, total=total, desc=desc, unit="wav")
        return

    for index, item in enumerate(items, start=1):
        yield item
        if progress_every > 0 and (index == total or index % progress_every == 0):
            print(f"{desc}: {index}/{total}", file=sys.stderr)


def cleanup_internal_fields(records: list[dict]) -> None:
    for record in records:
        record.pop("_wav_path", None)


def print_summary(total_wavs: int, written_records: int, errors: dict[str, list[str]], output_path: Path) -> None:
    print(f"Loaded wavs: {total_wavs}")
    print(f"Written records: {written_records}")
    print(f"Missing annotation: {len(errors['missing_annotation'])}")
    print(f"Missing text: {len(errors['missing_text'])}")
    print(f"Missing gender: {len(set(errors['missing_gender']))}")
    print(f"Duration read errors: {len(errors['duration_errors'])}")
    print(f"Output: {output_path}")


def print_error_examples(errors: dict[str, list[str]], max_examples: int = 10) -> None:
    print("Errors found; metadata JSON was not written.", file=sys.stderr)
    for name, values in errors.items():
        if not values:
            continue
        if name == "missing_gender":
            values = sorted(set(values))
        print(f"{name} examples:", file=sys.stderr)
        for value in values[:max_examples]:
            print(f"  {value}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
