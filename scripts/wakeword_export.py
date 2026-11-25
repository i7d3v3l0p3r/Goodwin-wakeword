"""Validate WAVs and export the corpus as LiveKit WakeWord Bench Parquet shards."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import re
import wave
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from wakeword_dataset import DEFAULT_PROFILE, RATE, PhraseProfile, read_json, word_match, write_json


def parquet_schema():
    fields = {
        **dict.fromkeys(("id", "split", "category", "wake_word", "text", "text_form", "phrase",
                         "provider", "model", "voice_id", "voice_name", "accent", "gender",
                         "noise_bucket", "noise_type", "noise_source", "noise_category",
                         "noise_file", "rir_source", "rir_file", "rir_category", "interferer",
                         "noise2_category", "eq", "bandlimit", "codec", "augmentations",
                         "tts_key", "phonemes", "tts_input"), pa.string()),
        **dict.fromkeys(("label", "adversarial_distance", "n_talkers"), pa.int32()),
        **dict.fromkeys(("held_out_voice", "reverb", "party", "clipping"), pa.bool_()),
        **dict.fromkeys(("speed", "snr_db", "sir_db", "level_dbfs", "onset_s", "offset_s",
                         "phrase_onset_s", "phrase_offset_s", "duration_s"), pa.float32()),
    }
    features = {
        name: {"dtype": "float32" if pa.types.is_float32(kind) else str(kind), "_type": "Value"}
        for name, kind in fields.items()
    }
    fields["audio"] = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    features["audio"] = {"_type": "Audio", "sampling_rate": None}
    return pa.schema(list(fields.items()), metadata={
        b"huggingface": json.dumps({"info": {"features": features}}).encode(),
    })


def inspect_file(root, row):
    path = root / row["path"]
    audio = path.read_bytes()
    with wave.open(io.BytesIO(audio), "rb") as recording:
        actual = recording.getframerate(), recording.getnchannels(), recording.getsampwidth()
        if actual != (RATE, 1, 2) or recording.getnframes() == 0:
            raise ValueError(f"Invalid WAV: {path}")
        duration = recording.getnframes() / RATE
        pcm = recording.readframes(recording.getnframes())
    if abs(duration - row["duration_s"]) > 0.002 or len(pcm) != round(duration * RATE) * 2:
        raise ValueError(f"Invalid duration: {path}")
    if row["label"] == 1 and not 0 <= row["phrase_onset_s"] < row["phrase_offset_s"] <= duration + 0.05:
        raise ValueError(f"Invalid wake word boundaries: {path}")
    return hashlib.sha256(pcm).hexdigest()


def parquet_row(root, row, schema, profile=DEFAULT_PROFILE):
    path = root / row["path"]
    audio, _ = sf.read(path, dtype="float32")
    flac = io.BytesIO()
    sf.write(flac, audio, RATE, format="FLAC", subtype="PCM_16")
    active = np.flatnonzero(np.abs(audio) > max(0.003, np.max(np.abs(audio)) * 0.02))
    onset = float(active[0] / RATE) if len(active) else 0.0
    offset = float((active[-1] + 1) / RATE) if len(active) else row["duration_s"]
    background = row.get("background_type") or ""
    voice = row.get("voice") or ""
    model = row.get("model") or ""
    category = row["category"]
    text = row.get("text") or ""
    meta = {
        **row, "wake_word": profile.phrase, "text": text,
        "phrase": profile.phrase if row["label"] else row["text"].split()[0] if category == "adversarial" else "",
        "provider": row.get("provider") or (model.split("/")[0] if model else "fleurs" if background == "speech" else "musan"),
        "model": model, "voice_id": row.get("group", "") if model else "", "voice_name": voice,
        "held_out_voice": bool(model and row["split"] == "test"),
        "noise_bucket": "clean", "noise_type": background,
        "noise_category": background, "noise_source": row.get("source", ""),
        "noise_file": row.get("source_recording", ""),
        "speed": 1.0, "adversarial_distance": -1, "codec": "none", "bandlimit": "none",
        "text_form": "isolated" if row["label"] and word_match(text, profile) else
        "carrier_prefix" if row["label"] else "transcript" if category == "real_speech" else "",
        "level_dbfs": 20 * math.log10(max(1e-8, float(np.sqrt(np.mean(audio**2))))),
        "onset_s": onset, "offset_s": offset,
        "tts_key": row["id"] if model else "", "tts_input": row.get("text", "") if model else "",
        "audio": {"bytes": flac.getvalue(), "path": f"{row['id']}.flac"},
    }
    values = {}
    for field in schema:
        if field.name in meta:
            values[field.name] = meta[field.name]
        elif pa.types.is_string(field.type):
            values[field.name] = ""
        elif pa.types.is_boolean(field.type):
            values[field.name] = False
        elif pa.types.is_floating(field.type):
            values[field.name] = math.nan
        elif pa.types.is_integer(field.type):
            values[field.name] = 0
    return values


def export(root, profile=DEFAULT_PROFILE):
    manifest = root / "manifest.jsonl"
    with manifest.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    expected = {str(row["path"]) for row in rows}
    actual = {path.relative_to(root).as_posix()
              for category in ("positive", "negative", "background")
              for path in (root / category).rglob("*.wav")}
    if actual != expected or len(expected) != len(rows):
        raise ValueError(f"Manifest mismatch: missing={len(expected - actual)}, extra={len(actual - expected)}")
    by_audio = collections.defaultdict(list)
    by_group = collections.defaultdict(set)
    for row in rows:
        by_audio[inspect_file(root, row)].append(row)
        by_group[row["group"]].add(row["split"])
    cross_split_groups = [group for group, splits in by_group.items() if len(splits) != 1]
    if cross_split_groups:
        raise ValueError(f"{len(cross_split_groups)} voice/source groups span multiple splits")
    clean = collections.defaultdict(list)
    duplicate_rows = []
    for recordings in by_audio.values():
        if len({row["label"] for row in recordings}) > 1:
            duplicate_rows.extend({"path": row["path"], "reason": "conflicting_label"} for row in recordings)
            continue
        # A WAV can be kept in the delivery to show provider coverage, but only
        # the canonical copy is exported so exact copies cannot cross splits.
        first, *duplicates = recordings
        clean[first["split"]].append(first)
        duplicate_rows.extend({"path": row["path"], "duplicate_of": first["path"]} for row in duplicates)
    schema = parquet_schema()
    target = root / "data"
    target.mkdir(exist_ok=True)
    shards = {}
    for split, items in sorted(clean.items()):
        for index in range(0, len(items), 250):
            batch = items[index:index + 250]
            destination = target / f"{split}-{index // 250:05d}.parquet"
            temporary = destination.with_suffix(".parquet.part")
            pq.write_table(pa.Table.from_pylist([parquet_row(root, row, schema, profile) for row in batch], schema=schema),
                           temporary, row_group_size=100)
            temporary.replace(destination)
            count = pq.ParquetFile(destination).metadata.num_rows
            if count != len(batch):
                raise ValueError(f"Parquet row count differs: {destination}")
            shards[destination.name] = count
    verification = root / "verification.json"
    if verification.exists():
        stale = set(read_json(verification).get("parquet_shards", {})) - shards.keys()
        for name in stale:
            if re.fullmatch(r"(train|validation|test)-[0-9]{5}\.parquet", name):
                (target / name).unlink(missing_ok=True)
    counts = collections.Counter(row["category"] for row in rows)
    pairs = collections.Counter((row.get("voice_id"), row["label"]) for row in rows if row.get("voice_id"))
    coverage_path = root / "coverage.json"
    coverage = read_json(coverage_path) if coverage_path.exists() else []
    with (root / "coverage.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        columns = ["model", "voice", "status", "reason", "positive", "negative"]
        report_csv = csv.DictWriter(stream, fieldnames=columns)
        report_csv.writeheader()
        for voice in coverage:
            report_csv.writerow({
                **{key: voice.get(key) for key in columns[:4]},
                "positive": pairs[voice["id"], 1], "negative": pairs[voice["id"], 0],
            })
    missing_classes = [
        {"model": voice["model"], "voice": voice["voice"], "voice_id": voice["id"],
         "positive": pairs[voice["id"], 1], "negative": pairs[voice["id"], 0]}
        for voice in coverage if voice["status"] == "accepted"
        and not (pairs[voice["id"], 1] and pairs[voice["id"], 0])
    ]
    report = {
        "wav_count": len(rows), "categories": dict(counts),
        "split_sizes": {split: len(items) for split, items in sorted(clean.items())},
        "passing_voices_missing_a_class": missing_classes,
        "duplicate_or_conflicting_rows": duplicate_rows, "parquet_shards": shards,
    }
    write_json(verification, report)
    print(f"Checked {len(rows)} WAV files. Exported {sum(shards.values())} distinct clips to {len(shards)} shards.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/goodwin"))
    parser.add_argument("--profile", type=Path, help="JSON file with the exported wake phrase")
    args = parser.parse_args()
    profile = PhraseProfile(**read_json(args.profile)) if args.profile else DEFAULT_PROFILE
    export(args.root, profile)
