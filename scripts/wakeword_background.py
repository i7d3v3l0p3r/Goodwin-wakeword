"""Add MUSAN noise and music and FLEURS Russian speech to a local corpus."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from urllib.parse import quote

import httpx
import numpy as np
import soundfile as sf

from wakeword_dataset import (
    DEFAULT_PROFILE, RATE, PhraseProfile, convert, digest, normalize, read_json, save_wave, split_for, write_json,
)


REPOSITORY = "FluidInference/musan"
HUB = "https://huggingface.co"
SOURCES = {
    "environment": ("noise/free-sound",),
}
PARQUET_SOURCES = {
    "music": "https://huggingface.co/datasets/corypaik/musan/resolve/refs%2Fconvert%2Fparquet/music/train/0008.parquet",
    "speech": "https://huggingface.co/datasets/google/fleurs/resolve/refs%2Fconvert%2Fparquet/ru_ru/validation/0000.parquet",
}


def get(client, url):
    for attempt in range(3):
        try:
            response = client.get(url)
            response.raise_for_status()
            return response
        except httpx.HTTPError:
            if attempt == 2:
                raise


def files(client, directories):
    for directory in directories:
        url = f"{HUB}/api/datasets/{REPOSITORY}/tree/main/{directory}?recursive=true&limit=100"
        while url:
            response = get(client, url)
            for item in response.json():
                if item["path"].endswith(".wav") and 200_000 <= item["size"] <= 24_000_000:
                    yield item["path"]
            url = response.links.get("next", {}).get("url")


def source_url(path):
    return f"{HUB}/datasets/{REPOSITORY}/resolve/main/{quote(path)}"


def windows(root, category, source, group, audio, remaining, rows, text="", metadata=None):
    folder = root / "background" / category
    folder.mkdir(parents=True, exist_ok=True)
    total = 0
    for start in range(0, len(audio) - RATE * 6 + 1, RATE * 10):
        payload = audio[start:start + RATE * 10]
        if np.std(payload) < 0.0002:
            continue
        recording = metadata.get("source_row") if metadata else source
        identifier = digest(group, recording, start)[:24]
        relative = Path("background") / category / (identifier + ".wav")
        if not (root / relative).exists():
            save_wave(root / relative, payload)
        duration = len(payload) / RATE
        rows.append({
            "id": identifier, "category": "noise" if category in {"environment", "music"} else "real_speech",
            "background_type": category, "status": "accepted",
            "label": 0, "path": relative.as_posix(), "duration_s": round(duration, 3),
            "group": group, "split": split_for(group), "source": source,
            "source_offset_s": start / RATE, "text": text, **(metadata or {}),
        })
        total += duration
        if total >= remaining:
            break
    return total


def write_background(client, root, category, paths, seconds, rows):
    total = 0
    for source in sorted(paths, key=lambda item: digest("musan-goodwin-1", item)):
        try:
            response = get(client, source_url(source))
            audio, source_rate = sf.read(io.BytesIO(response.content), dtype="float32", always_2d=True)
            audio = convert(audio.mean(axis=1), source_rate)
        except (httpx.HTTPError, ValueError, RuntimeError) as error:
            print(f"Skipping {source}: {type(error).__name__}", flush=True)
            continue
        group = "musan/" + source
        total += windows(root, category, source_url(source), group, audio, seconds - total, rows)
        if total >= seconds:
            return total
    return total


def parquet_background(root, category, path, seconds, rows, profile, source):
    import pyarrow.parquet as pq

    table = pq.ParquetFile(path)
    columns = [name for name in ("audio", "id", "transcription", "path", "title", "license", "source", "gender")
               if name in table.schema_arrow.names]
    aliases = set(profile.aliases)
    total = 0
    index = 0
    for batch in table.iter_batches(batch_size=4, columns=columns):
        for row in batch.to_pylist():
            index += 1
            text = row.get("transcription") or ""
            if aliases.intersection(normalize(text).split()):
                continue
            blob = row["audio"]
            if not blob or not blob.get("bytes"):
                continue
            audio, rate = sf.read(io.BytesIO(blob["bytes"]), dtype="float32", always_2d=True)
            audio = convert(audio.mean(axis=1), rate)
            original = row.get("path") or blob.get("path") or str(index)
            group = f"{category}/{row.get('id', original)}"
            metadata = {"source_recording": original, "source_row": index - 1}
            for name in ("title", "license"):
                if row.get(name):
                    metadata[name] = row[name]
            if row.get("source"):
                metadata["source_collection"] = row["source"]
            if row.get("gender") is not None:
                metadata["source_gender"] = row["gender"]
            if category == "speech" and len(audio) < RATE * 6:
                identifier = digest(group, index - 1, 0)[:24]
                relative = Path("background/speech") / (identifier + ".wav")
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                save_wave(root / relative, audio)
                duration = len(audio) / RATE
                rows.append({
                    "id": identifier, "category": "real_speech", "background_type": category,
                    "status": "accepted", "label": 0, "path": relative.as_posix(),
                    "duration_s": round(duration, 3), "group": group, "split": split_for(group),
                    "source": source, "source_offset_s": 0, "text": text, **metadata,
                })
                total += duration
            else:
                total += windows(root, category, source, group, audio, seconds - total, rows, text, metadata)
            if total >= seconds:
                return total
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/goodwin"))
    parser.add_argument("--profile", type=Path, help="Exclude speech containing this profile's wake phrase")
    parser.add_argument("--minutes-per-type", type=int, default=20)
    parser.add_argument("--music-parquet", type=Path)
    parser.add_argument("--music-source", default=PARQUET_SOURCES["music"])
    parser.add_argument("--speech-parquet", type=Path)
    parser.add_argument("--speech-source", default=PARQUET_SOURCES["speech"])
    args = parser.parse_args()
    profile = PhraseProfile(**read_json(args.profile)) if args.profile else DEFAULT_PROFILE
    index = args.root / "background.json"
    rows = read_json(index) if index.exists() else []
    with httpx.Client(timeout=httpx.Timeout(60, connect=12), follow_redirects=True) as client:
        for category, directories in SOURCES.items():
            if sum(row["duration_s"] for row in rows if row["background_type"] == category) >= args.minutes_per_type * 60:
                continue
            recording_paths = list(files(client, directories))
            print(f"{category}: {len(recording_paths)} source recordings", flush=True)
            duration = write_background(
                client, args.root, category, recording_paths, args.minutes_per_type * 60, rows,
            )
            print(f"{category}: {duration / 60:.1f} minutes", flush=True)
            write_json(index, rows)
        notices = args.root / "background" / "licenses"
        notices.mkdir(parents=True, exist_ok=True)
        for category, directories in SOURCES.items():
            for directory in directories:
                try:
                    license_data = get(client, source_url(directory + "/LICENSE")).content
                except httpx.HTTPError:
                    continue
                (notices / (directory.replace("/", "-") + ".txt")).write_bytes(license_data)
        (notices / "SOURCE.txt").write_text(
            "MUSAN: A Music, Speech, and Noise Corpus (Snyder, Chen, and Povey, 2015)\n"
            "Upstream: https://www.openslr.org/17/\n"
            f"Noise mirror: https://huggingface.co/datasets/{REPOSITORY}\n"
            "Music mirror: https://huggingface.co/datasets/corypaik/musan\n"
            "FLEURS Russian validation speech: https://huggingface.co/datasets/google/fleurs\n"
            "Files are 16 kHz mono excerpts. See background.json for exact source and offset.\n"
            "MUSAN lists CC BY 4.0 for the corpus; the source folders give per-recording terms.\n",
            encoding="utf-8",
        )
    for category, path, source in (
        ("music", args.music_parquet, args.music_source),
        ("speech", args.speech_parquet, args.speech_source),
    ):
        if path:
            rows = [row for row in rows if row["background_type"] != category]
            duration = parquet_background(args.root, category, path, args.minutes_per_type * 60, rows, profile, source)
            print(f"{category}: {duration / 60:.1f} minutes", flush=True)
            write_json(index, rows)


if __name__ == "__main__":
    main()
