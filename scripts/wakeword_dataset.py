"""Generate and check a Russian wake word corpus using OpenRouter TTS."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import io
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


RATE = 16_000
API = "https://openrouter.ai/api/v1"
PROBE = "Гудвин сколько стоит билет на поезд"
POSITIVE = (
    "Гудвин",
    "Гудвин включи музыку",
    "Гудвин поставь таймер",
    "Гудвин какая завтра погода",
    "Гудвин!",
)
NEGATIVE = (
    ("adversarial", "Гудман включи музыку"),
    ("speech", "Сколько стоит билет на поезд?"),
    ("adversarial", "Гудвилл этой компании растёт"),
    ("speech", "Включи музыку"),
    ("adversarial", "Будни проходят быстро"),
    ("speech", "Какая завтра погода?"),
)
EXTRA_POSITIVE = ("Гудвин который час", "Гудвин сделай потише", "Гудвин расскажи сказку",
                  "Гудвин какая температура", "Гудвин добрый вечер", "Гудвин выключи свет")
EXTRA_NEGATIVE = (
    ("adversarial", "Гудман поставь таймер"), ("speech", "Который сейчас час?"),
    ("adversarial", "Гудвилл вырос"), ("speech", "Сделай музыку потише"),
    ("adversarial", "Будни начнутся завтра"), ("speech", "На улице холодно"),
)


@dataclass(frozen=True)
class PhraseProfile:
    phrase: str
    probe: str
    probe_words: list[str]
    aliases: list[str]
    positive: list[str]
    negative: list[tuple[str, str]]
    extra_positive: list[str]
    extra_negative: list[tuple[str, str]]
    recovery_spellings: list[str]


DEFAULT_PROFILE = PhraseProfile(
    "Гудвин", PROBE, ["сколько", "стоит", "билет", "поезд"], ["гудвин", "goodwin", "гудуин"],
    list(POSITIVE), list(NEGATIVE), list(EXTRA_POSITIVE), list(EXTRA_NEGATIVE),
    ["годвин", "гудвен", "годвен", "гудвейн", "гудвинов", "годвинов"],
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    replace(temporary, path)


def replace(temporary, path):
    for attempt in range(7):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(0.08 * (attempt + 1))


def digest(*parts):
    data = json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(data).hexdigest()


def api_key(env_file):
    if key := os.environ.get("OPENROUTER_API_KEY"):
        return key.strip()
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            match = re.match(r"^\s*(?:export\s+)?OPENROUTER_API_KEY\s*=\s*(.+?)\s*$", line)
            if match:
                key = match[1].split("#", 1)[0].strip().strip("\"'")
                if key:
                    return key
    raise RuntimeError(f"Set OPENROUTER_API_KEY in the environment or in {env_file}")


def refresh_catalog(root, env_file):
    endpoints = {
        "openrouter-models.json": "/models?output_modalities=speech",
        "openrouter-speech.json": "/audio/speech/models",
    }
    headers = {"Authorization": f"Bearer {api_key(env_file)}"}
    snapshots = {}
    with httpx.Client(headers=headers, timeout=25) as client:
        for filename, endpoint in endpoints.items():
            response = client.get(API + endpoint)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload.get("data"), list) or not payload["data"]:
                raise ValueError(f"OpenRouter returned no models at {endpoint}")
            snapshots[filename] = payload
    root.mkdir(parents=True, exist_ok=True)
    for filename, payload in snapshots.items():
        write_json(root / filename, payload)
        print(f"{filename}: {len(payload['data'])} models")


def parameters(model):
    return next(
        (entry.get("supported_parameters", {}) for entry in model.get("output_modalities", [])
         if entry.get("type") == "speech"), {}
    )


def inventory(root):
    if not all((root / name).is_file() for name in ("openrouter-models.json", "openrouter-speech.json")):
        raise FileNotFoundError(f"Run the catalog command for {root} to fetch OpenRouter model metadata")
    old = read_json(root / "openrouter-models.json")["data"]
    versions = {row["id"]: row for row in read_json(root / "openrouter-speech.json")["data"]}
    extras_path = root / "voice-overrides.json"
    extras = read_json(extras_path) if extras_path.exists() else {}
    result = []
    for model in old:
        name = model["id"]
        family = name.split("/")[0]
        info = versions.get(name, versions.get(name.removesuffix(":free"), {}))
        meta = parameters(info)
        published_voices = (meta.get("voice", {}).get("values") or []) + (model.get("supported_voices") or [])
        voices = sorted(set(published_voices + extras.get(name, [])))
        if not published_voices:
            voices.append(None)
        codecs = meta.get("response_format", {}).get("values") or ["pcm", "mp3"]
        codec = next((value for value in ("wav", "pcm", "mp3") if value in codecs), codecs[0])
        for voice in voices or [None]:
            skip = None
            if family == "microsoft" and voice and re.match(r"^[a-z]{2}-[A-Z]{2}-", voice) and not voice.startswith("ru-"):
                skip = "fixed_non_russian_locale"
            if family == "deepgram" and voice and not voice.endswith("-ru"):
                skip = "fixed_non_russian_locale"
            if family == "hexgrad" and voice and not voice.startswith("r"):
                skip = "fixed_non_russian_locale"
            routes = [endpoint["provider_slug"] for endpoint in info.get("endpoints", [])
                      if not voice or not parameters(endpoint).get("voice", {}).get("values")
                      or voice in parameters(endpoint)["voice"]["values"]]
            result.append({
                "id": digest(name, voice)[:20],
                "model": name, "family": family, "voice": voice,
                "format": codec, "routes": routes, "skip": skip,
                "pricing": model.get("pricing", {}),
                "instructions": "instructions" in meta,
            })
    write_json(root / "inventory.json", result)
    return result


class Journal:
    def __init__(self, root):
        self.lock = threading.Lock()
        self.database = sqlite3.connect(root / "jobs.sqlite3", check_same_thread=False)
        self.database.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, row TEXT NOT NULL)")
        self.database.execute("PRAGMA journal_mode=WAL")
        self.database.execute("PRAGMA synchronous=NORMAL")
        self.rows = {key: json.loads(value) for key, value in self.database.execute("SELECT id, row FROM jobs")}

    def put(self, row):
        with self.lock, self.database:
            self.database.execute("INSERT OR REPLACE INTO jobs VALUES (?,?)",
                                  (row["id"], json.dumps(row, ensure_ascii=False)))
            self.rows[row["id"]] = row

    def successful(self, root, key):
        row = self.rows.get(key)
        return row and row["status"] == "accepted" and (root / row["path"]).is_file()


def voice_group(voice):
    identity = (voice["voice"] or "default").casefold().split(":")[0]
    return voice["family"] + "/" + identity


def split_for(group):
    if group.startswith("x-ai/"):
        return "test"
    index = int(digest("goodwin-split-1", group)[:8], 16) % 100
    return "train" if index < 78 else "validation" if index < 89 else "test"


def decode(data, content_type):
    mime, *parameters = content_type.lower().split(";")
    if mime.strip() in {"audio/pcm", "audio/l16", "audio/x-pcm"}:
        headers = dict(tuple(value.strip() for value in field.split("=", 1))
                       for field in parameters if "=" in field)
        source_rate = int(headers.get("rate", 24000).strip('"') if "rate" in headers else 24000)
        channels = int(headers.get("channels", 1).strip('"') if "channels" in headers else 1)
        byte_order = ">" if mime.strip() == "audio/l16" else "<"
        samples = np.frombuffer(data, dtype=byte_order + "i2").astype(np.float32) / 32768
        audio = samples.reshape(-1, channels).mean(axis=1)
    else:
        audio, source_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
    return convert(audio, source_rate)


def convert(audio, source_rate):
    if not 8000 <= source_rate <= 192000 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("invalid audio")
    if source_rate != RATE:
        divisor = math.gcd(RATE, source_rate)
        audio = resample_poly(audio, RATE // divisor, source_rate // divisor)
    return np.clip(audio, -1, 1)


def save_wave(path, audio):
    temporary = path.with_suffix(".part")
    sf.write(temporary, audio, RATE, format="WAV", subtype="PCM_16")
    replace(temporary, path)


def normalize(text):
    return re.sub(r"[^а-яa-z]+", " ", text.lower().replace("ё", "е")).strip()


def word_match(text, profile=DEFAULT_PROFILE):
    return normalize(text).replace(" ", "") in profile.aliases


class InspectSpeech:
    def __init__(self, model, cache, cpu_threads=4, profile=DEFAULT_PROFILE):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model, device="cpu", compute_type="int8",
                                  cpu_threads=cpu_threads, num_workers=3, download_root=str(cache))
        self.profile = profile

    def __call__(self, audio, hint=False):
        segments, _ = self.model.transcribe(
            audio, language="ru", beam_size=3 if hint else 2, word_timestamps=True,
            condition_on_previous_text=False, hotwords=self.profile.phrase if hint else None,
        )
        segments = list(segments)
        words = [word for segment in segments for word in segment.words or []]
        transcript = " ".join(segment.text.strip() for segment in segments)
        onset = offset = None
        for index, word in enumerate(words[:4]):
            if word_match(word.word, self.profile):
                onset, offset = word.start, word.end
                break
            if index + 1 < len(words) and word_match(word.word + words[index + 1].word, self.profile):
                onset, offset = word.start, words[index + 1].end
                break
        return transcript, onset, offset


def job_for(voice, text, category, stage):
    return {
        "id": digest(voice["id"], category, text)[:24],
        "voice_id": voice["id"], "model": voice["model"], "voice": voice["voice"],
        "group": voice_group(voice), "split": split_for(voice_group(voice)),
        "stage": stage, "category": category, "text": text,
        "label": int(category == "positive"),
    }


class Synthesis:
    def __init__(self, args, root, journal):
        self.args, self.root, self.journal = args, root, journal
        self.key = api_key(args.env_file)
        self.client = httpx.Client(timeout=httpx.Timeout(45, connect=12),
                                   headers={"Authorization": "Bearer " + self.key,
                                            "X-OpenRouter-Title": "Goodwin Wake Word Dataset"})
        self.providers = collections.defaultdict(lambda: threading.Semaphore(args.per_provider))
        self.stop = threading.Event()
        self.credit_blocked_models = set()
        self.inspect = InspectSpeech(args.asr, args.cache, args.asr_threads, args.profile)
        (root / ".review").mkdir(exist_ok=True)

    def key_info(self):
        try:
            response = self.client.get(API + "/key")
        except httpx.HTTPError:
            return {}
        if response.status_code == 200:
            info = response.json()["data"]
            return {"usage": info.get("usage"), "remaining": info.get("limit_remaining")}
        return {}

    def request(self, voice, text):
        payload = {"model": voice["model"], "input": text, "response_format": voice["format"]}
        if voice["voice"]:
            payload["voice"] = voice["voice"]
        if voice["routes"] and not voice["model"].endswith(":free"):
            payload["provider"] = {"only": voice["routes"]}
        if voice["family"] == "bytedance-seed":
            payload["input"] = f'Произнеси ровно одну короткую фразу по-русски: «{text}». Без других слов и звуков.'
        if voice["instructions"] and voice["family"] == "google":
            payload["instructions"] = "Read the text verbatim in natural Russian. Do not add any words or sounds."
        if voice["model"] in {"elevenlabs/eleven-flash-v2.5", "elevenlabs/eleven-turbo-v2.5",
                              "elevenlabs/eleven-multilingual-v2"}:
            payload.setdefault("provider", {})["options"] = {"elevenlabs": {"language_code": "ru"}}
        with self.providers[voice["family"]]:
            for attempt in range(2):
                if self.stop.is_set() or voice["model"] in self.credit_blocked_models:
                    return None
                try:
                    response = self.client.post(API + "/audio/speech", json=payload)
                except httpx.HTTPError:
                    if attempt == 0:
                        time.sleep(1)
                        continue
                    return None
                busy_balance = response.status_code == 402 and "in-flight" in response.text
                if (not busy_balance and response.status_code not in {429, 500, 502, 503, 504, 529}) or attempt == 1:
                    return response
                retry = response.headers.get("retry-after", "")
                time.sleep(min(20, float(retry)) if retry.replace(".", "", 1).isdigit()
                           else 5 if busy_balance else 2)
        return None

    def generate(self, voice, job):
        if self.stop.is_set():
            return
        result = dict(job)
        response = self.request(voice, job["text"])
        if response is None:
            if not self.stop.is_set() and voice["model"] not in self.credit_blocked_models:
                self.journal.put(dict(result, status="retryable", reason="transport"))
            return
        status = response.status_code
        if status != 200:
            if status == 401:
                self.stop.set()
            if status == 402 and "in-flight" not in response.text:
                self.credit_blocked_models.add(voice["model"])
                print(f"Credit check blocked {voice['model']}; continuing other models.", flush=True)
            try:
                error = response.json().get("error", {}).get("message") or ""
                message = str(error).replace(self.key, "[redacted]")[:220]
            except (ValueError, AttributeError):
                message = ""
            self.journal.put(dict(result, status="retryable" if status in {402, 429, 500, 502, 503, 504, 529} else "http_error",
                                  reason=f"http_{status}", detail=message))
            return
        try:
            audio = decode(response.content, response.headers.get("content-type", ""))
        except (ValueError, RuntimeError, TypeError) as error:
            self.journal.put(dict(result, status="rejected", reason="decode", detail=str(error)[:180]))
            return
        if not 0.18 <= len(audio) / RATE <= 15 or np.max(np.abs(audio)) < 0.01 or np.std(audio) < 0.001:
            self.journal.put(dict(result, status="rejected", reason="silent_or_invalid_duration"))
            return
        if np.mean(np.abs(audio) >= 0.999) > 0.015:
            self.journal.put(dict(result, status="rejected", reason="clipping"))
            return

        if job["category"] == "positive":
            text, onset, offset = self.inspect(audio)
            result["transcription"] = text
            if job["stage"] == "probe":
                words = set(normalize(text).split())
                expected = set(self.args.profile.probe_words)
                coverage = len(words & expected)
                if onset is None or coverage < min(3, len(expected)) or onset > 2.5:
                    save_wave(self.root / ".review" / (job["id"] + ".wav"), audio)
                    self.journal.put(dict(result, status="rejected", reason="russian_or_wake_mismatch",
                                          duration_s=round(len(audio) / RATE, 3)))
                    return
            elif onset is None:
                save_wave(self.root / ".review" / (job["id"] + ".wav"), audio)
                self.journal.put(dict(result, status="rejected", reason="wake_alignment"))
                return
            result.update(phrase_onset_s=onset, phrase_offset_s=offset)

        folder = "positive" if job["label"] else "negative"
        relative = Path(folder) / (job["id"] + ".wav")
        save_wave(self.root / relative, audio)
        result.update(status="accepted", path=relative.as_posix(),
                      duration_s=round(len(audio) / RATE, 3),
                      provider=response.headers.get("x-openrouter-provider"),
                      generation_id=response.headers.get("x-generation-id"),
                      sha256=hashlib.sha256((self.root / relative).read_bytes()).hexdigest())
        self.journal.put(result)

    def batch(self, jobs):
        info = self.key_info()
        print(f"Requests queued: {len(jobs)}. Key: {info}", flush=True)
        if not jobs:
            return
        if info.get("remaining") is not None and info["remaining"] < self.args.reserve:
            print("Key limit reached.", flush=True)
            self.stop.set()
            return
        queues = collections.defaultdict(collections.deque)
        for voice, job in jobs:
            queues[voice["family"]].append((voice, job))
        ordered = []
        while queues:
            for family in list(queues):
                ordered.append(queues[family].popleft())
                if not queues[family]:
                    del queues[family]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.workers) as executor:
            futures = [executor.submit(self.generate, voice, job) for voice, job in ordered]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                future.result()
                if index % 25 == 0 or index == len(jobs):
                    info = self.key_info()
                    if info:
                        try:
                            write_json(self.root / "openrouter-usage.json", info)
                        except PermissionError:
                            print("Could not update the usage snapshot; generation is continuing.", flush=True)
                    print(f"Finished {index}/{len(jobs)}; key {info}; "
                          f"accepted {sum(row['status'] == 'accepted' for row in self.journal.rows.copy().values())}",
                          flush=True)
                    if info.get("remaining") is not None and info["remaining"] < self.args.reserve:
                        self.stop.set()


def export(root, journal, catalog, profile=DEFAULT_PROFILE):
    rows = [dict(row, split=split_for(row["group"])) for row in journal.rows.values()
            if row["status"] == "accepted" and (root / row["path"]).is_file()]
    background_meta = root / "background.json"
    if background_meta.exists():
        rows.extend(read_json(background_meta))
    rows.sort(key=lambda row: (row["category"], row["id"]))
    temporary = root / "manifest.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    replace(temporary, root / "manifest.jsonl")
    coverage = []
    for voice in catalog:
        probe = journal.rows.get(job_for(voice, profile.probe, "positive", "probe")["id"], {})
        coverage.append({
            **voice,
            "status": "excluded" if voice["skip"] else probe.get("status", "untested"),
            "reason": voice["skip"] or probe.get("reason"),
            "detail": probe.get("detail"),
            "transcription": probe.get("transcription"),
        })
    write_json(root / "coverage.json", coverage)
    totals = collections.Counter(row["path"].split("/")[0] for row in rows)
    duration = collections.Counter()
    per_voice = collections.defaultdict(collections.Counter)
    for row in rows:
        duration[row["path"].split("/")[0]] += row["duration_s"]
        if "voice_id" in row:
            per_voice[row["voice_id"]]["positive" if row["label"] else "negative"] += 1
    write_json(root / "summary.json", {
        "sample_rate": RATE,
        "counts": dict(totals),
        "duration_s": {category: round(seconds, 3) for category, seconds in duration.items()},
        "coverage": dict(collections.Counter(row["status"] for row in coverage)),
        "working_models": sorted({row["model"] for row in coverage if row["status"] == "accepted"}),
        "positive_voice_pairs": len({row["voice_id"] for row in rows if row["category"] == "positive"}),
        "negative_voice_pairs": len({row["voice_id"] for row in rows if row["category"] in {"speech", "adversarial"}}),
        "pairs_with_six_of_each": sum(counts["positive"] >= 6 and counts["negative"] >= 6
                                      for counts in per_voice.values()),
    })
    print(json.dumps(dict(totals), ensure_ascii=False), flush=True)


def recover(args, root, journal):
    near_spellings = set(args.profile.recovery_spellings)
    rows = []
    for row in journal.rows.values():
        if row.get("reason") != "wake_alignment" or row.get("reviewed"):
            continue
        words = normalize(row.get("transcription", "")).split()
        if words and (words[0] in near_spellings or "".join(words[:2]) in near_spellings):
            if (root / ".review" / (row["id"] + ".wav")).is_file():
                rows.append(row)
    if not rows:
        return
    print(f"Checking {len(rows)} cached clips with a spelling hint", flush=True)
    inspect = InspectSpeech(args.asr, args.cache, args.asr_threads, args.profile)

    def verify(row):
        source = root / ".review" / (row["id"] + ".wav")
        audio, _ = sf.read(source, dtype="float32")
        text, onset, offset = inspect(audio, hint=True)
        result = dict(row, reviewed=True, review_transcription=text)
        expected_tail = set(normalize(row["text"]).split()[1:])
        decoded_words = set(normalize(text).split())
        if (onset is not None and 0 <= onset < min(1.4, len(audio) / RATE)
                and offset > onset and (not expected_tail or expected_tail & decoded_words)):
            relative = Path("positive") / (row["id"] + ".wav")
            shutil.copyfile(source, root / relative)
            result.update(
                status="accepted", path=relative.as_posix(),
                duration_s=round(len(audio) / RATE, 3),
                transcription=text, unhinted_transcription=row.get("transcription"),
                phrase_onset_s=onset, phrase_offset_s=min(offset, len(audio) / RATE),
                sha256=hashlib.sha256((root / relative).read_bytes()).hexdigest(),
            )
            result.pop("reason", None)
        journal.put(result)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        for index, _ in enumerate(executor.map(verify, rows), 1):
            if index % 50 == 0 or index == len(rows):
                print(f"Reviewed {index}/{len(rows)} cached clips.", flush=True)


def derive(root, journal, profile=DEFAULT_PROFILE):
    sources = collections.defaultdict(list)
    hashes = collections.defaultdict(set)
    for row in journal.rows.values():
        if row["status"] != "accepted":
            continue
        hashes[row["voice_id"], row["label"]].add(row["sha256"])
        words = row["text"].split()
        if (row["label"] and not row.get("derived_from") and len(words) >= 3
                and word_match(words[0], profile)):
            sources[row["voice_id"]].append(row)
    added = collections.Counter()
    for voice, rows in sources.items():
        for label in (0, 1):
            seen = hashes[voice, label]
            for row in rows:
                if len(seen) >= 6:
                    break
                category = "positive" if label else "speech"
                identifier = digest("within-utterance-1", row["id"], label)[:24]
                if identifier in journal.rows:
                    continue
                audio, rate = sf.read(root / row["path"], dtype="float32")
                if rate != RATE:
                    continue
                if label:
                    start = max(0, int((row["phrase_onset_s"] - 0.08) * RATE))
                    end = min(len(audio), round(row["phrase_offset_s"] * RATE))
                    minimum = RATE // 4
                else:
                    start = round((row["phrase_offset_s"] + 0.12) * RATE)
                    end = len(audio)
                    minimum = RATE // 2
                segment = audio[start:end].copy()
                if len(segment) < minimum or np.std(segment) < 0.001:
                    continue
                fade = min(RATE // 100, len(segment) // 4)
                segment[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
                segment[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
                relative = Path("positive" if label else "negative") / (identifier + ".wav")
                save_wave(root / relative, segment)
                clip_hash = hashlib.sha256((root / relative).read_bytes()).hexdigest()
                if clip_hash in seen:
                    (root / relative).unlink()
                    continue
                new_row = {
                    **row, "id": identifier, "stage": "derived", "label": label, "category": category,
                    "path": relative.as_posix(), "duration_s": round(len(segment) / RATE, 3),
                    "text": profile.phrase if label else row["text"].split(maxsplit=1)[1],
                    "derived_from": row["id"], "source_offset_s": start / RATE, "sha256": clip_hash,
                    "derivation": "isolated_keyword" if label else "request_after_keyword",
                }
                for field in ("transcription", "unhinted_transcription", "review_transcription", "reviewed"):
                    new_row.pop(field, None)
                if label:
                    new_row["phrase_onset_s"] = max(0.0, row["phrase_onset_s"] - start / RATE)
                    new_row["phrase_offset_s"] = len(segment) / RATE
                else:
                    new_row.pop("phrase_onset_s", None)
                    new_row.pop("phrase_offset_s", None)
                journal.put(new_row)
                seen.add(clip_hash)
                added[category] += 1
    print(f"Added aligned excerpts: {dict(added)}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["catalog", "inventory", "probe", "generate", "recover", "fill", "derive", "report"],
    )
    parser.add_argument("--root", type=Path, default=Path("datasets/goodwin"))
    parser.add_argument("--profile", type=Path, help="JSON file with wake phrase, probe and generation texts")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--cache", type=Path, default=Path("target/wakeword-models"))
    parser.add_argument("--asr", default="base")
    parser.add_argument("--asr-threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--per-provider", type=int, default=4)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--voices-per-model", type=int)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--start-round", type=int, default=1)
    parser.add_argument("--reserve", type=float, default=0.08)
    parser.add_argument("--retry", action="store_true")
    args = parser.parse_args()
    root = args.root
    if args.command == "catalog":
        refresh_catalog(root, args.env_file)
        return
    args.profile = PhraseProfile(**read_json(args.profile)) if args.profile else DEFAULT_PROFILE
    profile = args.profile
    for folder in ("positive", "negative", "background"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    catalog = inventory(root)
    journal = Journal(root)
    if args.command in {"inventory", "report"}:
        export(root, journal, catalog, profile)
        return
    if args.command == "recover":
        recover(args, root, journal)
        export(root, journal, catalog, profile)
        return
    if args.command == "derive":
        derive(root, journal, profile)
        export(root, journal, catalog, profile)
        return
    selected = [row for row in catalog if not row["skip"] and (not args.model or row["model"] in args.model)]
    if args.voices_per_model:
        groups = collections.Counter()
        samples = []
        for row in selected:
            if groups[row["model"]] < args.voices_per_model:
                samples.append(row)
                groups[row["model"]] += 1
        selected = samples
    synth = Synthesis(args, root, journal)

    def pending(voice, job):
        record = journal.rows.get(job["id"])
        return (voice, job) if record is None or (args.retry and record["status"] == "retryable") else None

    try:
        if args.command == "probe":
            jobs = [pending(voice, job_for(voice, profile.probe, "positive", "probe")) for voice in selected]
            synth.batch([job for job in jobs if job])
        elif args.command == "generate":
            selected = [voice for voice in selected
                        if journal.successful(root, job_for(voice, profile.probe, "positive", "probe")["id"])]
            print(f"Working pairs: {len(selected)}", flush=True)
            round_count = max(len(profile.positive), len(profile.negative))
            for round_index in range(max(0, args.start_round - 1), min(round_count, args.rounds)):
                if synth.stop.is_set():
                    break
                jobs = []
                for voice in selected:
                    if round_index < len(profile.negative):
                        category, text = profile.negative[round_index]
                        jobs.append(pending(voice, job_for(voice, text, category, "generate")))
                    if round_index < len(profile.positive):
                        jobs.append(pending(voice, job_for(voice, profile.positive[round_index], "positive", "generate")))
                synth.batch([job for job in jobs if job])
                export(root, journal, catalog, profile)
        else:
            selected = [voice for voice in selected
                        if journal.successful(root, job_for(voice, profile.probe, "positive", "probe")["id"])]
            round_count = max(len(profile.extra_positive), len(profile.extra_negative))
            for index in range(max(0, args.start_round - 1), min(args.rounds, round_count)):
                if synth.stop.is_set():
                    break
                unique = {(row["voice_id"], row["label"], row["sha256"])
                          for row in journal.rows.values() if row["status"] == "accepted"}
                counts = collections.Counter((voice_id, label) for voice_id, label, _ in unique)
                jobs = []
                for voice in selected:
                    if index < len(profile.extra_positive) and counts[voice["id"], 1] < 6:
                        jobs.append(pending(voice, job_for(voice, profile.extra_positive[index], "positive", "fill")))
                    if index < len(profile.extra_negative) and counts[voice["id"], 0] < 6:
                        category, text = profile.extra_negative[index]
                        jobs.append(pending(voice, job_for(voice, text, category, "fill")))
                synth.batch([job for job in jobs if job])
    finally:
        export(root, journal, catalog, profile)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
