"""Fine-tune LiveKit WakeWord v2.1.3 on Goodwin's local split manifest."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import time
import wave
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from livekit.wakeword_train import data, loop
from livekit.wakeword_train.encoders import load_pretrained
from livekit.wakeword_train.nn import N_CTX

RATE = 16000
FRAME = 1280
WARMUP = 25
PHRASE = "Гудвин"
VALIDATION_FPR = 0.01


@dataclass
class Clip:
    meta: dict
    pcm: np.ndarray

    @property
    def positive(self) -> bool:
        return self.meta["label"] == 1

    @property
    def duration(self) -> float:
        return len(self.pcm) / RATE

    def detection_window(self, times: np.ndarray) -> np.ndarray:
        end = self.meta["phrase_offset_s"]
        return (times >= max(self.meta["phrase_onset_s"], end - 0.25)) & (times <= end + 0.5)

    @property
    def inference_pcm(self) -> np.ndarray:
        # A standalone wake word at the end of a WAV still gets the normal detector response time.
        trailing = max(0, self.meta["phrase_offset_s"] + 0.5 - self.duration) if self.positive else 0
        target = math.ceil((len(self.pcm) + trailing * RATE) / FRAME) * FRAME
        return np.pad(self.pcm, (0, target - len(self.pcm)))


def clips_for(root: Path, split: str) -> list[Clip]:
    def read(meta: dict) -> Clip:
        with wave.open(str(root / meta["path"]), "rb") as source:
            if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (RATE, 1, 2):
                raise ValueError(f"Unexpected WAV format: {meta['path']}")
            return Clip(meta, np.frombuffer(source.readframes(source.getnframes()), dtype="<i2"))

    with (root / "manifest.jsonl").open(encoding="utf-8") as manifest:
        return [read(row) for line in manifest if (row := json.loads(line))["split"] == split]


def local_training_data(
    clips: list[Clip],
) -> tuple[dict[str, data.ClipSet], np.ndarray, list[list[np.ndarray]]]:
    by_category = defaultdict(list)
    for clip in clips:
        by_category[clip.meta["category"]].append(clip)
    sets = {}
    for category in ("positive", "adversarial", "speech"):
        rows = by_category[category]
        lengths = np.array([len(clip.pcm) for clip in rows], dtype=np.int64)
        offsets = lengths.cumsum() - lengths
        starts = np.array([round(clip.meta.get("phrase_onset_s", 0) * RATE) for clip in rows])
        ends = np.array([
            min(len(clip.pcm), round(clip.meta["phrase_offset_s"] * RATE))
            if clip.positive else len(clip.pcm)
            for clip in rows
        ])
        sets[category] = data.ClipSet(
            np.concatenate([clip.pcm for clip in rows]), offsets, lengths, starts, ends
        )
    noise = []
    for clip in by_category["noise"]:
        pcm = clip.pcm.astype(np.float32) / 32768
        rms = np.sqrt(np.mean(pcm**2))
        if rms > 1e-5:
            noise.append(np.clip(pcm / rms * 0.1, -1, 1).astype(np.float16))
    real_speech = [clip.pcm for clip in by_category["real_speech"]]
    ordinary_speech = [clip.pcm for clip in by_category["speech"]]
    return sets, np.concatenate(noise), [real_speech, ordinary_speech]


def adjust_training_clips(
    clips: list[Clip], label_delay_s: float, close_pattern: re.Pattern | None, extra_copies: int,
) -> list[Clip]:
    adjusted = []
    for clip in clips:
        if clip.positive and label_delay_s:
            offset = min(clip.duration, clip.meta["phrase_offset_s"] + label_delay_s)
            clip = replace(clip, meta={**clip.meta, "phrase_offset_s": offset})
        adjusted.append(clip)
        if close_pattern and clip.meta["category"] == "adversarial" and close_pattern.search(clip.meta.get("text") or ""):
            adjusted.extend([clip] * extra_copies)
    return adjusted


class LocalSpeech:
    def __init__(self, pools: list[list[np.ndarray]]) -> None:
        self.pools = pools

    def crop(self, rng: np.random.Generator, length: int) -> np.ndarray:
        clips = self.pools[rng.integers(len(self.pools))]
        audio = clips[rng.integers(len(clips))]
        if len(audio) >= length:
            start = rng.integers(len(audio) - length + 1)
            return audio[start:start + length]
        start = rng.integers(length - len(audio) + 1)
        return np.pad(audio, (start, length - len(audio) - start))


@torch.inference_mode()
def validation_logits(det: torch.nn.Module, clips: list[Clip], batch_size: int = 12) -> np.ndarray:
    audio = [clip.inference_pcm for clip in clips]
    order = sorted(range(len(clips)), key=lambda index: len(audio[index]))
    peaks = np.empty(len(clips))
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        frames = max(len(audio[i]) for i in indices) // FRAME
        batch = np.zeros((len(indices), (WARMUP + frames) * FRAME), dtype=np.float32)
        for j, i in enumerate(indices):
            batch[j, WARMUP * FRAME:WARMUP * FRAME + len(audio[i])] = audio[i] / 32768
        logits = det(torch.from_numpy(batch)).numpy()[:, WARMUP - (N_CTX - 1):]
        for j, i in enumerate(indices):
            scores = logits[j, :len(audio[i]) // FRAME]
            times = np.arange(1, len(scores) + 1) * FRAME / RATE
            peaks[i] = scores[clips[i].detection_window(times)].max() if clips[i].positive else scores.max()
    if not np.isfinite(peaks).all():
        raise ValueError("Non-finite validation score")
    return peaks


def measure_peaks(labels: np.ndarray, peaks: np.ndarray, fpr: float = VALIDATION_FPR) -> dict:
    positives = peaks[labels]
    negatives = peaks[~labels]
    allowed = math.floor(len(negatives) * fpr)
    threshold = float(np.nextafter(np.sort(negatives)[-(allowed + 1)], np.inf))
    order = np.argsort(-peaks, kind="stable")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return {
        "threshold": threshold,
        "average_precision": float(precision[sorted_labels].mean()),
        "validation_clip_bce": float(np.logaddexp(0, np.where(labels, -peaks, peaks)).mean()),
        "recall_at_validation_fpr": float(np.mean(positives >= threshold)),
        "validation_false_clips": int(np.count_nonzero(negatives >= threshold)),
    }


def save_checkpoint(det: torch.nn.Module, path: Path, metrics: dict, phrase: str) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"state": det.state_dict(), "validation": metrics, "phrase": phrase}, tmp)
    tmp.replace(path)


def train(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["LIVEKIT_WAKEWORD_ENCODERS"] = str(args.encoders.resolve())
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    train_clips = clips_for(args.dataset, "train")
    val_clips = clips_for(args.dataset, "validation")
    adjusted = adjust_training_clips(
        train_clips, args.label_delay_ms / 1000, args.close_negative_pattern, args.extra_close_negative_copies,
    )
    sets, noise, pools = local_training_data(adjusted)
    labels = np.array([clip.positive for clip in val_clips])
    detector = load_pretrained(args.tier).bake(args.phrase).eval()
    micro = args.tier == "micro"
    teacher = None
    teacher_step = None
    if micro:
        checkpoint = torch.load(args.teacher, map_location="cpu", weights_only=True)
        if checkpoint.get("phrase", PHRASE) != args.phrase:
            raise ValueError("Teacher checkpoint uses a different wake phrase")
        teacher = load_pretrained("standard").bake(args.phrase).eval()
        teacher.load_state_dict(checkpoint["state"])
        teacher_step = checkpoint["validation"]["step"]
    party_prob = args.party_probability
    if party_prob is None:
        party_prob = 0.6 if micro else 0.15
    kd_weight = args.distillation_weight
    if kd_weight is None:
        kd_weight = 2.0 if micro else 1.0
    aug = replace(loop.default_aug(detector), p_rir=0.0, p_party=party_prob)
    cfg = loop.TrainConfig(
        steps=args.steps, n_pos=12, n_adv=args.near_miss_count,
        n_speech=18 - args.near_miss_count, n_noise=2,
        warmup=max(1, min(100, args.steps // 10)), workers=0, seed=args.seed,
        ema=0.98, neg_weight=(1.0, 2.0), aug=aug,
        ft_top_blocks=None if micro else 2, kd_weight=kd_weight,
    )
    (args.out / "run.json").write_text(json.dumps({
        "livekit_version": "2.1.3", "torch_version": torch.__version__, "device": "cpu",
        "phrase": args.phrase, "seed": cfg.seed, "steps": cfg.steps, "threads": args.threads, "batch": 32,
        "tier": args.tier, "teacher_checkpoint_step": teacher_step,
        "training_loss": "weighted_frame_bce_plus_distillation" if micro else "weighted_frame_bce",
        "fine_tuned_top_blocks": cfg.ft_top_blocks, "checkpoint_every": args.checkpoint_every,
        "train_clips": len(train_clips), "validation_clips": len(val_clips),
        "validation_fpr_budget": VALIDATION_FPR, "room_impulse_response": False,
        "party_probability": cfg.aug.p_party, "distillation_weight": cfg.kd_weight,
        "near_miss_rows_per_batch": cfg.n_adv,
        "training_label_delay_ms": args.label_delay_ms,
        "close_negative_pattern": args.close_negative_pattern.pattern if args.close_negative_pattern else None,
        "extra_close_negative_copies": args.extra_close_negative_copies,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    best_rank = (-1.0, -1.0)
    with (args.out / "training.csv").open("w", newline="", encoding="utf-8") as train_file, (
        args.out / "validation.jsonl"
    ).open("w", encoding="utf-8") as val_file:
        writer = csv.writer(train_file)
        writer.writerow(["step", "training_loss", "head_learning_rate", "elapsed_s"])

        def check(step: int, model: torch.nn.Module) -> None:
            nonlocal best_rank
            peaks = validation_logits(model, val_clips)
            metrics = {"step": step, **measure_peaks(labels, peaks)}
            val_file.write(json.dumps(metrics) + "\n")
            val_file.flush()
            print(f"Validation {step}: {metrics}", flush=True)
            rank = (metrics["recall_at_validation_fpr"], metrics["average_precision"])
            if rank > best_rank:
                best_rank = rank
                save_checkpoint(model, args.out / "selected.pt", metrics, args.phrase)
            save_checkpoint(model, args.out / "latest.pt", metrics, args.phrase)

        def on_step(step: int, ema: torch.nn.Module, loss: float, lr: float) -> None:
            writer.writerow([step, loss, lr, round(time.monotonic() - started, 3)])
            train_file.flush()
            if step % args.checkpoint_every == 0 or step == args.steps:
                check(step, ema)

        check(0, detector)
        with (
            patch.object(data, "load_clipsets", lambda _: sets),
            patch.object(data, "noise_pool", lambda: noise),
            patch.object(data, "rir_pool", lambda: np.zeros((1, 1), dtype=np.float32)),
            patch.object(data, "negative_speech_files", lambda: []),
            patch.object(data, "StreamPool", lambda *a, **kw: LocalSpeech(pools)),
        ):
            loop.train(
                detector, str(args.dataset), cfg, device="cpu", log_every=25, on_step=on_step,
                teacher=teacher,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/goodwin"))
    parser.add_argument("--phrase", default=PHRASE)
    parser.add_argument("--encoders", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tier", choices=("standard", "micro"), default="standard")
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--party-probability", type=float)
    parser.add_argument("--distillation-weight", type=float)
    parser.add_argument("--near-miss-count", type=int, choices=range(1, 18), default=8, metavar="N")
    parser.add_argument("--label-delay-ms", type=int, default=0, help="Extend positive training labels after word end")
    parser.add_argument("--close-negative-pattern", help="Case-insensitive regex for adversarial examples to oversample")
    parser.add_argument("--extra-close-negative-copies", type=int, default=0)
    args = parser.parse_args()
    if args.tier == "micro" and not args.teacher:
        parser.error("--tier micro requires --teacher pointing to the selected Standard checkpoint")
    if args.label_delay_ms < 0 or args.extra_close_negative_copies < 0:
        parser.error("label delay and extra copies must be nonnegative")
    if bool(args.close_negative_pattern) != bool(args.extra_close_negative_copies):
        parser.error("--close-negative-pattern and --extra-close-negative-copies must be used together")
    if args.close_negative_pattern:
        try:
            args.close_negative_pattern = re.compile(args.close_negative_pattern, re.IGNORECASE)
        except re.error as error:
            parser.error(f"Invalid --close-negative-pattern: {error}")
    train(args)
