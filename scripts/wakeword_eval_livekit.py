"""Calibrate a Goodwin checkpoint and evaluate the exported streaming ONNX model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import torch
from livekit.wakeword import WakeWordModel
from livekit.wakeword_train.encoders import load_pretrained
from livekit.wakeword_train.export import calibrate, export_streaming
from livekit.wakeword_train.micro import MicroRuntime, export_micro
from onnxruntime import SessionOptions

from wakeword_train_livekit import FRAME, PHRASE, RATE, VALIDATION_FPR, Clip, clips_for

THRESHOLD = 0.5
DEBOUNCE_S = 2.0
CALIBRATION_SWEEP_TEMPERATURE = 4.0
GOODWIN_DEFAULT_THRESHOLD = 0.575


def score_streaming(
    model_path: Path, clips: list[Clip], output: Path, tier: str = "standard"
) -> list[dict]:
    if tier == "micro":
        micro_model = MicroRuntime(model_path)

        def score_clip(samples: np.ndarray) -> np.ndarray:
            return micro_model.scores(samples.astype(np.float32) / 32768)
    else:
        session = SessionOptions()
        session.intra_op_num_threads = 1
        session.inter_op_num_threads = 1
        model = WakeWordModel([model_path], sess_options=session)
        name = model.model_names[0]

        def score_clip(samples: np.ndarray) -> np.ndarray:
            model.reset()
            return np.array([model.predict(frame)[name] for frame in samples.reshape(-1, FRAME)])

    started = time.monotonic()
    results = []
    with output.open("w", encoding="utf-8") as stream:
        for index, clip in enumerate(clips, 1):
            scores = score_clip(clip.inference_pcm)
            ends = np.arange(1, len(scores) + 1) * FRAME / RATE
            valid = clip.detection_window(ends) if clip.positive else np.ones(len(scores), dtype=bool)
            if not valid.any() or not np.isfinite(scores).all():
                raise ValueError(f"Invalid scores or phrase window: {clip.meta['path']}")
            row = {
                **{k: clip.meta.get(k) for k in (
                    "id", "path", "category", "model", "voice", "text", "derived_from", "source",
                    "phrase_onset_s", "phrase_offset_s",
                )},
                "duration_s": clip.duration, "positive": clip.positive,
                "peak": float(scores[valid].max()), "clip_peak": float(scores.max()),
                "scores": scores.tolist(), "valid_frames": np.flatnonzero(valid).tolist(),
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            results.append(row)
            if index % 100 == 0 or index == len(clips):
                print(f"{output.stem} {index}/{len(clips)} in {time.monotonic() - started:.1f}s", flush=True)
    return results


def triggers(scores: list[float], threshold: float) -> list[int]:
    events = []
    for i, score in enumerate(scores):
        if score >= threshold and (not events or (i - events[-1]) * FRAME / RATE >= DEBOUNCE_S):
            events.append(i)
    return events


def wilson(successes: int, total: int) -> list[float]:
    p, z = successes / total, 1.96
    factor = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / factor
    radius = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / factor
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize(rows: list[dict], threshold: float, details: list[dict] | None = None) -> dict:
    counts = defaultdict(lambda: dict(clips=0, misses=0, false_alarm_clips=0, false_alarms=0,
                                      early_triggers=0, seconds=0.0))
    counts["positive"]
    latency = []
    for row in rows:
        category = row["category"]
        group = counts[category]
        group["clips"] += 1
        group["seconds"] += row["duration_s"]
        events = triggers(row["scores"], threshold)
        if row["positive"]:
            valid = row["valid_frames"]
            # In Goodwin an early trigger opens a session, so only the first event determines a hit.
            hit = len(events) > 0 and events[0] in valid
            group["misses"] += not hit
            early = bool(events and events[0] < valid[0])
            group["early_triggers"] += early
            if hit:
                outcome = "hit"
                latency.append((events[0] + 1) * FRAME / RATE - row["phrase_offset_s"])
            else:
                outcome = "early_trigger" if early else ("late_trigger" if events else "miss")
        else:
            group["false_alarm_clips"] += bool(events)
            group["false_alarms"] += len(events)
            outcome = "false_alarm" if events else "true_negative"
        if details is not None:
            fields = {k: v for k, v in row.items() if k not in ("scores", "valid_frames")}
            details.append({**fields, "outcome": outcome, "events": len(events),
                            "first_trigger_s": (events[0] + 1) * FRAME / RATE if events else None})
    positive = counts["positive"]
    negative = [v for k, v in counts.items() if k != "positive"]
    hours = sum(v["seconds"] for v in negative) / 3600
    alarms = sum(v["false_alarms"] for v in negative)
    negatives = sum(v["clips"] for v in negative)
    false_clips = sum(v["false_alarm_clips"] for v in negative)
    return {
        "threshold": threshold, "positive_clips": positive["clips"], "misses": positive["misses"],
        "recall": 1 - positive["misses"] / positive["clips"],
        "miss_rate_wilson_95": wilson(positive["misses"], positive["clips"]),
        "early_positive_triggers": positive["early_triggers"], "negative_clips": negatives,
        "false_alarm_clips": false_clips, "false_clip_rate": false_clips / negatives,
        "false_clip_rate_wilson_95": wilson(false_clips, negatives),
        "false_alarm_events": alarms, "negative_hours": hours, "false_alarms_per_hour": alarms / hours,
        "latency_median_ms": float(np.median(latency) * 1000) if latency else None,
        "latency_p95_ms": float(np.quantile(latency, 0.95) * 1000) if latency else None,
        "categories": dict(counts),
    }


def export_checkpoint(args: argparse.Namespace) -> tuple[torch.nn.Module, dict, str]:
    os.environ["LIVEKIT_WAKEWORD_ENCODERS"] = str(args.encoders.resolve())
    torch.set_num_threads(4)
    state = torch.load(args.out / "selected.pt", map_location="cpu", weights_only=True)
    phrase = state.get("phrase", PHRASE)
    det = load_pretrained(args.tier).bake(phrase)
    det.load_state_dict(state["state"])
    return det.eval(), state["validation"], phrase


def select_threshold(rows: list[dict], budget: int) -> dict:
    negative_peaks = sorted(row["peak"] for row in rows if not row["positive"])
    # Sweep the region that meets the negative clip budget, using first-trigger recall.
    scores = sorted({s for row in rows for s in row["scores"]})
    minimum = negative_peaks[-(budget + 1)] + 2e-6
    candidates = np.unique(np.r_[
        minimum, np.linspace(minimum, max(minimum, max(scores)), 192),
        [row["peak"] + 2e-6 for row in rows if row["peak"] >= minimum],
    ])
    return max(
        (summarize(rows, float(threshold)) for threshold in candidates if threshold < 1),
        key=lambda result: (result["recall"], -result["false_alarm_clips"], -result["threshold"]),
    )


def write_test(rows: list[dict], threshold: float, path: Path) -> dict:
    detail: list[dict] = []
    result = summarize(rows, threshold, detail)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)
    return result


def run(args: argparse.Namespace) -> None:
    if args.tier == "micro" and find_spec("onnxsim") is None:
        raise RuntimeError("Install onnxsim to fold the Micro ONNX padding for the Goodwin runtime")
    det, selection, phrase = export_checkpoint(args)
    metadata = {"wake_word": phrase, "language": "ru", "format": "livekit-wakeword-v2"}
    validation = clips_for(args.dataset, "validation")
    calibration_detector = calibrate(deepcopy(det), 0.0, CALIBRATION_SWEEP_TEMPERATURE)
    if args.tier == "micro":
        path = export_micro(calibration_detector, args.out, "uncalibrated", tflite=False)["onnx"]
    else:
        path = export_streaming(calibration_detector, args.out / "uncalibrated.onnx", metadata=metadata)
    raw = score_streaming(path, validation, args.out / "validation_raw_scores.jsonl", args.tier)
    budget = int(sum(not row["positive"] for row in raw) * VALIDATION_FPR)
    operating = select_threshold(raw, budget)
    raw_threshold = operating["threshold"]
    logit = float(np.log(raw_threshold / (1 - raw_threshold)) * CALIBRATION_SWEEP_TEMPERATURE)
    output_temperature = 4.0 if args.tier == "micro" else 2.0
    calibrated = calibrate(det, logit, temperature=output_temperature)
    if args.tier == "micro":
        model = export_micro(calibrated, args.out, args.name, tflite=False)["onnx"]
    else:
        model = export_streaming(calibrated, args.out / f"{args.name}.onnx",
                                 metadata={**metadata, "threshold": str(THRESHOLD)})
    validated = score_streaming(model, validation, args.out / "validation_scores.jsonl", args.tier)
    validation_result = summarize(validated, THRESHOLD)
    if validation_result["false_alarm_clips"] > budget:
        raise ValueError("Exported model exceeded the validation false-alarm budget")
    # Freeze the operating point before loading or scoring the held-out split.
    calibration = {
        "selected_step": selection["step"], "sweep_probability_threshold": raw_threshold,
        "sweep_temperature": CALIBRATION_SWEEP_TEMPERATURE,
        "calibration_logit": logit, "score_threshold": THRESHOLD, "debounce_s": DEBOUNCE_S,
        "tier": args.tier, "phrase": phrase, "output_temperature": output_temperature,
        "validation_dataset": args.dataset.as_posix(),
        "validation_false_clip_budget": budget, "validation": validation_result,
    }
    (args.out / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    test_dataset = args.test_dataset or args.dataset
    test = score_streaming(model, clips_for(test_dataset, "test"), args.out / "test_scores.jsonl", args.tier)
    result = write_test(test, THRESHOLD, args.out / "test_predictions.csv")
    comparison = (
        {"test_at_goodwin_default_threshold": summarize(test, GOODWIN_DEFAULT_THRESHOLD)}
        if args.tier == "standard" else {}
    )
    (args.out / "evaluation.json").write_text(json.dumps({
        **calibration, "test_dataset": test_dataset.as_posix(), "test": result, **comparison,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/goodwin"))
    parser.add_argument("--test-dataset", type=Path, help="Read the held-out split from a different corpus")
    parser.add_argument("--encoders", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default="goodwin", help="File stem for the exported model")
    parser.add_argument("--tier", choices=("standard", "micro"), default="standard")
    run(parser.parse_args())
