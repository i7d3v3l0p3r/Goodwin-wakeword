"""Render Goodwin LiveKit training curves and held-out streaming evaluation plots."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

GREEN = "#087a51"
ORANGE = "#b75522"
SLATE = "#526278"
BLUE = "#325b91"
CATEGORIES = {"adversarial": "Near misses", "speech": "Ordinary requests",
              "noise": "Noise + music", "real_speech": "Recorded speech"}


def jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip() and line.rstrip().endswith("}")]


def save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=160, facecolor=fig.get_facecolor())
    fig.savefig(out / f"{name}.svg", facecolor=fig.get_facecolor())
    plt.close(fig)


def training(run: Path, out: Path) -> None:
    with (run / "training.csv").open(encoding="utf-8", newline="") as file:
        train = [row for row in csv.DictReader(file) if all(value for value in row.values())]
    val = jsonl(run / "validation.jsonl")
    config = json.loads((run / "run.json").read_text(encoding="utf-8"))
    tier = config.get("tier", "standard")
    step = np.array([int(r["step"]) for r in train])
    loss = np.array([float(r.get("training_loss", r.get("weighted_frame_bce"))) for r in train])
    steps_val = [r["step"] for r in val]
    fig, axes = plt.subplots(2, 2, figsize=(12.3, 7.2), layout="constrained")
    phrase = config.get("phrase", "Goodwin")
    fig.suptitle(f"{phrase} · LiveKit WakeWord v2 {tier.title()} fine-tuning", fontsize=17)
    ax = axes[0, 0]
    ax.plot(step, loss, color=GREEN, alpha=0.23, label="Batch")
    width = min(25, len(loss))
    if width:
        mean = np.convolve(loss, np.ones(width) / width, mode="valid")
        ax.plot(step[width - 1:], mean, color=GREEN, label=f"{width}-step mean")
    loss_label = "Weighted frame BCE + distillation" if tier == "micro" else "Weighted frame BCE"
    ax.set(title="Training loss", ylabel=loss_label, xlabel="Optimizer step")
    ax.legend()
    ax = axes[0, 1]
    ax.plot(steps_val, [r["validation_clip_bce"] for r in val], ".-", color=BLUE)
    ax.set(title="Validation loss", ylabel="Clip-peak BCE", xlabel="Optimizer step")
    ax = axes[1, 0]
    ax.plot(steps_val, [r["recall_at_validation_fpr"] for r in val], "o-", color=GREEN,
            label="Recall at ≤ 1% validation FPR")
    ax.plot(steps_val, [r["average_precision"] for r in val], "s--", color=BLUE,
            label="Average precision")
    ax.set(title="Validation detection quality", ylabel="Clip metric", xlabel="Optimizer step", ylim=(0, 1.03))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="lower right")
    ax = axes[1, 1]
    rates = np.array([float(r["head_learning_rate"]) for r in train])
    ax.plot(step, rates, color=GREEN)
    ax.set(title="Learning rate schedule", ylabel="Head learning rate", xlabel="Optimizer step")
    if (run / "calibration.json").exists():
        selected = json.loads((run / "calibration.json").read_text())["selected_step"]
        for ax in axes.flat:
            ax.axvline(selected, color=SLATE, linewidth=1, linestyle=":")
        axes[0, 1].text(.02, .05, f"Selected checkpoint · step {selected}",
                        transform=axes[0, 1].transAxes, color=SLATE)
    save(fig, out, "training")


def evaluation(run: Path, out: Path) -> None:
    from wakeword_eval_livekit import summarize

    report = json.loads((run / "evaluation.json").read_text())
    threshold = report["score_threshold"]
    val, test = (jsonl(run / f"{split}_scores.jsonl") for split in ("validation", "test"))
    points = np.unique(np.r_[np.linspace(0.001, 0.999, 170), threshold])
    runs = [(val, "Validation", BLUE), (test, "Held-out test", GREEN)]
    quantization_report = run / "tflite_evaluation.json"
    quantized = json.loads(quantization_report.read_text()) if quantization_report.exists() else None
    if quantized is not None:
        runs.append((jsonl(run / "tflite_test_scores.jsonl"), "INT8 held-out test", ORANGE))
        points = np.unique(np.r_[points, quantized["score_threshold"]])
    fig, axes = plt.subplots(2, 2, figsize=(12.3, 8), layout="constrained")
    fig.suptitle("Goodwin · Streaming threshold and score distributions", fontsize=17)
    for rows, label, color in runs:
        curve = [summarize(rows, float(point)) for point in points]
        recall = np.array([r["recall"] for r in curve])
        fpr = np.array([r["false_clip_rate"] for r in curve])
        alarms = [r["false_alarms_per_hour"] for r in curve]
        tp = recall * curve[0]["positive_clips"]
        fp = fpr * curve[0]["negative_clips"]
        precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=tp + fp > 0)
        axes[0, 0].plot(points, 1 - recall, color=color, label=label)
        axes[0, 1].plot(points, alarms, color=color, label=label)
        axes[1, 0].plot(recall, precision, color=color, label=label)
    for ax in axes[0]:
        ax.axvline(threshold, color=SLATE, linestyle=":", label="Selected on validation")
        if quantized is not None and quantized["score_threshold"] != threshold:
            ax.axvline(quantized["score_threshold"], color=ORANGE, linestyle=":",
                       label="INT8 threshold")
        ax.set_xlabel("Streaming threshold")
        ax.legend()
    axes[0, 0].set(title="Miss rate by threshold", ylabel="Positive clips missed")
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 1].set(title="False alarms by threshold", ylabel="Events per hour of negative audio")
    axes[1, 0].set(title="Precision and recall", xlabel="Recall", ylabel="Precision")
    axes[1, 0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1, 0].legend()
    ax = axes[1, 1]
    bins = np.linspace(0, 1, 31)
    for label, color, is_positive in (("Positives in phrase window", GREEN, True),
                                     ("Negatives", ORANGE, False)):
        peaks = [row["peak"] for row in test if row["positive"] == is_positive]
        ax.hist(peaks, bins=bins, weights=np.ones(len(peaks)) / len(peaks),
                color=color, alpha=.6, label=label)
    ax.axvline(threshold, color=SLATE, linestyle=":")
    ax.set(title="Test score distributions", xlabel="Highest score per clip", ylabel="Share in class")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()
    save(fig, out, "thresholds")

    with (run / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as file:
        predictions = list(csv.DictReader(file))
    result = report["test"]
    fig, axes = plt.subplots(2, 2, figsize=(12.3, 7.8), layout="constrained")
    fig.suptitle(f"Goodwin · Held-out test at threshold {threshold:.2f}", fontsize=17)
    true_pos = result["positive_clips"] - result["misses"]
    true_neg = result["negative_clips"] - result["false_alarm_clips"]
    ax = axes[0, 0]
    values = [[true_pos, result["misses"]], [result["false_alarm_clips"], true_neg]]
    ax.imshow(np.array(values) / np.sum(values, axis=1, keepdims=True), cmap="Greens", vmin=0, vmax=1)
    for i, line in enumerate(values):
        for j, n in enumerate(line):
            ax.text(j, i, str(n), ha="center", va="center", fontsize=16,
                    color="white" if n / sum(line) > .6 else "#14251e")
    ax.set(title="Confusion matrix · clips", yticks=[0, 1], yticklabels=["Positive", "Negative"],
           xticks=[0, 1], xticklabels=["Hit / false alarm", "Miss / correct rejection"])
    ax = axes[0, 1]
    categories = [key for key in CATEGORIES if key in result["categories"]]
    groups = [result["categories"][key] for key in categories]
    bars = ax.barh(list(range(len(groups))), [g["false_alarm_clips"] / g["clips"] for g in groups],
                   color=ORANGE)
    ax.bar_label(bars, labels=[f'{g["false_alarm_clips"]}/{g["clips"]}' for g in groups], padding=5)
    ax.set(title="False activations by category",
           xlabel="Share of negative clips activated",
           xlim=(0, max(.06, max(bar.get_width() for bar in bars) * 1.3)))
    ax.set_yticks(range(len(groups)), [CATEGORIES[key] for key in categories])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax = axes[1, 0]
    positive = [r for r in predictions if r["positive"] == "True"]
    groups = defaultdict(list)
    for row in positive:
        groups[row["model"].split("/")[0]].append(row)
    labels, misses, totals = [], [], []
    for model_family, rows in sorted(groups.items()):
        labels.append(model_family)
        misses.append(sum(row["outcome"] != "hit" for row in rows))
        totals.append(len(rows))
    bars = ax.barh(range(len(labels)), np.divide(misses, totals), color=ORANGE)
    ax.bar_label(bars, labels=[f"{m}/{t}" for m, t in zip(misses, totals)], padding=5)
    ax.set(title="Misses by audio source", xlabel="Share of positive clips missed",
           xlim=(0, max(.10, max(bar.get_width() for bar in bars) * 1.3)))
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax = axes[1, 1]
    latency = np.array([(float(r["first_trigger_s"]) - float(r["phrase_offset_s"])) * 1000
                        for r in positive if r["outcome"] == "hit"])
    if latency.size:
        ax.hist(latency, bins=np.arange(-260, 521, 40), color=GREEN)
        ax.axvline(np.median(latency), linestyle=":", color=SLATE,
                   label=f"Median {np.median(latency):.0f} ms")
        ax.legend()
    ax.set(title="Detection latency · true hits", xlabel="Milliseconds relative to wake word end",
           ylabel="Clips")
    save(fig, out, "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--training-only", action="store_true")
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .22,
                         "axes.axisbelow": True, "figure.facecolor": "#fcfdfc",
                         "svg.fonttype": "none", "legend.frameon": False})
    path = args.run / "charts"
    path.mkdir(exist_ok=True)
    training(args.run, path)
    if not args.training_only:
        evaluation(args.run, path)
