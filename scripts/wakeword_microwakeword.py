import argparse
import hashlib
import json
import shutil
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pymicro_features import MicroFrontend


RATE = 16_000
FEATURE_STEP = 160
FEATURE_WINDOW = 480
STRIDE = 3
SEED = 327
SPLITS = {"train": "training", "validation": "validation", "test": "testing"}
PREROLL = 120 * FEATURE_STEP


def read_pcm(root, row):
    with wave.open(str(root / row["path"]), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (RATE, 1, 2):
            raise ValueError(f"Unexpected WAV format in {row['path']}")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")


def frontend(pcm):
    data = memoryview(pcm.astype("<i2", copy=False)).cast("B")
    worker = MicroFrontend()
    frames = []
    for start in range(0, len(data) - 2 * FEATURE_STEP + 1, 2 * FEATURE_STEP):
        result = worker.process_samples(bytes(data[start : start + 2 * FEATURE_STEP]))
        if result.samples_read != FEATURE_STEP:
            raise ValueError("Unexpected microfrontend stride")
        if len(result.features):
            frames.append(result.features)
    return np.asarray(frames, dtype=np.float32).reshape(-1, 40)


def load_rows(root):
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text("utf-8").splitlines()]
    speakers = defaultdict(set)
    recordings = defaultdict(set)
    for row in rows:
        speakers[row["group"]].add(row["split"])
        if row.get("sha256"):
            recordings[row["sha256"]].add(row["split"])
    if any(len(splits) > 1 for splits in speakers.values()):
        raise ValueError("A corpus group appears in more than one split")
    if any(len(splits) > 1 for splits in recordings.values()):
        raise ValueError("An audio hash appears in more than one split")
    return rows


def feature_group(row):
    if row["category"] in {"noise", "real_speech"}:
        return "ambient"
    return row["category"]


def preceding_audio(pcm, end):
    duration = 2 * RATE
    start = max(0, end - duration)
    return np.pad(pcm[start:end], (max(0, duration - end), max(0, end - len(pcm))))


def mix_background(audio, backgrounds, rng):
    noise = backgrounds[rng.integers(len(backgrounds))]
    if len(noise) < len(audio):
        noise = np.resize(noise, len(audio))
    offset = rng.integers(len(noise) - len(audio) + 1)
    noise = noise[offset : offset + len(audio)].astype(np.float32)
    strength = np.sqrt(np.mean(noise**2))
    if strength <= 1:
        return audio
    speech = audio.astype(np.float32)
    level = np.sqrt(np.mean(speech**2))
    gain = level / strength * 10 ** (-rng.uniform(8, 24) / 20)
    return np.clip(speech + gain * noise, -32768, 32767).astype(np.int16)


def prepare(root, rows, out, steps):
    import yaml
    from mmap_ninja.ragged import RaggedMmap

    out = out.resolve()
    features_dir = out / "features"
    if features_dir.exists():
        raise FileExistsError(features_dir)
    rng = np.random.default_rng(SEED)
    groups = defaultdict(list)
    for row in rows:
        groups[row["split"], feature_group(row)].append(row)

    backgrounds = [read_pcm(root, row) for row in groups["train", "ambient"]]
    backgrounds = [pcm for pcm in backgrounds if len(pcm) >= RATE and np.any(pcm)]

    def make_features(group_rows, split, category):
        for row in group_rows:
            pcm = read_pcm(root, row)
            if row["label"]:
                delays = (0.0, 0.06, 0.12, 0.18) if split == "train" else (0.09,)
                endpoints = [round((row["phrase_offset_s"] + delay) * RATE) for delay in delays]
            elif split == "train" and category != "ambient":
                # Cover the early parts of each utterance as well as its last window.
                endpoints = [round((time + rng.uniform(-0.06, 0.06)) * RATE)
                             for time in (0.25, 0.45, 0.60, 0.75, 0.95, 1.20, 1.55)]
                endpoints = [end for end in endpoints if end < len(pcm)] + [len(pcm)]
            else:
                yield np.clip(frontend(pcm), 0, 26)
                continue
            for index, end in enumerate(endpoints):
                audio = preceding_audio(pcm, end)
                if backgrounds and split == "train":
                    augment = index >= 2 if row["label"] else index % 3 == 2
                    if augment:
                        audio = mix_background(audio, backgrounds, rng)
                yield np.clip(frontend(audio), 0, 26)

    for (split, category), group_rows in sorted(groups.items()):
        # Raw test WAVs are scored only after checkpoint and threshold selection.
        if split == "test":
            continue
        folder = SPLITS[split]
        if split == "validation" and category == "ambient":
            folder += "_ambient"
        dest = features_dir / category / folder / "audio_mmap"
        dest.parent.mkdir(parents=True)
        print(f"{split} {category}: {len(group_rows)} clips", flush=True)
        RaggedMmap.from_generator(str(dest), make_features(group_rows, split, category), batch_size=128)

    first, second = steps * 2 // 3, steps // 4
    config = {
        "train_dir": (out / "trainer").as_posix(),
        "window_step_ms": 10,
        "clip_duration_ms": 240,
        "features": [
            {
                "features_dir": (features_dir / category).as_posix(),
                "sampling_weight": weight,
                "penalty_weight": 1.0,
                "truth": category == "positive",
                "truncation_strategy": strategy,
                "type": "mmap",
            }
            for category, weight, strategy in (
                ("positive", 4.0, "truncate_start"),
                ("adversarial", 3.0, "truncate_start"),
                ("speech", 2.5, "truncate_start"),
                ("ambient", 2.0, "random"),
            )
        ],
        "training_steps": [first, second, steps - first - second],
        "learning_rates": [0.001, 0.0004, 0.00015],
        "positive_class_weight": [1.0],
        "negative_class_weight": [2.0],
        "batch_size": 64,
        "eval_step_interval": max(100, steps // 10),
        "time_mask_max_size": [6],
        "time_mask_count": [1],
        "freq_mask_max_size": [3],
        "freq_mask_count": [1],
        "minimization_metric": None,
        "target_minimization": 0.0,
        "maximization_metric": "average_viable_recall",
    }
    (out / "training.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def trainer_setup(out, threads):
    import tensorflow as tf
    from microwakeword import mixednet
    from microwakeword.data import FeatureHandler
    from microwakeword.model_train_eval import load_config

    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(SEED)

    parser = argparse.ArgumentParser()
    mixednet.model_parameters(parser)
    flags = parser.parse_args([
        "--pointwise_filters", "32,32,48,48",
        "--repeat_in_block", "1,1,1,1",
        "--mixconv_kernel_sizes", "[5],[7,11],[9,15],[17]",
        "--residual_connection", "0,0,0,0",
        "--first_conv_filters", "24",
        "--first_conv_kernel_size", "5",
        "--stride", str(STRIDE),
    ])
    flags.training_config = str(out / "training.yaml")
    config = load_config(flags, mixednet)
    return mixednet, flags, config, FeatureHandler(config)


def train(out, threads, resume):
    from absl import logging
    from microwakeword.model_train_eval import train_model

    logging.set_verbosity(logging.INFO)
    architecture, flags, config, data = trainer_setup(out, threads)
    model = architecture.model(flags, config["training_input_shape"], config["batch_size"])
    train_model(config, model, data, resume)


def export(out, threads, checkpoint):
    from microwakeword import utils
    from microwakeword.layers.modes import Modes

    architecture, flags, config, data = trainer_setup(out, threads)
    model = architecture.model(flags, config["training_input_shape"], batch_size=1)
    model.load_weights(Path(config["train_dir"]) / f"{checkpoint}.weights.h5")
    name = Path(checkpoint).name
    folder = f"exports/{name}"
    utils.convert_model_saved(model, config, folder, Modes.STREAM_INTERNAL_STATE_INFERENCE)
    model_path = Path(config["train_dir"]) / folder
    utils.convert_saved_model_to_tflite(
        config, data, str(model_path), str(out / "candidates"), f"{name}.tflite", quantize=True,
    )


def labeled_features(root, rows, split):
    clips = []
    for row in rows:
        if row["split"] != split:
            continue
        pcm = read_pcm(root, row)
        duration = len(pcm) / RATE
        end = row.get("phrase_offset_s") if row["label"] else None
        trailing = round(max(0, end + 0.5 - duration) * RATE) if end is not None else 0
        features = frontend(np.pad(pcm, (PREROLL, trailing)))
        # Undo pymicro-features scaling, then apply ESPHome's integer rounding.
        raw = np.rint(features * 25.6).astype(np.int32)
        quantized = np.clip((raw * 256 + 333) // 666 - 128, -128, 127).astype(np.int8)
        frame_count = len(features) // STRIDE
        chunks = quantized[:frame_count * STRIDE].reshape(-1, 1, STRIDE, 40)
        times = (np.arange(1, frame_count + 1) * STRIDE * FEATURE_STEP
                 + FEATURE_WINDOW - FEATURE_STEP - PREROLL) / RATE
        clips.append((row, duration, chunks, times))
    return clips


def stream_scores(model, clips):
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    model_bytes = model.read_bytes()
    traces = []
    for row, duration, chunks, times in clips:
        # Resource variables must begin in the exported initial state for every clip.
        interpreter = Interpreter(
            model_content=model_bytes,
            num_threads=1,
            experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        interpreter.allocate_tensors()
        [input_tensor] = interpreter.get_input_details()
        [output_tensor] = interpreter.get_output_details()
        scale, zero_point = input_tensor["quantization"]
        if (tuple(input_tensor["shape"]) != (1, STRIDE, 40) or input_tensor["dtype"] != np.int8
                or not np.isclose(scale, 26 / 255) or zero_point != -128
                or output_tensor["dtype"] != np.uint8 or tuple(output_tensor["shape"]) != (1, 1)
                or output_tensor["quantization"] != (1 / 256, 0)):
            raise ValueError(f"Unexpected ESPHome tensor interface in {model.name}")
        scores = []
        for chunk in chunks:
            interpreter.set_tensor(input_tensor["index"], chunk)
            interpreter.invoke()
            scores.append(int(interpreter.get_tensor(output_tensor["index"])[0, 0]))
        traces.append((row, duration, times, np.asarray(scores, dtype=np.int32)))
    return traces


def summarize(traces, window, cutoffs):
    hits = np.zeros(len(cutoffs), dtype=int)
    early = np.zeros_like(hits)
    false_alarms = np.zeros_like(hits)
    latency = []
    outcomes = []
    for row, _, times, scores in traces:
        sums = np.convolve(np.pad(scores, (window - 1, 0)), np.ones(window, dtype=int), mode="valid")
        passed = (sums[:, None] > cutoffs * window) & (times[:, None] >= 0)
        detected = passed.any(axis=0)
        first = times[passed.argmax(axis=0)]
        if row["label"]:
            end = row["phrase_offset_s"]
            start = max(row["phrase_onset_s"], end - 0.25)
            hit = detected & (first >= start) & (first <= end + 0.5)
            before = detected & (first < start)
            hits += hit
            early += before
            if len(cutoffs) == 1:
                if hit[0]:
                    latency.append((first[0] - end) * 1000)
                if hit[0]:
                    outcome = "hit"
                elif before[0]:
                    outcome = "early_trigger"
                else:
                    outcome = "late_trigger" if detected[0] else "miss"
        else:
            false_alarms += detected
            if len(cutoffs) == 1:
                outcome = "false_alarm" if detected[0] else "true_negative"
        if len(cutoffs) == 1:
            outcomes.append({"id": row["id"], "category": row["category"], "outcome": outcome,
                             "first_trigger_s": float(first[0]) if detected[0] else None})
    return hits, early, false_alarms, latency, outcomes


def report(traces, window, cutoff):
    hits, early, false_alarms, latency, outcomes = summarize(traces, window, np.array([cutoff]))
    positives = sum(row["label"] for row, *_ in traces)
    return {
        "positive_clips": positives,
        "negative_clips": len(traces) - positives,
        "hits": int(hits[0]),
        "early_positive_triggers": int(early[0]),
        "false_alarm_clips": int(false_alarms[0]),
        "negative_hours": sum(duration for row, duration, *_ in traces if not row["label"]) / 3600,
        "latency_median_ms": float(np.median(latency)) if latency else None,
        "latency_p95_ms": float(np.percentile(latency, 95)) if latency else None,
        "outcomes": dict(Counter(outcome["outcome"] for outcome in outcomes)),
    }, outcomes


def evaluate(root, rows, out):
    models = sorted((out / "candidates").glob("*.tflite"))
    if not models:
        raise FileNotFoundError("Export at least one TFLite checkpoint before evaluation")
    validation = labeled_features(root, rows, "validation")
    budget = sum(not row["label"] for row, *_ in validation) // 100
    cutoffs = np.arange(1, 255)
    best = None
    candidates = []
    for model in models:
        traces = stream_scores(model, validation)
        model_best = None
        for window in range(1, 9):
            hits, early, alarms, _, _ = summarize(traces, window, cutoffs)
            for i in np.flatnonzero(alarms <= budget):
                key = (int(hits[i]), -int(alarms[i]), -int(early[i]), -window, int(cutoffs[i]))
                if model_best is None or key > model_best[0]:
                    model_best = key, model, window, int(cutoffs[i])
        if model_best:
            print(f"Validation {model.name}: {model_best[0][0]} hits, {-model_best[0][1]} false alarm clips", flush=True)
            candidates.append({"checkpoint": model.stem, "hits": model_best[0][0],
                               "false_alarm_clips": -model_best[0][1], "early_triggers": -model_best[0][2],
                               "sliding_window_size": model_best[2], "uint8_cutoff": model_best[3]})
            if best is None or model_best[0] > best[0]:
                best = model_best
        else:
            print(f"Validation {model.name}: no cutoff met the false alarm budget", flush=True)
    if best is None:
        raise ValueError("No candidate met the false alarm budget")
    _, model, window, cutoff = best
    valid_report, _ = report(stream_scores(model, validation), window, cutoff)
    test = labeled_features(root, rows, "test")
    test_report, outcomes = report(stream_scores(model, test), window, cutoff)
    selected = out / "goodwin-microwakeword.tflite"
    shutil.copyfile(model, selected)
    # ESPHome converts the JSON threshold to uint8 by truncating p * 255.
    probability_cutoff = (cutoff + 0.5) / 255
    result = {
        "source_checkpoint": model.stem,
        "sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
        "model_bytes": selected.stat().st_size,
        "probability_cutoff": probability_cutoff,
        "uint8_cutoff": cutoff,
        "sliding_window_size": window,
        "validation_false_clip_budget": budget,
        "validation_candidates": candidates,
        "validation": valid_report,
        "test": test_report,
    }
    (out / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (out / "test_predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in outcomes), encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Prepare and evaluate microWakeWord models")
    parser.add_argument("action", choices=["prepare", "train", "export", "evaluate"])
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", default="best_weights")
    args = parser.parse_args()
    if args.action in {"prepare", "evaluate"} and args.dataset is None:
        parser.error(f"{args.action} requires --dataset")
    if args.action == "prepare":
        if args.steps < 4:
            parser.error("prepare requires at least four steps")
        prepare(args.dataset, load_rows(args.dataset), args.out, args.steps)
    elif args.action == "train":
        train(args.out, args.threads, args.resume)
    elif args.action == "export":
        export(args.out, args.threads, args.checkpoint)
    else:
        evaluate(args.dataset, load_rows(args.dataset), args.out)


if __name__ == "__main__":
    main()
