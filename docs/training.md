# Train LiveKit WakeWord models

The local workflow uses the pretrained LiveKit v2.1.3 encoders and runs on CPU. It samples augmentation audio from training, selects the checkpoint and operating threshold on validation, and scores held-out audio after calibration.

## Set up the trainer

Use Python 3.12 in the virtual environment from [dataset setup](datasets.md), or create a separate one. Run from the repository root. Download the [trainer source](https://github.com/livekit/livekit-wakeword/archive/refs/tags/v2.1.3.zip) and the [encoder archive](https://github.com/livekit/livekit-wakeword/releases/download/v2.0.0/encoders.tar.gz) as `target/source.zip` and `target/encoders.tar.gz`.

```powershell
$hash = (Get-FileHash target/encoders.tar.gz -Algorithm SHA256).Hash.ToLowerInvariant()
if ($hash -ne "a13a83c34752f6e1c20c47f1b048c02217f6d147ee721ea101eb58e59d4922e7") {
    throw "Unexpected LiveKit encoder archive SHA-256"
}
Expand-Archive target/source.zip -DestinationPath target
tar -xzf target/encoders.tar.gz -C target
git apply --directory=target/livekit-wakeword-2.1.3 scripts/livekit-wakeword-cpu.patch
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0+cpu torchaudio==2.10.0+cpu
python -m pip install -r requirements/train.txt
python -m pip install --no-deps ./target/livekit-wakeword-2.1.3
```

The patch enables single-process CPU loading, deterministic sampling, and per-step checkpoint evaluation. See [LiveKit's training guide](https://github.com/livekit/livekit-wakeword/blob/v2.1.3/client/python/TRAINING.md) for the upstream trainer.

## Standard and Micro ONNX

```powershell
$encoders = "target/encoders"
$corpus = "datasets/goodwin"
$standard = "runs/goodwin-standard"
$micro = "runs/goodwin-micro"

python scripts/wakeword_train_livekit.py --dataset $corpus --encoders $encoders --out $standard
python scripts/wakeword_eval_livekit.py --dataset $corpus --encoders $encoders --out $standard
python scripts/wakeword_plot_livekit.py --run $standard

python scripts/wakeword_train_livekit.py --dataset $corpus --encoders $encoders --out $micro --tier micro --teacher "$standard/selected.pt" --steps 600 --checkpoint-every 50 --threads 4 --party-probability 0.15 --distillation-weight 0.75 --near-miss-count 12
python scripts/wakeword_eval_livekit.py --dataset $corpus --encoders $encoders --out $micro --tier micro
python scripts/wakeword_plot_livekit.py --run $micro
```

The calibrated ONNX files are `$standard/goodwin.onnx` and `$micro/goodwin_micro.onnx`. The evaluator maps the selected operating point to 0.50, using a 1% false-alarm budget on negative validation clips. It writes `calibration.json`, `evaluation.json`, score traces, and per-clip predictions. Use `--training-only` with the plot script to inspect the training curves before evaluating.

For a different wake phrase, set `--phrase` in training and use `--name` in evaluation to choose the export filename. Use separate run directories for different variants. A dedicated test corpus can be passed with `--test-dataset`.

### Expanded «Егор» settings

The expanded run uses the original corpus for the held-out test and an expanded corpus for training and validation. The extra training supervision extends the positive label by 120 ms and oversamples the closest negative utterances. Evaluation uses the original annotations.

```powershell
$corpus = "datasets/egor/expanded"
$test = "datasets/egor"
$standard = "runs/egor-standard"
$micro = "runs/egor-micro"
$boundary = @("--label-delay-ms", "120", "--close-negative-pattern",
    '\b(?:егор[аувкш][а-я]*|его р[а-я]+|с гор)\b', "--extra-close-negative-copies", "5",
    "--party-probability", "0.15", "--near-miss-count", "12")

python scripts/wakeword_train_livekit.py --dataset $corpus --phrase "Егор" --encoders $encoders --out $standard --steps 800 --threads 6 @boundary
python scripts/wakeword_eval_livekit.py --dataset $corpus --test-dataset $test --encoders $encoders --out $standard --name egor
python scripts/wakeword_train_livekit.py --dataset $corpus --phrase "Егор" --encoders $encoders --out $micro --tier micro --teacher "$standard/selected.pt" --steps 600 --threads 4 --checkpoint-every 50 --distillation-weight 0.75 @boundary
python scripts/wakeword_eval_livekit.py --dataset $corpus --test-dataset $test --encoders $encoders --out $micro --tier micro --name egor
```

## Optional Micro INT8 TFLite export

After evaluating the Micro ONNX, use the same run directory, corpus, and export name to convert to TFLite. Quantization ranges use training audio. The INT8 threshold is selected separately on validation.

```powershell
python -m pip install -r requirements/quantize.txt
python scripts/wakeword_quantize_livekit.py --dataset datasets/goodwin --encoders $encoders --out runs/goodwin-micro --name goodwin
```

The example writes `goodwin_micro.tflite`, `tflite_calibration.json`, and `tflite_evaluation.json` into the run directory.
