# Generate a wake-word dataset

Run the commands from the repository root with Python 3.12. The examples use PowerShell.

```powershell
python -m venv .venv
. ./.venv/Scripts/Activate.ps1
python -m pip install -r requirements/dataset.txt
```

Use `source .venv/bin/activate` on Linux. Set `OPENROUTER_API_KEY` in the environment or copy `.env.example` to `.env` and fill in the value before running catalog or synthesis commands.

## Choose a phrase

Use [goodwin.json](../profiles/goodwin.json) or [egor.json](../profiles/egor.json). To add a wake word, copy one of these profiles and edit the fields.

- `phrase` is the display and training text. `aliases` are lowercase spellings that the local transcriber can return for that phrase.
- `probe` contains the phrase in a short request. `probe_words` are the other words that verify intelligible Russian speech.
- `positive` contains the isolated phrase and requests with the wake word in different positions.
- `negative` contains `["adversarial", "…"]` near misses or `["speech", "…"]` ordinary speech.
- `extra_positive` and `extra_negative` give `fill` more prompts. `recovery_spellings` lists common ASR errors to revisit during alignment recovery.

Use a separate dataset directory per phrase. The optional [voice overrides](../profiles/voice-overrides.example.json) add provider voice IDs omitted from the OpenRouter catalog. Copy the example to the dataset directory as `voice-overrides.json` and edit it for the models you want to probe.

## Synthesize and align clips

```powershell
$root = "datasets/goodwin"
$profile = "profiles/goodwin.json"
python scripts/wakeword_dataset.py catalog --root $root
Copy-Item profiles/voice-overrides.example.json "$root/voice-overrides.json"
python scripts/wakeword_dataset.py inventory --root $root --profile $profile
python scripts/wakeword_dataset.py probe --root $root --profile $profile --voices-per-model 1
python scripts/wakeword_dataset.py generate --root $root --profile $profile --voices-per-model 1 --rounds 6
```

The probe and generation commands send paid TTS requests. The example probes one voice per model. Use `--model <model-id>` to narrow the models, and raise or omit `--voices-per-model` to broaden coverage. The probe and generation filters should match. Set a spending limit on the API key and adjust `--reserve` and `--workers` for the run. The local journal resumes completed work. Use `--retry` to revisit transient failures. The initial probe loads the local Whisper model, downloading its weights when absent.

Run `fill` to add prompts for underrepresented classes, `recover` to realign review clips, or `derive` to extract aligned excerpts from existing audio. Pass the same `--root` and `--profile` for each command.

## Add background audio and export

Download the [MUSAN music shard](https://huggingface.co/datasets/corypaik/musan/resolve/refs%2Fconvert%2Fparquet/music/train/0008.parquet) and [FLEURS Russian speech shard](https://huggingface.co/datasets/google/fleurs/resolve/refs%2Fconvert%2Fparquet/ru_ru/validation/0000.parquet) to `target/music.parquet` and `target/speech.parquet`. The background script fetches MUSAN environmental noise. It records source information and excludes speech whose transcript contains an alias in the profile. For other local Parquet shards, pass their source URLs using `--music-source` and `--speech-source`.

```powershell
python scripts/wakeword_background.py --root $root --profile $profile --music-parquet target/music.parquet --speech-parquet target/speech.parquet
python scripts/wakeword_dataset.py report --root $root --profile $profile
python scripts/wakeword_export.py --root $root
```

The trainer reads `manifest.jsonl` and its referenced audio. The last command optionally writes deduplicated WakeWord Bench Parquet shards under `data/`. Review `summary.json`, `coverage.csv`, and any clips under `.review/` before training.

## Manifest format

Each JSONL line identifies a 16 kHz mono PCM16 WAV relative to the manifest directory. A training corpus needs `positive`, `adversarial`, `speech`, `noise`, and `real_speech` examples in training, plus both labels in validation and test. Keep each speaker and each original recording in one split. Keep held-out audio separate before inspecting test results.

The following rows show the shape of an annotated positive and a near miss. Fill in actual paths, durations, and boundaries for your recordings.

```jsonl
{"id":"example-positive","path":"positive/example.wav","group":"speaker-01","category":"positive","label":1,"split":"train","duration_s":1.8,"phrase_onset_s":0.2,"phrase_offset_s":0.7,"text":"Гудвин, поставь таймер","source":"local"}
{"id":"example-negative","path":"negative/example.wav","group":"speaker-02","category":"adversarial","label":0,"split":"validation","duration_s":1.5,"text":"Гудман поставил таймер","source":"local"}
```

An existing corpus can be copied under `datasets/` and used directly. Keep its manifest and all WAV paths together. For an expanded corpus whose manifest contains `../` paths, preserve the parent directory structure as well.
