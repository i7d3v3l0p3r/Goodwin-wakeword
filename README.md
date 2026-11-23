# Goodwin Wakeword

Russian wake-word models and tools for [Goodwin](https://github.com/i7d3v3l0p3r/Goodwin). The models detect a wake phrase locally from streaming audio.

| Wake word | Standard ONNX | Micro ONNX | Micro INT8 TFLite |
| --- | --- | --- | --- |
| Гудвин | [goodwin-standard.onnx](models/goodwin-standard.onnx) | [goodwin-micro.onnx](models/goodwin-micro.onnx) | [goodwin-micro-int8.tflite](models/goodwin-micro-int8.tflite) |
| Егор | [egor-standard.onnx](models/egor-standard.onnx) | [egor-micro.onnx](models/egor-micro.onnx) | |

The «Гудвин» ONNX files are the default models bundled by Goodwin. The «Егор» models come from the expanded corpus run. See the [model notes](models/README.md) for provenance, evaluation, and attribution.

To use an ONNX model, run `goodwin wake-check <model.onnx>`, then place it at `goodwin-data/goodwin.onnx` and restart Goodwin. The app detects the ONNX input layout automatically. The TFLite export uses LiveKit's Micro runtime with LiteRT.

- [Generate a dataset](docs/datasets.md) using a Russian phrase profile, OpenRouter TTS, and local alignment.
- [Train and evaluate](docs/training.md) a Standard model or distill a Micro model.

## Model releases

Pushing a tag in the form `vMAJOR.MINOR.PATCH` triggers the [release workflow](.github/workflows/release.yml). It publishes the models in [the catalog](models/catalog.json), the model license and notice, `metadata.json`, and `SHA256SUMS`. The workflow publishes the files from the tagged commit.

Preview the release files locally with Python 3.12. Use a new output directory for each preview.

```sh
python scripts/package_release.py --tag v0.1.0 --out dist/preview
```

Create and push a tag when the model and catalog updates are ready.

```sh
git tag -a v0.1.0 -m "Release wake-word models v0.1.0"
git push origin v0.1.0
```

### Release metadata

`metadata.json` is UTF-8 JSON with a numeric `schema_version`, the release `tag` (including `v`), and a `models` array. Each model has these fields.

| Field | Meaning |
| --- | --- |
| `id` | Stable identifier across releases |
| `filename` | Exact GitHub Release asset name |
| `wake_word` | Phrase as displayed to users |
| `language` | BCP 47 language tag, currently `ru` |
| `runtime` | `livekit-v2-standard-onnx`, `livekit-v2-micro-onnx`, or `livekit-v2-micro-tflite` |
| `tier`, `format` | Model size tier and file format |
| `quantization` | Numeric representation for quantized variants, when present |
| `sample_rate_hz` | Audio sample rate before runtime preprocessing |
| `score_threshold` | Score threshold selected on validation, independent of the Goodwin sensitivity setting |
| `goodwin_compatible` | Whether the Goodwin app can load this file directly |
| `size_bytes`, `sha256` | Download size and lowercase SHA-256 of the published file |

Use `filename` to find the matching downloadable asset in the same GitHub Release. The code is MIT licensed. The LiveKit encoder weights and derived models carry the [Apache 2.0 license](models/LICENSE) and [notice](models/NOTICE).
