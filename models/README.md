# Model notes

The ONNX models use the LiveKit WakeWord v2 streaming format. Standard takes PCM features directly. Micro takes log-mel frames and recurrent state. Goodwin computes the Micro frontend from 16 kHz PCM. LiteRT runs the INT8 TFLite variant with the corresponding quantized Micro frontend and state.

## «Гудвин»

The ONNX files match the models bundled by Goodwin. The Standard model was fine-tuned locally and the Micro model was distilled from the selected Standard checkpoint. These streaming results use 613 held-out clips and thresholds selected on validation.

| Model | Score threshold | Hits / 290 | False alarm clips / 323 |
| --- | ---: | ---: | ---: |
| Standard ONNX | 0.50 | 266 | 6 |
| Micro ONNX | 0.50 | 191 | 0 |
| Micro INT8 TFLite | 0.390625 | 209 | 6 |

## «Егор»

These are the ONNX exports from the expanded training run with the original 797-clip held-out set. Both use a calibrated score threshold of 0.50.

| Model | Human hits / 11 | Synthetic hits / 130 | False alarm clips / 656 |
| --- | ---: | ---: | ---: |
| Standard ONNX | 5 | 83 | 3 |
| Micro ONNX | 6 | 61 | 3 |

The small human test set contains three isolated pronunciations and eight occurrences inside a sentence. The recorded results describe offline WAV evaluation. Detection rates on the physical Station may differ.

The [model catalog](catalog.json) lists the downloadable files and runtimes. The models incorporate LiveKit's pretrained encoders and are distributed with the accompanying [license](LICENSE) and [notice](NOTICE).
