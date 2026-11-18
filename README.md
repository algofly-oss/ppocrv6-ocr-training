# PP-OCRv6 recognition experiments

This repository documents PP-OCRv6 recognition experiments on cropped license plates and retail price tags. The model reads a supplied crop. Detection and crop generation are outside the benchmark.

| Task and checkpoint | Parameters | Input H×W | Development exact match | RGB latency | Encoded latency | Peak GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| Price tags, mixed precision | 15.58M | 128×320 | 104/144 (72.22%) | 0.884 ms | 1.087 ms | 654 MiB |
| ALPR, height 128 | 15.58M | 128×320 | 1,027/1,167 (88.00%) | 0.964 ms | 1.094 ms | 654 MiB |

The RTX 5090 measurements use batch size one and include preprocessing, TensorRT inference, and text decoding. Encoded latency also includes decoding image bytes already in memory. Loading and warmup are excluded from latency. Peak memory is driver-accounted for the inference process.

Exact match compares the full string, including decimal punctuation and currency text. Mixed precision preserved the measured price-tag exact-match total, but failed a stricter numerical comparison. A separate FP32 TensorRT run scored 104/144 at 1.413 ms with 588 MiB peak GPU memory.

On 984 reviewed wild plate crops, optimized recognition matched 534 with the height-128 configuration and 626 with the height-48 configuration. The higher development score at height 128 did not carry over to that cohort. The price-tag mixed-precision model matched 163/206 on the broader historical set, which includes 68 provisional references.

Training and deployment notebooks require a portable export of the research workflow. This initial repository contains the benchmark summary. Private datasets, trained checkpoints, and GPU engines are excluded.
