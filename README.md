# ECE 313 Project 1 — GPU XID Error Analysis on NVIDIA A100 Clusters

**Team:** Shuban Iyer, Yogi Patel, Swarit Tumuluri

## Overview

Statistical analysis of real-world XID error logs collected from NVIDIA A100 GPU clusters (the Delta supercomputer). The project applies probability theory and statistical inference to characterize GPU failure behavior, detect anomalies, and predict GPU failures from error telemetry.

## Dataset

- `delta_a100_gpu_xid_dataset.pkl` — raw XID error logs (timestamped per node/device)
- `delta_a100_gpu_xid_dataset_with_noise.pkl` — dataset with duplicate/noisy events
- `delta_a100_gpu_xid_dataset_with_GPU_failure_label.pkl` — dataset with binary GPU failure labels

## Project Structure

### Task 1 — Exploratory Data Analysis
- Parsed and structured raw XID logs into a Pandas DataFrame
- Computed empirical probabilities of specific XID error types (e.g., XID 43, XID 119)
- Plotted XID frequency distributions on linear and logarithmic scales
- Analyzed inter-arrival times: computed mean, standard deviation, and distribution fit

### Task 2 — Reliability Modeling
- Validated exponential distribution fit for inter-arrival times using Kolmogorov-Smirnov goodness-of-fit test
- Linked exponential inter-arrivals to a Poisson arrival process; estimated rate parameter λ
- Computed empirical Mean Time Between Failures (MTBF) and compared to distributional expectation
- Built the Nelson-Aalen cumulative hazard estimator and compared to theoretical exponential hazard rate

### Task 3 — Noise Filtering and Error Propagation
- Implemented a deduplication algorithm removing burst duplicates within a 5-second window per (node, device, XID) tuple
- Analyzed sensitivity of duplicate count to window size (1–20 seconds)
- Detected coalesced XID 13–43 event pairs indicating correlated failures
- Clustered events with a 600-second gap threshold; computed cluster size and duration statistics
- Constructed a system-level and device-level XID propagation matrix to model conditional transition probabilities between error types

### Task 4 — Failure Detection and LLM Integration
- **Part 1:** Binary hypothesis testing (H₀: no GPU failure, H₁: GPU failure) using Maximum Likelihood Estimation over XID distributions; evaluated detection threshold and performance metrics
- **Part 2:** Integrated Qwen 2.5-7B LLM (4-bit quantized via HuggingFace) with RAG-style context injection to generate XID descriptions and failure explanations; explored temperature effects and hallucination behavior

## Requirements

```
python >= 3.9
pandas
numpy
matplotlib
scipy
pickle
transformers
torch
llama_index==0.10.19
auto_gptq
bitsandbytes
```
