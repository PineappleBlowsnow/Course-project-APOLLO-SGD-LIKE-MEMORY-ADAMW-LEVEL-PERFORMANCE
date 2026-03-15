# APOLLO Poster Report Skeleton

## 1. Claim

This project verifies the following claims on a GPT-2 124M + WikiText-2 setup:

1. APOLLO keeps training quality close to AdamW.
2. APOLLO reduces optimizer-state memory enough to unlock larger batch sizes and higher throughput.
3. APOLLO-Mini remains stable in the extreme `r = 1` regime where GaLore or channel-wise APOLLO may fail.

## 2. Protocol

- Model: GPT-2 124M
- Dataset: WikiText-2
- Steps: 3000
- Main optimizers: SGD, AdamW, GaLore rank 1/4, GaLore rank 1/8, APOLLO rank 1/4, APOLLO rank 1/8, APOLLO-Mini
- Benchmarks: memory breakdown, max batch size, throughput, optimizer step time

## 3. Results

### 3.1 Convergence Race

Insert `convergence_race.png`

### 3.2 Scaling Factor Ratio

Insert `scaling_ratio.png`

### 3.3 Table-2-like comparison

Insert `table2_like.csv`

### 3.4 Pareto Frontier

Insert `pareto_frontier.png`

### 3.5 Memory Breakdown

Insert `memory_breakdown.png`

### 3.6 Max Batch Size and Throughput

Insert `batch_throughput.png`

### 3.7 Optimizer Step Spikes

Insert `optimizer_step_spikes.png`

### 3.8 Directional Sharpness

Insert `sharpness_curve.png`

### 3.9 Rank / Granularity Ablation

Insert `rank_ablation.png` and `granularity_r1.png`

## 4. Discussion

- Why APOLLO reaches a better memory-quality frontier
- Why GaLore shows periodic spikes
- Why `r = 1` needs tensor-wise scaling
- Why APOLLO may act like an implicit noise filter

## 5. Limitations

- WikiText-2 is much smaller than the original paper's scale.
- The GaLore implementation is a small-scale reproduction, not a full industrial training stack.
- Pressure testing is implemented via logical memory budgets in benchmarking.

