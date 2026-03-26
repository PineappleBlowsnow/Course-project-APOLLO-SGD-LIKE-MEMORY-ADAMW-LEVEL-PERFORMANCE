# APOLLO Paper Reproduction

This repository contains the training, benchmarking, plotting, and analysis code used to evaluate the claim that Approximated Gradient Scaling for Memory Efficient LLM Optimization (APOLLO) achieves AdamW-level performance with lower memory usage. The original paper is: https://arxiv.org/pdf/2412.05270. 

The current default setup is centered on small LLaMA-style models:

- pretraining: `LLaMA-60M` on `TinyStories`; `nanoGPT 40M` on `TinyStories`; `nanoGPT 130M` on `WikiText`. 
- optimizer diagnostics: scaling traces, memory profiling, step-time profiling, and directional sharpness
- finetuning: `LLaMA-60M` on `ARC-Easy` / `ARC-Challenge` (We didn't conduct finetuning experiments)

Legacy GPT-style configs are still present under `configs/base.yaml` and some older suites, but the main maintained path is the LLaMA-60M workflow.

## Install

```bash
pip install -e .
```

Quick optimizer smoke test:

```bash
python scripts/smoke_test.py
```

## Main Scripts

- `scripts/run_suite.py`: run a YAML suite of training, benchmark, or finetune experiments
- `scripts/train_experiment.py`: run one training experiment from a base config
- `scripts/analyze_sharpness.py`: post-hoc directional sharpness analysis from saved analysis checkpoints
- `scripts/make_table10.py`: merge `sharpness_analysis.jsonl` files into a Table-10-like CSV
- `scripts/make_plots.py`: generate all figures/tables for a results root
- `scripts/build_poster_assets.py`: copy generated figures into `poster/assets/generated`

## Recommended LLaMA-60M Workflows

### Table-2-like pretraining

```bash
python scripts/run_suite.py --suite configs/suites/table2_llama60_pretrain.yaml --output-root results/table2_llama60
python scripts/make_plots.py --results-root results/table2_llama60
```

This suite runs:

- `adamw`
- `apollo_svd`
- `apollo_rank_1_4`
- `apollo_rank_1_8`
- `apollo_mini`

and produces convergence curves, validation curves, scaling-ratio plots, Pareto plots, and `table2_like.csv`.

###  Memory analysis and OOM test

Default microbenchmark:

```bash
python scripts/run_suite.py --suite configs/suites/table2_llama60_microbench.yaml --output-root results/table2_llama60_microbench
python scripts/make_plots.py --results-root results/table2_llama60_microbench
```

Fair fixed-batch comparison:

```bash
python scripts/run_suite.py --suite configs/suites/table7_llama60_fixed_batch_fair.yaml --output-root results/fair_llama60_fixed_batch
python scripts/make_plots.py --results-root results/fair_llama60_fixed_batch
```

Shared-memory-budget comparison:

```bash
python scripts/run_suite.py --suite configs/suites/table7_llama60_shared_budget.yaml --output-root results/fair_llama60_shared_budget
python scripts/make_plots.py --results-root results/fair_llama60_shared_budget
```

These suites generate benchmark summaries, memory breakdowns, throughput plots, step-time breakdowns, and `table7_like.csv`.

### Sharpness / Table 10

Run the sharpness-enabled suite first:

```bash
python scripts/run_suite.py --suite configs/suites/table10_llama60.yaml --output-root results/table10_llama60
```

Then analyze checkpoints and build the merged table:

```bash
python scripts/analyze_sharpness.py --suite-dir results/table10_llama60/full_pretrain
python scripts/make_table10.py --suite-dir results/table10_llama60/full_pretrain
```

If you store outputs under a named root such as `results_table2_llama60/full_pretrain`, pass that full suite directory:

```bash
python scripts/analyze_sharpness.py --suite-dir results_table2_llama60/full_pretrain
python scripts/make_table10.py --suite-dir results_table2_llama60/full_pretrain
```

### Finetuning

```bash
python scripts/run_suite.py --suite configs/suites/finetune_llama60_arc.yaml --output-root results/finetune_llama60_arc
python scripts/make_plots.py --results-root results/finetune_llama60_arc
```

This runs ARC-Easy / ARC-Challenge finetuning for `adamw` and `apollo_mini`.

## Resume Support

Resume an interrupted suite:

```bash
python scripts/run_suite.py --suite configs/suites/table2_llama60_pretrain.yaml --output-root results/table2_llama60 --resume latest
```

Resume a single training experiment:

```bash
python scripts/train_experiment.py --config configs/base_llama60_pretrain.yaml --experiment-name adamw --output-root results/manual --resume latest
```

or from an explicit checkpoint:

```bash
python scripts/train_experiment.py --config configs/base_llama60_pretrain.yaml --experiment-name adamw --output-root results/manual --resume results/manual/adamw/checkpoint_step_5000.pt
```

## Results Layout

`scripts/make_plots.py --results-root <root>` expects a root directory that may contain:

- `<root>/full_pretrain`
- `<root>/microbench`
- `<root>/finetune`
- `<root>/ablation`
- `<root>/figures`

Important: pass the experiment root, not a nested subdirectory. For example:

- correct: `--results-root results/fair_llama60_fixed_batch`
- wrong: `--results-root results/fair_llama60_fixed_batch/microbench`

Generated outputs are written to:

- `<root>/figures/*.png`
- `<root>/figures/table2_like.csv`
- `<root>/figures/table7_like.csv`
- `<root>/figures/table10_like.csv`

