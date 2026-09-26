# Results, contribution, and evidence

This MVA project studies the memory/convergence trade-off of APOLLO variants against AdamW in small language models. Ying Jin carried out the implementation and experiments. The [submitted report](../report/Apollo_report.pdf) credits Ying Jin and Felipe Vicentin; the historical report and its author list are preserved unchanged. Method attribution belongs to the original APOLLO authors.

## LLaMA-style 60M on TinyStories

The following rows come from the archived 10,000-step pretraining runs. Optimizer-state measurements are taken from the recorded step-10,000 checkpoints. Perplexity is the **best recorded validation perplexity** for each run, not a test-set result or a multi-seed average.

| Variant | Optimizer state (bytes) | Optimizer state (GiB) | Best validation perplexity |
|---|---:|---:|---:|
| AdamW | 464,588,800 | 0.432682 | 3.6072 |
| APOLLO rank 1/4 | 116,199,424 | 0.108219 | 4.1306 |
| APOLLO rank 1/8 | 58,134,528 | 0.054142 | 4.1324 |

For rank 1/8, `1 - 58,134,528 / 464,588,800` is approximately **87.5% less optimizer-state storage**. Perplexity is higher than AdamW in the same recorded comparison, so the result should be presented as a trade-off rather than matching AdamW quality.

The original exports are available as [optimizer_memory.csv](evidence/optimizer_memory.csv) and [table2_like.csv](evidence/table2_like.csv). Their bytes are preserved; [provenance.json](evidence/provenance.json) records SHA-256 hashes and source artifact names.

## Measurement boundaries

- The legacy CSV column `optimizer_state_gb` uses bytes divided by `1024**3`, so its unit is **GiB**, despite the original column name.
- Optimizer state is one component of training memory. Parameters, gradients, activations, temporary buffers and allocator behavior are additional costs. The approximately 87.5% figure is **not** a claim about total GPU memory, throughput, or training cost.
- The exported `peak_memory_gb` values are retained for traceability but are not used here as a comparable total-memory benchmark. They do not justify a matched peak-memory claim across these rows.
- The CSV also contains APOLLO-mini rows, including a step-2,500 checkpoint. Those rows are not included in the matched step-10,000 comparison above. No quality equivalence is inferred from their smaller optimizer state.
- No uncertainty estimate or multi-seed result is supplied by these two exports. They do not support extrapolation to larger models or other datasets.

## Code and reproducibility

The [main README](../README.md) contains setup instructions and commands for training, checkpoint resume, fixed-batch memory diagnostics, sharpness analysis, and plotting. Relevant source entry points are [optimizers.py](../src/apollo_story/optimizers.py), [train.py](../src/apollo_story/train.py), [benchmark.py](../src/apollo_story/benchmark.py), and [plotting.py](../src/apollo_story/plotting.py). The repository includes a [small optimizer smoke test](../scripts/smoke_test.py).

The repository also contains nanoGPT experiments and ARC finetuning code. The selected numerical comparison above is restricted to the documented 60M TinyStories exports. ARC finetuning was not conducted and is not an experimental result.

On 26 September 2026, this documentation update checked the source paths, CSV schemas and hashes, checkpoint labels, unit conversion, and arithmetic. Training, GPU benchmarks, and the smoke test were **not rerun**. The CSV files are compact historical result exports; they are not a replacement for checkpoints or complete training logs.
