# Colab Run Sheet

## 1. Install

```bash
!git clone <your-repo-url>
%cd project
!pip install -e .
```

## 2. Main training suite

```bash
!python scripts/run_suite.py --suite configs/suites/full_pretrain.yaml --output-root results
```

## 3. System microbench

```bash
!python scripts/run_suite.py --suite configs/suites/microbench.yaml --output-root results
```

## 4. Rank / failure ablations

```bash
!python scripts/run_suite.py --suite configs/suites/ablation.yaml --output-root results
```

## 5. Sharpness analysis

```bash
!python scripts/analyze_sharpness.py --suite-dir results/full_pretrain
```

## 6. Make all figures

```bash
!python scripts/make_plots.py --results-root results
!python scripts/build_poster_assets.py --results-root results
```

## 7. Open poster

Open `poster/poster.html` and export to PDF after you fill in the final numbers.

