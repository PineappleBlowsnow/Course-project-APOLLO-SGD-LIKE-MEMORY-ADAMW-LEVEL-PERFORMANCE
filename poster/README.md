# Poster

The poster is already structured around your final narrative:

1. The Dilemma
2. AdamW-level Performance
3. SGD-like Memory & Speed
4. Sanity Checks
5. Ablation & Failure Modes

After running:

```bash
python scripts/make_plots.py --results-root results
python scripts/build_poster_assets.py --results-root results
```

Open `poster/poster.html`. If images exist in `poster/assets/generated/`, they will appear automatically.

