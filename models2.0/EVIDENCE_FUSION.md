# Evidence Fusion

Evidence Fusion combines the two existing models — **Logistic Regression** (LogReg)
and **Semantic Evidence Voting** (SEV) — at the class-score level.

## Method

Both models output a score vector over the same three labels (LogReg
probabilities, SEV vote proportions). Each is clipped to non-negative and
renormalised to sum to 1, then blended with a single weight `w` (the LogReg
share):

    fused[c] = w * logreg[c] + (1 - w) * sev[c]      (renormalised)

`w` is tuned on a **validation** split only (grid 0.0–1.0), never on the test
set. The predicted label is the argmax of the fused scores.

### Human-review signals

The fused result carries flags that only ever *raise* attention (review is always
required regardless):

| Signal | Condition |
| --- | --- |
| `models_agree` | LogReg and SEV pick the same label |
| `low_confidence` | top fused score `< low_confidence` threshold (default 0.50) |
| `low_margin` | gap between top two fused scores `< low_margin` (default 0.15) |
| `review_flag` | disagreement **or** low confidence **or** low margin |

`fusion_config.json` holds the validation-tuned `weight` and the thresholds; the
app loads it at runtime.

## Files

- `evidence_fusion.py` — `normalize_scores`, `fuse_scores`, `fuse`, `tune_weight`.
- `evaluate_fusion.py` — 3-way grouped split, tunes `w` on validation, compares all three on test.
- `fusion_config.json` — tuned weight + thresholds for the app.
- `evaluation_fusion.json`, `confusion_fusion_heldout.csv` — saved metrics.

## Honest evaluation (same untouched test set)

Grouped train/val/test split by `scenario_family` (seed 42; test 20%, val 20%,
train 60% + augmentation). SEV's retrieval pool is the **train** split only
(test/val never in the pool). Fusion weight tuned on **validation** only →
`w = 0.7` (70% LogReg).

| Metric (52 test rows) | LogReg | SEV | Evidence Fusion |
| --- | --- | --- | --- |
| accuracy | 0.769 | 0.615 | 0.769 |
| macro F1 | 0.772 | 0.615 | **0.776** |
| PRIORITY_REVIEW recall | 1.000 | 1.000 | 1.000 |

**Honest conclusion:** Fusion has the highest macro F1, but the margin over
LogReg is **+0.004 (one UNCLEAR example reclassified)** — not a meaningful
improvement on 52 rows. Fusion is best understood as **≈ tied with LogReg** and
clearly better than SEV alone, with the added value of explicit
disagreement/low-confidence review flags. The tuned weight (0.7 toward LogReg)
reflects LogReg being the stronger component on this romanized-Hinglish corpus.
No method is claimed as a validated improvement.

> Note: these absolute numbers are lower than the 2-model report
> (`evaluation_semantic_vs_logreg.json`, LogReg macro F1 0.939) because this
> comparison uses a smaller 60% train split (to carve out a validation set) and a
> different test fold. All three models here are evaluated on the *same* fold, so
> the comparison between them is fair.

## Reproduce

```bash
python models2.0/evaluate_fusion.py --save   # 3-way split, tune weight, compare
```

## Safety

Decision support only. Fused scores and review flags are not validated risk
scores and must never auto-dispatch. Every result requires human verification.
