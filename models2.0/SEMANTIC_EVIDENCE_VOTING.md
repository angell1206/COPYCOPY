# Semantic Evidence Voting (SEV)

**"SEV" = Semantic Evidence Voting** — the name we propose for this method. We do
not assume it stands for anything else (e.g. "JEV" is not used here). It is an
experimental alternative to the Logistic Regression baseline, shipped for
transparency and comparison — **not** as a claimed improvement.

## Method

1. Embed every labelled **training** transcript with
   `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (L2-normalised).
2. For a new transcript, embed it with the same encoder and compute cosine
   similarity to every training embedding.
3. Take the top-k most similar training examples (the "evidence").
4. Cast a similarity-weighted vote for each example's `label` (negative
   similarities clipped to 0).
5. Normalise the per-class vote mass into **vote proportions** and predict the
   class with the largest proportion.

Input = `transcript` text; target = `label`. Voting is over **training examples
only** — never the query itself or any held-out/test example — so there is no
data leakage. The nearest neighbours are shown in the UI as human-readable
supporting evidence, and the vote proportions are the normalised share of
similarity mass per class — **not** calibrated probabilities.

Files: `semantic_evidence_voting.py` (model), `build_sev_index.py` (builds
`sev_index.npz` for the app), `evaluate_semantic.py` (honest comparison).

## Honest evaluation (same untouched test set)

Both models trained on the same training portion and evaluated on the same
leakage-free grouped held-out split (GroupShuffleSplit by `scenario_family`,
test_size=0.2, seed=42; the `calls_fresh_eval_36.csv` named in the dataset
guidelines is absent, so this split is the untouched test set). SEV's k was
selected by leave-one-out macro F1 on the **training** set only (k=5).

| Metric (52 held-out rows) | Logistic Regression | Semantic Evidence Voting |
| --- | --- | --- |
| macro F1 | **0.939** | 0.748 |
| accuracy | **0.942** | 0.750 |
| PRIORITY_REVIEW recall | **1.000** | 0.917 |

Per-class and confusion matrices are in `evaluation_semantic_vs_logreg.json`,
`confusion_logreg_heldout.csv`, and `confusion_sev_heldout.csv`.

**Conclusion (not an improvement):** on this corpus the Logistic Regression
baseline is clearly stronger (macro F1 0.939 vs 0.748, Δ = −0.191 for SEV). The
dataset is heavily romanized Hinglish (Latin-script Hindi), which the multilingual
sentence encoder embeds weakly, while the char-level TF-IDF baseline captures those
token patterns well. Logistic Regression therefore remains the **default** model;
SEV is available in the UI as an experimental, transparent alternative.

## Reproduce

```bash
python models2.0/evaluate_semantic.py --save   # honest LogReg vs SEV comparison
python models2.0/build_sev_index.py            # rebuild the app's SEV index
```

## Safety

Decision support only. Vote proportions and nearest-neighbour similarity are not
proof of a shared caller or incident and are not validated risk scores. Every
result requires human verification; the system must never auto-dispatch services.
