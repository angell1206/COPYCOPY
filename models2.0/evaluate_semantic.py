"""Compare the Logistic Regression baseline against Semantic Evidence Voting (SEV).

Both models are trained on the SAME training portion and evaluated on the SAME
untouched, leakage-free held-out split (grouped by ``scenario_family`` so no
scenario family appears in both train and test). The curated augmentation set is
added to the train side only.

SEV's only hyper-parameter (k = number of voting neighbours) is selected by
leave-one-out macro F1 on the TRAINING portion only — never on the test split.

Reports for each model: accuracy, macro F1, per-class precision/recall/F1,
confusion matrix, and PRIORITY_REVIEW recall. Results are printed and (with
--save) written to evaluation_semantic_vs_logreg.json + confusion CSVs.

Usage:
    python models2.0/evaluate_semantic.py            # print comparison
    python models2.0/evaluate_semantic.py --save     # also write report files
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

from train_model import LABELS, TRAIN_CSV, build_pipeline  # local import
from semantic_evidence_voting import SemanticEvidenceVoter, embed

HERE = Path(__file__).resolve().parent
AUGMENTATION_CSV = HERE / "augmentation_v2_1.csv"
REPORT_JSON = HERE / "evaluation_semantic_vs_logreg.json"
RANDOM_STATE = 42
K_CANDIDATES = [5, 10, 15, 20, 25, 30, 40, 50]


def _augmentation() -> pd.DataFrame:
    if AUGMENTATION_CSV.exists():
        aug = pd.read_csv(AUGMENTATION_CSV)
        return aug[["transcript", "label"]].assign(transcript=lambda d: d["transcript"].astype(str))
    return pd.DataFrame(columns=["transcript", "label"])


def _metrics(y_true, y_pred) -> dict:
    report = classification_report(y_true, y_pred, labels=LABELS, digits=3, zero_division=0, output_dict=True)
    return {
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "priority_recall": float(recall_score(y_true, y_pred, labels=["PRIORITY_REVIEW"], average="macro", zero_division=0)),
        "per_class": {k: report[k] for k in LABELS},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def _print_block(name: str, m: dict, y_true, y_pred) -> None:
    print(f"\n================  {name}  ================")
    print(classification_report(y_true, y_pred, labels=LABELS, digits=3, zero_division=0))
    print(f"macro F1: {m['macro_f1']:.3f} | accuracy: {m['accuracy']:.3f} | PRIORITY recall: {m['priority_recall']:.3f}")
    print("confusion (rows=true, cols=pred):     " + "  ".join(f"{l.split('_')[0]:>9}" for l in LABELS))
    for lbl, row in zip(LABELS, m["confusion_matrix"]):
        print(f"  {lbl:>14}  " + "  ".join(f"{v:9d}" for v in row))


def _select_k_loo(train_texts, train_labels) -> tuple[int, dict[int, float]]:
    """Pick k by leave-one-out macro F1 on the training set only (no test leakage)."""
    probe = SemanticEvidenceVoter(k=max(K_CANDIDATES)).fit(train_texts, train_labels)
    emb, y = probe._emb, probe._y
    sims_all = emb @ emb.T
    np.fill_diagonal(sims_all, -np.inf)
    scores: dict[int, float] = {}
    for k in K_CANDIDATES:
        probe.k = k
        preds = [probe._vote_from_similarities(sims_all[i])["predicted_label"] for i in range(emb.shape[0])]
        scores[k] = float(f1_score(y, preds, average="macro"))
    best_k = max(scores, key=scores.get)
    return best_k, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    train = pd.read_csv(TRAIN_CSV)
    X = train["transcript"].astype(str).values
    y = train["label"].values
    groups = train["scenario_family"].astype(str).values
    aug = _augmentation()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(splitter.split(X, y, groups))
    assert not (set(groups[tr_idx]) & set(groups[te_idx])), "scenario family leakage!"

    train_texts = np.concatenate([X[tr_idx], aug["transcript"].values])
    train_labels = np.concatenate([y[tr_idx], aug["label"].values])
    X_test, y_test = X[te_idx], y[te_idx]
    print(f"train rows: {len(train_texts)} (incl. {len(aug)} augmentation) | untouched test rows: {len(X_test)}")

    # ---- Logistic Regression baseline (same training data) ------------------
    logreg = build_pipeline().fit(train_texts, train_labels)
    logreg_pred = logreg.predict(X_test)
    m_logreg = _metrics(y_test, logreg_pred)

    # ---- Semantic Evidence Voting -------------------------------------------
    best_k, k_scores = _select_k_loo(list(train_texts), list(train_labels))
    print(f"\nSEV k selected by leave-one-out macro F1 on TRAIN only: k={best_k}")
    print("  LOO macro F1 by k: " + ", ".join(f"{k}:{v:.3f}" for k, v in k_scores.items()))
    sev = SemanticEvidenceVoter(k=best_k).fit(list(train_texts), list(train_labels))
    sev_pred = sev.predict(list(X_test))
    m_sev = _metrics(y_test, sev_pred)

    _print_block("Logistic Regression (baseline)", m_logreg, y_test, logreg_pred)
    _print_block(f"Semantic Evidence Voting (k={best_k})", m_sev, y_test, sev_pred)

    print("\n================  SUMMARY (same untouched test set)  ================")
    print(f"{'metric':<22}{'LogReg':>10}{'SEV':>10}")
    for key, label in [("macro_f1", "macro F1"), ("accuracy", "accuracy"), ("priority_recall", "PRIORITY recall")]:
        print(f"{label:<22}{m_logreg[key]:>10.3f}{m_sev[key]:>10.3f}")
    delta = m_sev["macro_f1"] - m_logreg["macro_f1"]
    verdict = "SEV higher" if delta > 0 else ("LogReg higher" if delta < 0 else "tie")
    print(f"\nmacro F1 delta (SEV - LogReg): {delta:+.3f}  → {verdict}")

    if args.save:
        payload = {
            "note": (
                "Both models trained on the same training portion and evaluated on the "
                "same grouped held-out split (GroupShuffleSplit by scenario_family, "
                "test_size=0.2, seed=42). calls_fresh_eval_36.csv named in the dataset "
                "guidelines is absent, so this leakage-free split is the untouched test "
                "set. SEV k chosen by leave-one-out on TRAIN only. Scores are on synthetic "
                "data and are not evidence of real-world performance."
            ),
            "encoder": SemanticEvidenceVoter().model_name,
            "sev_k": best_k,
            "sev_k_loo_macro_f1": k_scores,
            "test_rows": int(len(X_test)),
            "logistic_regression": m_logreg,
            "semantic_evidence_voting": m_sev,
            "macro_f1_delta_sev_minus_logreg": float(delta),
        }
        REPORT_JSON.write_text(json.dumps(payload, indent=2))
        pd.DataFrame(m_logreg["confusion_matrix"], index=LABELS, columns=LABELS).to_csv(HERE / "confusion_logreg_heldout.csv")
        pd.DataFrame(m_sev["confusion_matrix"], index=LABELS, columns=LABELS).to_csv(HERE / "confusion_sev_heldout.csv")
        print(f"\nSaved {REPORT_JSON.name} + confusion CSVs.")


if __name__ == "__main__":
    main()
