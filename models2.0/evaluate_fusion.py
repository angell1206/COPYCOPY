"""Compare Logistic Regression, Semantic Evidence Voting, and Evidence Fusion.

All three are evaluated on the SAME untouched held-out TEST set. Data is split by
``scenario_family`` into train / validation / test with no family in more than one
split (GroupShuffleSplit, seed 42). The curated augmentation set is added to the
TRAIN side only.

  * Logistic Regression — fit on train.
  * SEV                 — retrieval pool = train only (test/val never in the pool).
  * Evidence Fusion     — blends the two models' class scores; the blend weight is
                          tuned on the VALIDATION split only (never on test).

Reports accuracy, macro F1, per-class precision/recall, and confusion matrix for
each model on the test set, and writes the tuned fusion weight to
fusion_config.json for the app to reuse.

Usage:
    python models2.0/evaluate_fusion.py            # print comparison
    python models2.0/evaluate_fusion.py --save     # also write report + config
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

from train_model import LABELS, TRAIN_CSV, build_pipeline
from semantic_evidence_voting import SemanticEvidenceVoter
from evidence_fusion import DEFAULT_LOW_CONFIDENCE, DEFAULT_LOW_MARGIN, fused_predictions, tune_weight

HERE = Path(__file__).resolve().parent
AUGMENTATION_CSV = HERE / "augmentation_v2_1.csv"
REPORT_JSON = HERE / "evaluation_fusion.json"
CONFIG_JSON = HERE / "fusion_config.json"
RANDOM_STATE = 42
SEV_K = 5


def _augmentation() -> pd.DataFrame:
    if AUGMENTATION_CSV.exists():
        return pd.read_csv(AUGMENTATION_CSV)[["transcript", "label"]].assign(
            transcript=lambda d: d["transcript"].astype(str))
    return pd.DataFrame(columns=["transcript", "label"])


def _proba_dicts(pipeline, texts) -> list[dict]:
    classes = list(pipeline.classes_)
    probs = pipeline.predict_proba(list(texts))
    return [{cls: float(p) for cls, p in zip(classes, row)} for row in probs]


def _metrics(y_true, y_pred) -> dict:
    rep = classification_report(y_true, y_pred, labels=LABELS, digits=3, zero_division=0, output_dict=True)
    return {
        "accuracy": float(rep["accuracy"]),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "priority_recall": float(recall_score(y_true, y_pred, labels=["PRIORITY_REVIEW"], average="macro", zero_division=0)),
        "per_class": {k: rep[k] for k in LABELS},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def _print_block(name, m, y_true, y_pred):
    print(f"\n============  {name}  ============")
    print(classification_report(y_true, y_pred, labels=LABELS, digits=3, zero_division=0))
    print(f"macro F1: {m['macro_f1']:.3f} | accuracy: {m['accuracy']:.3f} | PRIORITY recall: {m['priority_recall']:.3f}")
    print("confusion (rows=true):  " + "  ".join(f"{l.split('_')[0]:>9}" for l in LABELS))
    for lbl, row in zip(LABELS, m["confusion_matrix"]):
        print(f"  {lbl:>14}  " + "  ".join(f"{v:9d}" for v in row))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(TRAIN_CSV)
    X, y = df["transcript"].astype(str).values, df["label"].values
    groups = df["scenario_family"].astype(str).values
    aug = _augmentation()

    # ---- grouped 3-way split: test (20%), then val (20% of all), rest train ----
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    dev_idx, test_idx = next(gss1.split(X, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)  # 0.25*0.8 = 0.2
    tr_rel, va_rel = next(gss2.split(X[dev_idx], y[dev_idx], groups[dev_idx]))
    train_idx, val_idx = dev_idx[tr_rel], dev_idx[va_rel]

    fam_tr, fam_va, fam_te = set(groups[train_idx]), set(groups[val_idx]), set(groups[test_idx])
    assert not (fam_tr & fam_va) and not (fam_tr & fam_te) and not (fam_va & fam_te), "family leakage!"

    train_texts = np.concatenate([X[train_idx], aug["transcript"].values])
    train_labels = np.concatenate([y[train_idx], aug["label"].values])
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"train: {len(train_texts)} (incl. {len(aug)} aug) | val: {len(X_val)} | test: {len(X_test)} (untouched)")

    # ---- fit both base models on the SAME training data ----------------------
    logreg = build_pipeline().fit(train_texts, train_labels)
    sev = SemanticEvidenceVoter(k=SEV_K).fit(list(train_texts), list(train_labels))

    # ---- tune fusion weight on validation only -------------------------------
    val_lr = _proba_dicts(logreg, X_val)
    val_sev = [d["vote_proportions"] for d in sev.predict_details(list(X_val))]
    best_w, w_scores = tune_weight(val_lr, val_sev, y_val)
    print(f"\nFusion weight (LogReg share) tuned on VALIDATION: w={best_w}")
    print("  val macro F1 by weight: " + ", ".join(f"{w:.1f}:{s:.3f}" for w, s in w_scores.items()))

    # ---- evaluate all three on the untouched test set ------------------------
    test_lr = _proba_dicts(logreg, X_test)
    test_sev_details = sev.predict_details(list(X_test))
    test_sev = [d["vote_proportions"] for d in test_sev_details]

    logreg_pred = logreg.predict(X_test)
    sev_pred = [max(s, key=s.get) for s in test_sev]
    fusion_pred = fused_predictions(test_lr, test_sev, best_w)

    m_lr, m_sev, m_fus = _metrics(y_test, logreg_pred), _metrics(y_test, sev_pred), _metrics(y_test, fusion_pred)
    _print_block("Logistic Regression", m_lr, y_test, logreg_pred)
    _print_block(f"Semantic Evidence Voting (k={SEV_K})", m_sev, y_test, sev_pred)
    _print_block(f"Evidence Fusion (w={best_w})", m_fus, y_test, fusion_pred)

    print("\n============  SUMMARY (same untouched test set)  ============")
    print(f"{'metric':<20}{'LogReg':>9}{'SEV':>9}{'Fusion':>9}")
    for key, lbl in [("macro_f1", "macro F1"), ("accuracy", "accuracy"), ("priority_recall", "PRIORITY recall")]:
        print(f"{lbl:<20}{m_lr[key]:>9.3f}{m_sev[key]:>9.3f}{m_fus[key]:>9.3f}")
    best_model = max([("LogReg", m_lr), ("SEV", m_sev), ("Fusion", m_fus)], key=lambda kv: kv[1]["macro_f1"])[0]
    print(f"\nHighest macro F1: {best_model}")

    if args.save:
        CONFIG_JSON.write_text(json.dumps({
            "weight": best_w,
            "low_confidence": DEFAULT_LOW_CONFIDENCE,
            "low_margin": DEFAULT_LOW_MARGIN,
            "note": "LogReg share of the fused score, tuned on the validation split only.",
        }, indent=2))
        REPORT_JSON.write_text(json.dumps({
            "note": (
                "Grouped train/val/test split by scenario_family (seed 42; test 20%, "
                "val 20%, train 60%+augmentation). SEV retrieval pool = train only; "
                "fusion weight tuned on validation only. Test set is untouched. Scores "
                "are on synthetic data and are not evidence of real-world performance."
            ),
            "fusion_weight_logreg_share": best_w,
            "fusion_weight_val_macro_f1": w_scores,
            "test_rows": int(len(X_test)),
            "logistic_regression": m_lr,
            "semantic_evidence_voting": m_sev,
            "evidence_fusion": m_fus,
        }, indent=2))
        pd.DataFrame(m_fus["confusion_matrix"], index=LABELS, columns=LABELS).to_csv(HERE / "confusion_fusion_heldout.csv")
        print(f"\nSaved {REPORT_JSON.name}, {CONFIG_JSON.name}, confusion_fusion_heldout.csv")


if __name__ == "__main__":
    main()
