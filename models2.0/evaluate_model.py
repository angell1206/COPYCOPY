"""Honest, leakage-free evaluation of the models2.0 triage pipeline.

The dataset guidelines reference a frozen 36-row fresh-eval file
(calls_fresh_eval_36.csv) that is NOT present in this repository, so a grouped
held-out split is constructed here instead. Splitting is done by
``scenario_family`` with GroupShuffleSplit so that no scenario family appears in
both train and test (prevents near-duplicate leakage). The curated augmentation
set is added to the TRAIN side only and never to the test side.

Reported: per-class precision/recall/F1, macro F1, accuracy, and the confusion
matrix on the held-out split, plus grouped 5-fold cross-validation on the dev
corpus for stability. Hyperparameters are NOT tuned against the test split.

Usage:
    python models2.0/evaluate_model.py            # print report
    python models2.0/evaluate_model.py --save     # also write JSON + CSV
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from train_model import LABELS, TRAIN_CSV, build_pipeline  # local import

HERE = Path(__file__).resolve().parent
AUGMENTATION_CSV = HERE / "augmentation_v2_1.csv"
REPORT_JSON = HERE / "evaluation_report_v2_1.json"
CONFUSION_CSV = HERE / "confusion_matrix_v2_1.csv"
RANDOM_STATE = 42


def _augmentation() -> pd.DataFrame:
    if AUGMENTATION_CSV.exists():
        aug = pd.read_csv(AUGMENTATION_CSV)
        return aug[["transcript", "label"]].assign(transcript=lambda d: d["transcript"].astype(str))
    return pd.DataFrame(columns=["transcript", "label"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="Write JSON + CSV report.")
    args = parser.parse_args()

    train = pd.read_csv(TRAIN_CSV)
    X = train["transcript"].astype(str).values
    y = train["label"].values
    groups = train["scenario_family"].astype(str).values
    aug = _augmentation()

    # ---- Grouped held-out split (no scenario family in both sides) ------------
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    overlap = set(groups[train_idx]) & set(groups[test_idx])
    assert not overlap, f"Family leakage between train and test: {overlap}"

    X_train = np.concatenate([X[train_idx], aug["transcript"].values])
    y_train = np.concatenate([y[train_idx], aug["label"].values])
    X_test, y_test = X[test_idx], y[test_idx]

    model = build_pipeline().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test, y_pred, labels=LABELS, digits=3, zero_division=0, output_dict=True
    )
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)

    print("=== Held-out grouped split (augmentation on train side only) ===")
    print(f"train rows: {len(X_train)} (incl. {len(aug)} augmentation) | test rows: {len(X_test)}")
    print(classification_report(y_test, y_pred, labels=LABELS, digits=3, zero_division=0))
    print(f"macro F1: {macro_f1:.3f}")
    print("Confusion matrix (rows = true, cols = predicted):")
    print("            " + "  ".join(f"{l.split('_')[0]:>9}" for l in LABELS))
    for lbl, row in zip(LABELS, cm):
        print(f"{lbl:>12}  " + "  ".join(f"{v:9d}" for v in row))

    # ---- Grouped 5-fold CV on the dev corpus (augmentation added per fold) ----
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_f1 = []
    for tr_i, va_i in cv.split(X, y, groups):
        Xtr = np.concatenate([X[tr_i], aug["transcript"].values])
        ytr = np.concatenate([y[tr_i], aug["label"].values])
        fold = build_pipeline().fit(Xtr, ytr)
        cv_f1.append(f1_score(y[va_i], fold.predict(X[va_i]), average="macro"))
    print(f"\nGrouped 5-fold CV macro F1: mean={np.mean(cv_f1):.3f} sd={np.std(cv_f1):.3f}")

    if args.save:
        payload = {
            "note": (
                "Grouped held-out split by scenario_family (GroupShuffleSplit, "
                "test_size=0.2, seed=42). calls_fresh_eval_36.csv referenced in the "
                "dataset guidelines is absent from the repo, so this leakage-free "
                "split is used. Augmentation is on the train side only. Scores are "
                "uncalibrated model outputs on synthetic data, not real-world risk."
            ),
            "model": "FeatureUnion(word 1-2 + char_wb 3-5) -> LogisticRegression C=3 balanced",
            "heldout_rows": int(len(X_test)),
            "heldout_accuracy": float(report["accuracy"]),
            "heldout_macro_f1": macro_f1,
            "per_class": {k: report[k] for k in LABELS},
            "confusion_matrix": {"labels": LABELS, "rows_true_cols_pred": cm.tolist()},
            "grouped_cv_macro_f1_mean": float(np.mean(cv_f1)),
            "grouped_cv_macro_f1_sd": float(np.std(cv_f1)),
        }
        REPORT_JSON.write_text(json.dumps(payload, indent=2))
        pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(CONFUSION_CSV)
        print(f"\nSaved {REPORT_JSON.name} and {CONFUSION_CSV.name}")


if __name__ == "__main__":
    main()
