"""Train the models2.0 helpline triage classifier (reproducible).

Pipeline: TF-IDF word (1-2) + character_wb (3-5) FeatureUnion -> balanced
Logistic Regression. This matches the word+character configuration the team's
own model-comparison report selected, and fixes a discovered mismatch where the
previously shipped artefact was word-only.

Training data = data/calls_train_v2.csv (270 dev examples) + a small curated,
rubric-consistent augmentation set (augmentation_v2_1.csv) that closes a coverage
gap for present-tense "being followed / please send help" style transcripts. The
augmentation teaches the pattern across English, Hindi and Romanized Hinglish; it
is not the specific flagged example and does not hardcode any prediction.

The SHIPPED artefact is trained on ALL available training data for the strongest
demo. Honest held-out metrics are produced separately by evaluate_model.py using
a grouped (leakage-free) split; do not read accuracy off this script.

Usage:
    python models2.0/train_model.py          # retrain and overwrite the artefact
    python models2.0/train_model.py --dry-run # build + sanity-check, do not save
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TRAIN_CSV = REPO / "data" / "calls_train_v2.csv"
AUGMENTATION_CSV = HERE / "augmentation_v2_1.csv"
MODEL_PATH = HERE / "helpline_triage_model.joblib"

LABELS = ["PRIORITY_REVIEW", "SUPPORT_REVIEW", "UNCLEAR_REVIEW"]


def build_pipeline() -> Pipeline:
    """Word + character TF-IDF -> balanced Logistic Regression."""
    word = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), strip_accents="unicode",
        sublinear_tf=True, min_df=2, max_features=30000,
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), strip_accents="unicode",
        sublinear_tf=True, min_df=2, max_features=40000,
    )
    features = FeatureUnion([("word", word), ("char", char)])
    classifier = LogisticRegression(
        C=3.0, class_weight="balanced", max_iter=2000, random_state=42,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def load_training_frame() -> pd.DataFrame:
    """Combine the dev corpus with the curated augmentation set."""
    train = pd.read_csv(TRAIN_CSV)
    frames = [train[["transcript", "label"]]]
    if AUGMENTATION_CSV.exists():
        aug = pd.read_csv(AUGMENTATION_CSV)
        frames.append(aug[["transcript", "label"]])
    combined = pd.concat(frames, ignore_index=True)
    combined["transcript"] = combined["transcript"].astype(str)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Build without saving.")
    args = parser.parse_args()

    data = load_training_frame()
    print(f"Training rows: {len(data)}")
    print(data["label"].value_counts().to_string())

    pipeline = build_pipeline()
    pipeline.fit(data["transcript"].values, data["label"].values)

    # Sanity check: the pipeline must expose the three routing labels.
    classes = list(pipeline.classes_)
    assert set(classes) == set(LABELS), f"Unexpected classes: {classes}"
    print(f"Fitted. Classes: {classes}")

    if args.dry_run:
        print("Dry run — model not saved.")
        return

    import joblib

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
