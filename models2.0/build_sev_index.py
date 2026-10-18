"""Build and save the Semantic Evidence Voting index for the app.

Encodes ALL labelled training transcripts (data/calls_train_v2.csv +
augmentation_v2_1.csv) with the multilingual Sentence Transformer and saves the
embeddings, labels, and transcripts to ``sev_index.npz`` so the Streamlit app can
load the index without re-encoding the training set on every start. The query
transcript is still encoded live at inference time.

Note: this is the FULL-training index used for the live demo. The honest
model comparison in evaluate_semantic.py uses a separate leakage-free held-out
split and never touches this file.

Usage:
    python models2.0/build_sev_index.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from train_model import TRAIN_CSV
from semantic_evidence_voting import SemanticEvidenceVoter

HERE = Path(__file__).resolve().parent
AUGMENTATION_CSV = HERE / "augmentation_v2_1.csv"
INDEX_PATH = HERE / "sev_index.npz"
# k chosen by leave-one-out on the training set in evaluate_semantic.py.
INDEX_K = 5


def main() -> None:
    train = pd.read_csv(TRAIN_CSV)[["transcript", "label"]]
    frames = [train]
    if AUGMENTATION_CSV.exists():
        frames.append(pd.read_csv(AUGMENTATION_CSV)[["transcript", "label"]])
    data = pd.concat(frames, ignore_index=True)
    data["transcript"] = data["transcript"].astype(str)
    print(f"Encoding {len(data)} training transcripts…")

    voter = SemanticEvidenceVoter(k=INDEX_K).fit(data["transcript"].tolist(), data["label"].tolist())
    voter.save(INDEX_PATH)
    print(f"Saved SEV index ({voter.n_examples} examples) to {INDEX_PATH}")


if __name__ == "__main__":
    main()
