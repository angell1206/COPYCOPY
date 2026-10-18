"""Tests for Semantic Evidence Voting (SEV).

The voting-math tests run with no model download (embeddings are injected). The
integration tests load the real encoder + saved index and are skipped if
sentence-transformers or the index file is unavailable.

Run with:  python -m pytest -q   (or: python tests/test_semantic.py)
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "models2.0"))

import semantic_evidence_voting as sev  # noqa: E402

VALID_LABELS = {"PRIORITY_REVIEW", "SUPPORT_REVIEW", "UNCLEAR_REVIEW"}
INDEX_PATH = REPO / "models2.0" / "sev_index.npz"


# --------------------- voting math (no model download) -----------------------
def _dummy_voter(labels, texts, k):
    v = sev.SemanticEvidenceVoter(k=k)
    v.fit_from_embeddings(np.zeros((len(labels), 3), dtype=np.float32), labels, texts)
    return v


def test_vote_proportions_sum_to_one_and_pick_argmax():
    labels = ["PRIORITY_REVIEW", "PRIORITY_REVIEW", "SUPPORT_REVIEW", "UNCLEAR_REVIEW"]
    texts = ["a", "b", "c", "d"]
    v = _dummy_voter(labels, texts, k=4)
    # Highest similarities go to the two PRIORITY examples.
    sims = np.array([0.9, 0.8, 0.2, 0.1])
    out = v._vote_from_similarities(sims)
    assert out["predicted_label"] == "PRIORITY_REVIEW"
    assert abs(sum(out["vote_proportions"].values()) - 1.0) < 1e-6
    assert out["confidence"] == max(out["vote_proportions"].values())


def test_supporting_examples_sorted_by_similarity():
    labels = ["PRIORITY_REVIEW", "SUPPORT_REVIEW", "UNCLEAR_REVIEW"]
    texts = ["p", "s", "u"]
    v = _dummy_voter(labels, texts, k=3)
    out = v._vote_from_similarities(np.array([0.1, 0.9, 0.5]), top_n_evidence=3)
    sims = [e["similarity"] for e in out["supporting_examples"]]
    assert sims == sorted(sims, reverse=True)
    assert out["supporting_examples"][0]["label"] == "SUPPORT_REVIEW"


def test_all_negative_similarity_abstains_to_unclear():
    labels = ["PRIORITY_REVIEW", "SUPPORT_REVIEW"]
    v = _dummy_voter(labels, ["p", "s"], k=2)
    out = v._vote_from_similarities(np.array([-0.3, -0.5]))
    assert out["predicted_label"] == "UNCLEAR_REVIEW"
    assert out["vote_proportions"]["UNCLEAR_REVIEW"] == 1.0


def test_k_limits_voting_neighbours():
    labels = ["PRIORITY_REVIEW", "SUPPORT_REVIEW", "SUPPORT_REVIEW"]
    v = _dummy_voter(labels, ["p", "s1", "s2"], k=1)
    # Only the single nearest neighbour (PRIORITY) should vote.
    out = v._vote_from_similarities(np.array([0.9, 0.8, 0.7]))
    assert out["predicted_label"] == "PRIORITY_REVIEW"
    assert out["neighbours_used"] == 1


# --------------------------- integration (real model) ------------------------
_HAS_ST = importlib.util.find_spec("sentence_transformers") is not None
needs_model = pytest.mark.skipif(
    not (_HAS_ST and INDEX_PATH.exists()),
    reason="sentence-transformers or sev_index.npz not available",
)


@needs_model
def test_loaded_index_predicts_valid_label():
    voter = sev.SemanticEvidenceVoter.load(INDEX_PATH)
    assert voter.n_examples > 100
    out = voter.predict_one("woh mera peecha kar raha hai aur dhamki de raha hai")
    assert out["predicted_label"] in VALID_LABELS
    assert abs(sum(out["vote_proportions"].values()) - 1.0) < 1e-6
    assert len(out["supporting_examples"]) >= 1


@needs_model
def test_predict_empty_raises():
    voter = sev.SemanticEvidenceVoter.load(INDEX_PATH)
    with pytest.raises(ValueError):
        voter.predict_one("   ")


@needs_model
def test_loo_excludes_self_and_covers_all_rows():
    voter = sev.SemanticEvidenceVoter.load(INDEX_PATH)
    preds = voter.predict_loo()
    assert len(preds) == voter.n_examples
    assert set(preds).issubset(VALID_LABELS)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
