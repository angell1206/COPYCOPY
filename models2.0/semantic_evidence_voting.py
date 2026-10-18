"""Semantic Evidence Voting (SEV) — our proposed triage approach.

SEV is a transparent, non-parametric classifier. It embeds every labelled
*training* transcript with a multilingual Sentence Transformer
(``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2``), then classifies
a new transcript by:

  1. embedding the query transcript with the same encoder,
  2. computing cosine similarity to each training embedding,
  3. taking the top-k most similar training examples ("evidence"),
  4. casting a similarity-weighted vote for each example's ``label``,
  5. normalising the per-class vote mass into vote proportions,
  6. predicting the class with the largest vote proportion.

Voting is over labelled TRAINING examples only (never the query itself or any
held-out/test example), so there is no data leakage. The nearest neighbours are
returned as human-readable supporting evidence, and the vote proportions are the
normalised share of similarity mass per class — NOT calibrated probabilities of
real-world correctness.

"SEV" here stands for Semantic Evidence Voting, the name we propose for this
method. It is decision support only and must never auto-dispatch or close a case.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LABELS = ["PRIORITY_REVIEW", "SUPPORT_REVIEW", "UNCLEAR_REVIEW"]
DEFAULT_K = 25

_ENCODER_CACHE: dict[str, Any] = {}


def get_encoder(model_name: str = MODEL_NAME):
    """Load (and cache) the Sentence Transformer encoder for this process."""
    if model_name not in _ENCODER_CACHE:
        from sentence_transformers import SentenceTransformer  # lazy, heavy import

        _ENCODER_CACHE[model_name] = SentenceTransformer(model_name)
    return _ENCODER_CACHE[model_name]


def embed(texts: list[str], model_name: str = MODEL_NAME) -> np.ndarray:
    """Return L2-normalised embeddings (cosine similarity == dot product)."""
    encoder = get_encoder(model_name)
    vecs = encoder.encode(
        list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )
    return np.asarray(vecs, dtype=np.float32)


@dataclass
class SemanticEvidenceVoter:
    """Similarity-weighted voting over labelled training embeddings."""

    k: int = DEFAULT_K
    model_name: str = MODEL_NAME
    labels: list[str] = field(default_factory=lambda: list(LABELS))
    _emb: np.ndarray | None = None
    _y: np.ndarray | None = None
    _texts: list[str] = field(default_factory=list)

    # ---- fitting / index I/O ------------------------------------------------
    def fit(self, transcripts: list[str], labels: list[str]) -> "SemanticEvidenceVoter":
        self._texts = [str(t) for t in transcripts]
        self._y = np.asarray([str(l) for l in labels])
        self._emb = embed(self._texts, self.model_name)
        return self

    def fit_from_embeddings(self, emb: np.ndarray, labels, transcripts) -> "SemanticEvidenceVoter":
        self._emb = np.asarray(emb, dtype=np.float32)
        self._y = np.asarray([str(l) for l in labels])
        self._texts = [str(t) for t in transcripts]
        return self

    def save(self, path: str | Path) -> None:
        if self._emb is None:
            raise RuntimeError("Nothing to save — call fit() first.")
        np.savez_compressed(
            Path(path), embeddings=self._emb, labels=self._y,
            transcripts=np.array(self._texts, dtype=object), k=self.k, model_name=self.model_name,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SemanticEvidenceVoter":
        data = np.load(Path(path), allow_pickle=True)
        voter = cls(k=int(data["k"]), model_name=str(data["model_name"]))
        voter._emb = data["embeddings"].astype(np.float32)
        voter._y = data["labels"].astype(str)
        voter._texts = [str(t) for t in data["transcripts"]]
        return voter

    @property
    def n_examples(self) -> int:
        return 0 if self._emb is None else int(self._emb.shape[0])

    # ---- prediction ---------------------------------------------------------
    def _vote_from_similarities(self, sims: np.ndarray, top_n_evidence: int = 5) -> dict[str, Any]:
        k = min(self.k, sims.shape[0])
        order = np.argsort(-sims)
        top_idx = order[:k]
        weights = np.clip(sims[top_idx], 0.0, None)  # only positive similarity votes

        votes = {label: 0.0 for label in self.labels}
        for idx, w in zip(top_idx, weights):
            lbl = str(self._y[idx])
            votes[lbl] = votes.get(lbl, 0.0) + float(w)

        total = sum(votes.values())
        if total <= 0.0:  # degenerate: no positive similarity — abstain to UNCLEAR
            proportions = {label: 0.0 for label in self.labels}
            proportions["UNCLEAR_REVIEW"] = 1.0
            predicted = "UNCLEAR_REVIEW"
        else:
            proportions = {label: votes[label] / total for label in self.labels}
            predicted = max(proportions, key=proportions.get)

        evidence = [
            {
                "transcript": self._texts[int(i)],
                "label": str(self._y[int(i)]),
                "similarity": float(sims[int(i)]),
            }
            for i in order[:top_n_evidence]
        ]
        return {
            "predicted_label": predicted,
            "vote_proportions": proportions,
            "confidence": float(proportions[predicted]),
            "neighbours_used": int(k),
            "supporting_examples": evidence,
        }

    def predict_one(self, transcript: str, top_n_evidence: int = 5) -> dict[str, Any]:
        if self._emb is None:
            raise RuntimeError("Voter is not fitted.")
        text = (transcript or "").strip()
        if not text:
            raise ValueError("Enter a transcript before analysing the call.")
        q = embed([text], self.model_name)[0]
        sims = self._emb @ q
        return self._vote_from_similarities(sims, top_n_evidence=top_n_evidence)

    def predict(self, transcripts: list[str]) -> list[str]:
        """Batch prediction (query embeddings are NOT added to the index)."""
        if self._emb is None:
            raise RuntimeError("Voter is not fitted.")
        Q = embed([str(t) for t in transcripts], self.model_name)
        sims_all = Q @ self._emb.T  # (n_query, n_train)
        return [self._vote_from_similarities(sims_all[i])["predicted_label"] for i in range(Q.shape[0])]

    def predict_loo(self) -> list[str]:
        """Leave-one-out prediction over the training index (self excluded).

        Used only for leakage-free hyper-parameter selection on training data.
        """
        if self._emb is None:
            raise RuntimeError("Voter is not fitted.")
        sims_all = self._emb @ self._emb.T
        np.fill_diagonal(sims_all, -np.inf)  # exclude self
        return [self._vote_from_similarities(sims_all[i])["predicted_label"] for i in range(sims_all.shape[0])]
