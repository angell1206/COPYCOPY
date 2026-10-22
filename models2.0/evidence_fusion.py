"""Evidence Fusion — combine Logistic Regression and Semantic Evidence Voting.

The fusion layer takes the class-level scores of both existing models and blends
them with a single configurable weight ``w`` (the Logistic Regression share):

    fused[c] = w * logreg[c] + (1 - w) * sev[c]      (then renormalised)

``w`` is tuned on a VALIDATION split (see evaluate_fusion.py), never on the test
set. Both inputs are first normalised to non-negative scores summing to 1, so
LogReg probabilities and SEV vote proportions are combined on the same scale.

The fused result also carries human-review signals:
  * ``models_agree``   — do LogReg and SEV pick the same label?
  * ``low_confidence`` — is the top fused score below a threshold?
  * ``low_margin``     — is the gap between the top two fused scores small?
  * ``review_flag``    — any of the above → recommend prompt human review.

These flags only ever *raise* attention; they never suppress review. This is
decision support, not a validated risk model, and must never auto-dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

LABELS = ["PRIORITY_REVIEW", "SUPPORT_REVIEW", "UNCLEAR_REVIEW"]

DEFAULT_WEIGHT = 0.5          # LogReg share; overridden by the tuned config
DEFAULT_LOW_CONFIDENCE = 0.50  # top fused score below this → low-confidence flag
DEFAULT_LOW_MARGIN = 0.15      # top1 - top2 below this → ambiguous flag


def normalize_scores(scores: dict[str, float], labels: list[str] = LABELS) -> dict[str, float]:
    """Clip negatives and renormalise to sum to 1 over the fixed label set."""
    vals = {lbl: max(float(scores.get(lbl, 0.0)), 0.0) for lbl in labels}
    total = sum(vals.values())
    if total <= 0.0:
        return {lbl: 1.0 / len(labels) for lbl in labels}
    return {lbl: v / total for lbl, v in vals.items()}


def fuse_scores(logreg_scores: dict[str, float], sev_scores: dict[str, float],
                weight: float = DEFAULT_WEIGHT, labels: list[str] = LABELS) -> dict[str, float]:
    """Weighted blend of two score dicts; ``weight`` is the LogReg share in [0, 1]."""
    w = min(max(float(weight), 0.0), 1.0)
    lr = normalize_scores(logreg_scores, labels)
    sv = normalize_scores(sev_scores, labels)
    fused = {lbl: w * lr[lbl] + (1.0 - w) * sv[lbl] for lbl in labels}
    return normalize_scores(fused, labels)


@dataclass
class FusionConfig:
    weight: float = DEFAULT_WEIGHT
    low_confidence: float = DEFAULT_LOW_CONFIDENCE
    low_margin: float = DEFAULT_LOW_MARGIN


def fuse(logreg_scores: dict[str, float], sev_scores: dict[str, float],
         config: FusionConfig | None = None, labels: list[str] = LABELS) -> dict[str, Any]:
    """Fuse two models' class scores and derive the human-review signals."""
    cfg = config or FusionConfig()
    lr = normalize_scores(logreg_scores, labels)
    sv = normalize_scores(sev_scores, labels)
    fused = fuse_scores(lr, sv, cfg.weight, labels)

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    predicted, top_score = ordered[0]
    margin = top_score - (ordered[1][1] if len(ordered) > 1 else 0.0)

    lr_label = max(lr, key=lr.get)
    sev_label = max(sv, key=sv.get)
    models_agree = lr_label == sev_label
    low_confidence = top_score < cfg.low_confidence
    low_margin = margin < cfg.low_margin

    reasons: list[str] = []
    if not models_agree:
        reasons.append(f"Models disagree (Logistic Regression → {lr_label}, SEV → {sev_label}).")
    if low_confidence:
        reasons.append(f"Low fused confidence (top score {top_score*100:.0f}% < {cfg.low_confidence*100:.0f}%).")
    if low_margin:
        reasons.append(f"Narrow margin between top classes ({margin*100:.0f}%).")

    return {
        "predicted_label": predicted,
        "scores": fused,
        "confidence": float(top_score),
        "margin": float(margin),
        "weight": float(cfg.weight),
        "logreg_label": lr_label,
        "sev_label": sev_label,
        "models_agree": models_agree,
        "low_confidence": low_confidence,
        "low_margin": low_margin,
        "review_flag": bool((not models_agree) or low_confidence or low_margin),
        "review_reasons": reasons,
    }


def fused_predictions(logreg_scores_list, sev_scores_list, weight: float, labels: list[str] = LABELS) -> list[str]:
    """Batch argmax of fused scores (used by weight tuning / evaluation)."""
    cfg = FusionConfig(weight=weight)
    return [fuse(lr, sv, cfg, labels)["predicted_label"] for lr, sv in zip(logreg_scores_list, sev_scores_list)]


def tune_weight(val_logreg_scores, val_sev_scores, y_val, candidates=None,
                labels: list[str] = LABELS):
    """Pick the LogReg weight maximising macro F1 on the validation split only.

    Returns (best_weight, {weight: macro_f1}). No test data is used here.
    """
    from sklearn.metrics import f1_score

    if candidates is None:
        candidates = [round(0.1 * i, 2) for i in range(0, 11)]  # 0.0 .. 1.0
    scores: dict[float, float] = {}
    for w in candidates:
        preds = fused_predictions(val_logreg_scores, val_sev_scores, w, labels)
        scores[w] = float(f1_score(y_val, preds, average="macro"))
    # Prefer the best; on ties prefer the weight closest to 0.5 (balanced blend).
    best = max(candidates, key=lambda w: (scores[w], -abs(w - 0.5)))
    return best, scores
