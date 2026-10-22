"""Tests for the Evidence Fusion layer (pure score math — no model download)."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "models2.0"))

import evidence_fusion as ef  # noqa: E402

L = ef.LABELS
VALID = set(L)


def _d(p, s, u):
    return {"PRIORITY_REVIEW": p, "SUPPORT_REVIEW": s, "UNCLEAR_REVIEW": u}


def test_normalize_clips_negatives_and_sums_to_one():
    out = ef.normalize_scores(_d(-0.2, 0.6, 0.6))
    assert out["PRIORITY_REVIEW"] == 0.0
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_normalize_degenerate_is_uniform():
    out = ef.normalize_scores(_d(0.0, 0.0, 0.0))
    assert all(abs(v - 1 / 3) < 1e-9 for v in out.values())


def test_fuse_scores_weighted_blend():
    lr = _d(0.8, 0.1, 0.1)
    sev = _d(0.0, 0.0, 1.0)
    fused = ef.fuse_scores(lr, sev, weight=0.5)
    # 0.5*0.8 + 0.5*0.0 = 0.4 priority; 0.5*0.1 + 0.5*1.0 = 0.55 unclear
    assert abs(sum(fused.values()) - 1.0) < 1e-9
    assert fused["UNCLEAR_REVIEW"] > fused["PRIORITY_REVIEW"]


def test_fuse_weight_one_is_pure_logreg():
    lr = _d(0.7, 0.2, 0.1)
    sev = _d(0.1, 0.1, 0.8)
    fused = ef.fuse_scores(lr, sev, weight=1.0)
    assert fused == pytest.approx(ef.normalize_scores(lr))


def test_fuse_detects_agreement():
    out = ef.fuse(_d(0.8, 0.1, 0.1), _d(0.7, 0.2, 0.1))
    assert out["predicted_label"] == "PRIORITY_REVIEW"
    assert out["models_agree"] is True
    assert "PRIORITY" in out["logreg_label"] and "PRIORITY" in out["sev_label"]


def test_fuse_flags_disagreement():
    out = ef.fuse(_d(0.8, 0.1, 0.1), _d(0.1, 0.1, 0.8))
    assert out["models_agree"] is False
    assert out["review_flag"] is True
    assert any("disagree" in r.lower() for r in out["review_reasons"])


def test_fuse_flags_low_confidence():
    cfg = ef.FusionConfig(weight=0.5, low_confidence=0.6, low_margin=0.0)
    out = ef.fuse(_d(0.4, 0.35, 0.25), _d(0.4, 0.35, 0.25), cfg)
    assert out["low_confidence"] is True
    assert out["review_flag"] is True


def test_fuse_flags_low_margin():
    cfg = ef.FusionConfig(weight=0.5, low_confidence=0.0, low_margin=0.1)
    out = ef.fuse(_d(0.5, 0.48, 0.02), _d(0.5, 0.48, 0.02), cfg)
    assert out["low_margin"] is True
    assert out["review_flag"] is True


def test_fuse_output_contract():
    out = ef.fuse(_d(0.6, 0.3, 0.1), _d(0.5, 0.4, 0.1))
    assert out["predicted_label"] in VALID
    assert abs(sum(out["scores"].values()) - 1.0) < 1e-9
    assert 0.0 <= out["confidence"] <= 1.0


def test_tune_weight_prefers_better_component():
    # Validation where SEV is always right and LogReg always wrong → weight favours SEV (w small).
    y = ["PRIORITY_REVIEW", "SUPPORT_REVIEW", "UNCLEAR_REVIEW"]
    lr = [_d(0.1, 0.8, 0.1), _d(0.1, 0.1, 0.8), _d(0.8, 0.1, 0.1)]       # wrong
    sev = [_d(0.8, 0.1, 0.1), _d(0.1, 0.8, 0.1), _d(0.1, 0.1, 0.8)]      # right
    best_w, scores = ef.tune_weight(lr, sev, y)
    assert best_w <= 0.5
    assert scores[0.0] >= scores[1.0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
