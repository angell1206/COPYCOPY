"""Tests for the rule-based evidence layer and the models2.0 ML inference.

Run with:  python -m pytest -q   (or: python tests/test_triage.py)
"""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from triage_rules import analyze_transcript, find_active_phrases, HIGH_PHRASES, CRITICAL_PHRASES

# Load the models2.0 inference helper by file path (dotted folder name).
_spec = importlib.util.spec_from_file_location(
    "triage_model_v2", REPO / "models2.0" / "triage_model.py"
)
triage_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(triage_model)

VALID_LABELS = {"PRIORITY_REVIEW", "SUPPORT_REVIEW", "UNCLEAR_REVIEW"}


# --------------------------- rule-based evidence -----------------------------
def test_rules_critical_weapon():
    r = analyze_transcript("He has a weapon in his hand and is attacking me right now.")
    assert r["urgency"] == "Critical"
    assert r["evidence"]


def test_rules_following_help_english():
    # The originally-flagged pattern must now surface rule evidence.
    r = analyze_transcript("I am being followed by a man. Please send help.")
    assert r["urgency"] in {"Critical", "High"}
    assert any("follow" in e or "help" in e for e in r["evidence"])


def test_rules_feeling_unsafe():
    r = analyze_transcript("I feel unsafe because a man keeps following me.")
    assert r["urgency"] == "High"


def test_rules_hinglish_following():
    r = analyze_transcript("Woh mera peecha kar raha hai aur dhamki de raha hai.")
    assert r["urgency"] == "High"
    assert r["threat"] in {"Possible stalking / following", "Possible threat / coercion"}


def test_rules_devanagari_help():
    r = analyze_transcript("कोई मेरा पीछा कर रहा है, कृपया मदद भेजें।")
    assert r["urgency"] in {"Critical", "High"}


def test_rules_medium_harassment():
    r = analyze_transcript("Someone keeps sending unwanted messages and harassing me online.")
    assert r["urgency"] == "Medium"


def test_rules_negation_not_following():
    r = analyze_transcript("He is not following me and he has not threatened me.")
    assert r["urgency"] == "Unclear"


def test_rules_information_request():
    r = analyze_transcript("I need information about reporting harassment. No immediate danger.")
    assert r["urgency"] == "Unclear"
    assert r["information_request"] is True


def test_rules_empty_transcript():
    r = analyze_transcript("")
    assert r["urgency"] == "Unclear"
    assert r["evidence"] == []


def test_rules_location_clue():
    r = analyze_transcript("A man is following me near the metro station right now.")
    assert "metro station" in r["location_clues"]


def test_find_active_phrases_dedup_and_negation():
    assert "following me" in find_active_phrases("he is following me", HIGH_PHRASES)
    assert "trying to kill me" not in find_active_phrases("he is not trying to kill me", CRITICAL_PHRASES)


# ------------------------------ ML inference ---------------------------------
def test_model_loads_and_has_three_classes():
    model = triage_model.load_model()
    assert VALID_LABELS.issubset(set(triage_model._model_classes(model)))


def test_predict_returns_contract():
    r = triage_model.predict_transcript("A man has a knife and is attacking me now.")
    for key in ["predicted_label", "class_scores", "model_confidence", "review_required"]:
        assert key in r
    assert r["review_required"] is True
    assert abs(sum(r["class_scores"].values()) - 1.0) < 1e-6
    assert r["predicted_label"] in VALID_LABELS


def test_predict_empty_raises():
    try:
        triage_model.predict_transcript("   ")
    except ValueError:
        return
    raise AssertionError("empty transcript should raise ValueError")


def test_followed_help_is_not_unclear():
    # Root-cause regression: this must no longer be UNCLEAR_REVIEW.
    r = triage_model.predict_transcript("i am being followed by a man. please send help")
    assert r["predicted_label"] != "UNCLEAR_REVIEW"


def test_hypothetical_followed_stays_unclear():
    # The fix must not break the hypothetical/quoted boundary.
    r = triage_model.predict_transcript("My cousin asked what to do if somebody is being followed.")
    assert r["predicted_label"] == "UNCLEAR_REVIEW"


def test_information_request_unclear():
    r = triage_model.predict_transcript("I just want information about reporting an old incident.")
    assert r["predicted_label"] == "UNCLEAR_REVIEW"


if __name__ == "__main__":
    import traceback

    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                passed += 1
                print(f"PASS {name}")
            except Exception:
                failed += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
