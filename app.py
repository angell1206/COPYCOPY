"""AI-Based Help Line Triage — Hinglish helpline transcript demo (v2.1).

Operator-support dashboard that pairs a trained ML triage classifier
(``models2.0/``) with a transparent, rule-based evidence layer
(``triage_rules.py``). Transcripts can be typed or produced from a short audio
clip via Sarvam AI speech-to-text. The ML model gives the headline routing label
with an uncalibrated model score; the keyword rules give human-readable evidence,
location clues, and information still to verify. They are kept separate so an
operator can cross-check them.

Fictional/demo transcripts only. This is decision support, not a validated
emergency-risk model, and it must never dispatch services.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import math
import requests
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from triage_rules import analyze_transcript

AUDIO_TYPES = ["wav", "mp3", "aac", "aiff", "ogg", "opus", "flac", "mp4", "m4a", "amr", "wma", "webm"]

# -----------------------------------------------------------------------------
# Page setup and design system — warm ivory / sage / muted teal / champagne
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Helpline Triage",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --ivory: #f5efe3;        /* warm page background */
        --cream: #fffdf8;        /* elevated card surface */
        --cream-2: #faf5ea;      /* secondary surface */
        --champagne: #ece0c8;    /* beige highlight */
        --sage: #7e9b84;         /* primary accent */
        --sage-deep: #5f7d66;
        --teal: #4f8688;         /* secondary accent */
        --teal-deep: #3c6b6d;
        --ink: #3a382f;          /* soft charcoal text */
        --muted: #8c867a;        /* muted text */
        --border: #e6dcc8;       /* elegant warm border */
        --shadow: 0 6px 20px rgba(90, 77, 48, 0.08);
        --shadow-sm: 0 2px 8px rgba(90, 77, 48, 0.06);

        --priority: #b05f45;  --priority-bg: #f4e6dd;  /* muted clay */
        --support:  #a9843f;  --support-bg:  #f3ebd6;  /* muted gold */
        --unclear:  #5f7d76;  --unclear-bg:  #e6ede8;  /* sage-teal */
      }

      /* Force the warm theme regardless of system light/dark preference. */
      .stApp, [data-testid="stAppViewContainer"] { background: var(--ivory); }
      [data-testid="stHeader"] { background: transparent; }
      [data-testid="stSidebar"] { background: var(--cream-2); border-right: 1px solid var(--border); }
      .block-container { padding-top: 1.1rem; padding-bottom: 2.6rem; max-width: 1180px; }

      html, body, [class*="css"], .stMarkdown, p, span, label, li { color: var(--ink); }
      h1, h2, h3, h4, h5 { color: var(--ink); font-weight: 700; letter-spacing: -0.01em; }
      h2 { font-size: 1.28rem !important; }
      h5 { font-size: 1.02rem !important; margin-bottom: 0.2rem; }

      .app-header {
        display: flex; align-items: center; gap: 0.85rem;
        background: linear-gradient(180deg, var(--cream) 0%, var(--cream-2) 100%);
        border: 1px solid var(--border); border-left: 5px solid var(--sage);
        border-radius: 0.9rem; padding: 0.85rem 1.1rem; box-shadow: var(--shadow-sm);
        margin-bottom: 0.7rem;
      }
      .app-header .mark { font-size: 1.7rem; }
      .app-header .title { font-size: 1.32rem; font-weight: 700; color: var(--ink); line-height: 1.15; }
      .app-header .sub { font-size: 0.82rem; color: var(--muted); }

      [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--cream); border: 1px solid var(--border) !important;
        border-radius: 0.9rem; box-shadow: var(--shadow);
        padding: 0.3rem 0.25rem;
      }
      [data-testid="column"] [data-testid="stVerticalBlockBorderWrapper"] { box-shadow: var(--shadow-sm); }

      .eyebrow {
        display: inline-block; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: var(--sage-deep); margin-bottom: 0.1rem;
      }
      .card-title { font-size: 1.05rem; font-weight: 700; color: var(--ink); margin: 0 0 0.3rem; }

      .triage-badge {
        display: inline-flex; align-items: center; gap: 0.45rem;
        padding: 0.42rem 0.9rem; border-radius: 0.65rem;
        font-weight: 700; font-size: 1rem; border: 1px solid transparent;
      }
      .badge-sm { font-size: 0.82rem; padding: 0.28rem 0.65rem; }
      .b-priority { background: var(--priority-bg); color: var(--priority); border-color: var(--priority); }
      .b-support  { background: var(--support-bg);  color: var(--support);  border-color: var(--support); }
      .b-unclear  { background: var(--unclear-bg);  color: var(--unclear);  border-color: var(--unclear); }
      .b-neutral  { background: var(--champagne); color: var(--ink); border-color: var(--border); }

      .score-row { margin: 0.4rem 0; }
      .score-label { font-size: 0.86rem; color: var(--ink); display: flex; justify-content: space-between; margin-bottom: 0.22rem; font-weight: 600; }
      .score-track { background: var(--champagne); border-radius: 7px; height: 0.55rem; overflow: hidden; }
      .score-fill { height: 100%; border-radius: 7px; }
      .f-priority { background: linear-gradient(90deg, #c07558, var(--priority)); }
      .f-support  { background: linear-gradient(90deg, #c2a05a, var(--support)); }
      .f-unclear  { background: linear-gradient(90deg, #8aa79c, var(--unclear)); }

      .chip {
        display: inline-block; background: var(--cream-2); color: var(--ink);
        border: 1px solid var(--border); border-radius: 0.55rem;
        padding: 0.18rem 0.55rem; margin: 0.14rem 0.22rem 0.14rem 0; font-size: 0.82rem;
      }
      .chip-sage { background: var(--unclear-bg); border-color: var(--sage); color: var(--sage-deep); }

      .kv { margin: 0.22rem 0; color: var(--ink); font-size: 0.93rem; }
      .kv b { color: var(--muted); font-weight: 600; }
      .muted { color: var(--muted); font-size: 0.84rem; }
      .verify-item {
        background: var(--cream-2); border-left: 3px solid var(--sage);
        border-radius: 0.4rem; padding: 0.4rem 0.6rem; margin: 0.3rem 0; font-size: 0.9rem;
      }

      .stButton > button {
        border-radius: 0.6rem; font-weight: 600; border: 1px solid var(--border);
        background: var(--cream); color: var(--ink); transition: all 0.15s ease;
      }
      .stButton > button:hover { border-color: var(--sage); color: var(--sage-deep); background: var(--cream-2); }
      .stButton > button[kind="primary"] {
        background: linear-gradient(180deg, var(--sage) 0%, var(--sage-deep) 100%);
        color: #fff; border: 1px solid var(--sage-deep);
      }
      .stButton > button[kind="primary"]:hover { background: var(--sage-deep); color: #fff; }
      [data-testid="stDownloadButton"] > button { border-radius: 0.6rem; border: 1px solid var(--border); background: var(--cream); color: var(--ink); font-weight: 600; }
      [data-testid="stDownloadButton"] > button:hover { border-color: var(--teal); color: var(--teal-deep); }

      textarea, [data-baseweb="select"] > div, .stTextArea textarea {
        background: var(--cream) !important; border-radius: 0.6rem !important;
        border-color: var(--border) !important; color: var(--ink) !important;
      }
      [data-testid="stDataFrame"] { border-radius: 0.7rem; overflow: hidden; border: 1px solid var(--border); }
      [data-testid="stNotification"], .stAlert { border-radius: 0.7rem; }
      hr { border-color: var(--border); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
      <div class="mark">🛡️</div>
      <div>
        <div class="title">Helpline Triage — Operator Console</div>
        <div class="sub">Ten Seconds to Read a Call · Hinglish helpline triage · decision-support prototype</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.warning(
    "**Demo only — use fictional or appropriately authorised audio/text.** The ML label and keyword "
    "rules are decision support, not a validated urgency model; they can miss danger or flag harmless "
    "text. A trained operator must verify the transcript and follow local protocol. This tool must "
    "not independently dispatch emergency services.",
    icon="⚠️",
)


# -----------------------------------------------------------------------------
# models2.0: trained ML triage classifier (loaded by file path)
# -----------------------------------------------------------------------------
_MODELS2_DIR = Path(__file__).resolve().parent / "models2.0"
_TRIAGE_SPEC = importlib.util.spec_from_file_location(
    "triage_model_v2", _MODELS2_DIR / "triage_model.py"
)
triage_model_v2 = importlib.util.module_from_spec(_TRIAGE_SPEC)
sys.modules[_TRIAGE_SPEC.name] = triage_model_v2  # so dataclasses/typing resolve the module
try:
    _TRIAGE_SPEC.loader.exec_module(triage_model_v2)  # type: ignore[union-attr]
    _TRIAGE_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - defensive
    _TRIAGE_IMPORT_ERROR = str(exc)


@st.cache_resource(show_spinner="Loading triage model…")
def get_ml_model():
    if _TRIAGE_IMPORT_ERROR is not None:
        raise RuntimeError(_TRIAGE_IMPORT_ERROR)
    return triage_model_v2.load_model()


def ml_predict(text: str) -> dict[str, Any] | None:
    try:
        model = get_ml_model()
        return triage_model_v2.predict_transcript(text, model)
    except Exception:
        return None


ML_LABEL_META = {
    "PRIORITY_REVIEW": {"title": "Priority review", "emoji": "🔴", "badge": "b-priority", "fill": "f-priority"},
    "SUPPORT_REVIEW": {"title": "Support review", "emoji": "🟠", "badge": "b-support", "fill": "f-support"},
    "UNCLEAR_REVIEW": {"title": "Unclear — clarify", "emoji": "⚪", "badge": "b-unclear", "fill": "f-unclear"},
}


def ml_meta(label: str) -> dict[str, Any]:
    return ML_LABEL_META.get(
        label, {"title": label, "emoji": "•", "badge": "b-unclear", "fill": "f-unclear"}
    )


# -----------------------------------------------------------------------------
# Semantic Evidence Voting (SEV) — experimental alternative model
# -----------------------------------------------------------------------------
# NOTE (honest result): on the same untouched held-out split, the Logistic
# Regression baseline scores macro F1 0.939 vs SEV 0.748 (see
# models2.0/evaluation_semantic_vs_logreg.json). The multilingual encoder embeds
# this heavily romanized-Hinglish corpus weakly. LogReg is the default; SEV is
# offered for transparency and comparison, not as an improvement.
METHOD_LOGREG = "Logistic Regression"
METHOD_SEV = "Semantic Evidence Voting"
METHOD_FUSION = "Evidence Fusion"

_SEV_SPEC = importlib.util.spec_from_file_location(
    "sev_v2", _MODELS2_DIR / "semantic_evidence_voting.py"
)
sev_module = importlib.util.module_from_spec(_SEV_SPEC)
sys.modules[_SEV_SPEC.name] = sev_module  # required so @dataclass can resolve its module
try:
    _SEV_SPEC.loader.exec_module(sev_module)  # type: ignore[union-attr]
    _SEV_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - defensive
    _SEV_IMPORT_ERROR = str(exc)

_SEV_INDEX_PATH = _MODELS2_DIR / "sev_index.npz"

# Evidence Fusion module + its validation-tuned config.
_FUSION_SPEC = importlib.util.spec_from_file_location(
    "evidence_fusion_v2", _MODELS2_DIR / "evidence_fusion.py"
)
fusion_module = importlib.util.module_from_spec(_FUSION_SPEC)
sys.modules[_FUSION_SPEC.name] = fusion_module  # required so @dataclass can resolve its module
try:
    _FUSION_SPEC.loader.exec_module(fusion_module)  # type: ignore[union-attr]
    _FUSION_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - defensive
    _FUSION_IMPORT_ERROR = str(exc)

_FUSION_CONFIG_PATH = _MODELS2_DIR / "fusion_config.json"


def get_fusion_config():
    """Load the validation-tuned fusion config (weight + review thresholds)."""
    cfg = fusion_module.FusionConfig()
    try:
        import json
        data = json.loads(_FUSION_CONFIG_PATH.read_text())
        cfg.weight = float(data.get("weight", cfg.weight))
        cfg.low_confidence = float(data.get("low_confidence", cfg.low_confidence))
        cfg.low_margin = float(data.get("low_margin", cfg.low_margin))
    except Exception:
        pass
    return cfg


def sev_available() -> bool:
    """True only if sentence-transformers is importable and the index exists."""
    if _SEV_IMPORT_ERROR is not None or not _SEV_INDEX_PATH.exists():
        return False
    return importlib.util.find_spec("sentence_transformers") is not None


def fusion_available() -> bool:
    """Fusion needs the LogReg model, SEV, and the fusion module."""
    return _FUSION_IMPORT_ERROR is None and sev_available()


@st.cache_resource(show_spinner="Loading Semantic Evidence Voting index…")
def get_sev_voter():
    return sev_module.SemanticEvidenceVoter.load(_SEV_INDEX_PATH)


def predict_with_method(text: str, method: str) -> dict[str, Any] | None:
    """Unified prediction dict for either model (keys are model-agnostic)."""
    if method == METHOD_FUSION:
        try:
            lr = ml_predict(text)
            sev = get_sev_voter().predict_one(text)
            if not lr:
                return None
            fused = fusion_module.fuse(lr["class_scores"], sev["vote_proportions"], get_fusion_config())
            meta_labels = {
                "logreg": fused["logreg_label"], "sev": fused["sev_label"],
            }
            note = (
                f"Fused score = {fused['weight']:.0%} Logistic Regression + "
                f"{1 - fused['weight']:.0%} SEV (weight tuned on validation). "
                "Not a probability of real-world correctness."
            )
            return {
                "method": METHOD_FUSION,
                "predicted_label": fused["predicted_label"],
                "scores": fused["scores"],
                "confidence": fused["confidence"],
                "score_label": "fused score",
                "score_note": note,
                "supporting_examples": sev["supporting_examples"],
                "description": triage_model_v2.LABEL_DESCRIPTIONS.get(fused["predicted_label"], ""),
                "human_review_note": "Verify the transcript, current danger, caller location, and appropriate response with a trained operator.",
                "fusion": {**fused, "component_labels": meta_labels},
            }
        except Exception:
            return None
    if method == METHOD_SEV:
        try:
            r = get_sev_voter().predict_one(text)
            return {
                "method": METHOD_SEV,
                "predicted_label": r["predicted_label"],
                "scores": r["vote_proportions"],
                "confidence": r["confidence"],
                "score_label": "vote proportion",
                "score_note": (
                    f"Vote proportions are the normalised share of similarity mass among the "
                    f"top-{r['neighbours_used']} nearest training examples — not probabilities "
                    f"of real-world correctness."
                ),
                "supporting_examples": r["supporting_examples"],
                "description": triage_model_v2.LABEL_DESCRIPTIONS.get(r["predicted_label"], ""),
                "human_review_note": "Verify the transcript, current danger, caller location, and appropriate response with a trained operator.",
            }
        except Exception:
            return None
    # Default: Logistic Regression baseline.
    r = ml_predict(text)
    if not r:
        return None
    return {
        "method": METHOD_LOGREG,
        "predicted_label": r["predicted_label"],
        "scores": r["class_scores"],
        "confidence": r["model_confidence"],
        "score_label": "model score",
        "score_note": r.get("confidence_note", "Uncalibrated model score; not a probability of real-world correctness."),
        "supporting_examples": None,
        "description": r.get("description", ""),
        "human_review_note": r.get("human_review_note", ""),
    }


def card_header(eyebrow: str, title: str) -> None:
    st.markdown(f'<div class="eyebrow">{eyebrow}</div><div class="card-title">{title}</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sarvam AI speech-to-text (optional audio input)
# -----------------------------------------------------------------------------
def get_sarvam_api_key() -> str | None:
    """Streamlit Community Cloud secrets first; environment variable as fallback."""
    try:
        value = st.secrets.get("SARVAM_API_KEY", None)
        if value:
            return str(value).strip()
    except Exception:
        pass
    value = os.environ.get("SARVAM_API_KEY")
    return value.strip() if value else None


def transcribe_audio(audio_bytes: bytes, filename: str, mode: str) -> tuple[str, str | None]:
    """Transcribe a short audio clip with Sarvam Saaras v4."""
    api_key = get_sarvam_api_key()
    if not api_key:
        raise RuntimeError("Sarvam API key is not configured. Add SARVAM_API_KEY to Streamlit Secrets or the environment.")

    from sarvamai import SarvamAI  # lazy import so the app runs without the package

    suffix = Path(filename).suffix or ".wav"
    temp_path = None
    client = SarvamAI(api_subscription_key=api_key)
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name
        with open(temp_path, "rb") as audio_file:
            response = client.speech_to_text.transcribe(file=audio_file, model="saaras:v4", mode=mode)
        transcript = getattr(response, "transcript", None)
        language_code = getattr(response, "language_code", None)
        if not transcript or not str(transcript).strip():
            raise RuntimeError("Sarvam returned an empty transcript. Try a clearer short clip.")
        return str(transcript).strip(), language_code
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# -----------------------------------------------------------------------------
# Sample data and case records
# -----------------------------------------------------------------------------
SAMPLE_TRANSCRIPTS = {
    "Following near metro": "A man is following me near the metro station. He is threatening me and I am scared right now.",
    "Being followed — help": "I am being followed by a man. Please send help.",
    "Threat at home": "My partner is threatening me at home. I do not feel safe right now.",
    "Repeated harassment": "Someone keeps sending unwanted messages and following me after college.",
    "Information request": "I need information about reporting harassment. There is no immediate danger that I know of.",
    "Hinglish example": "Woh mera peecha kar raha hai aur dhamki de raha hai. Mujhe darr lag raha hai abhi.",
    "Negation test": "He is not following me and he has not threatened me. I am asking how to report an old incident.",
    "Short / ambiguous": "Help please. I am scared.",
}


def make_case(case_id: str, transcript: str, created_at: str, source: str = "Analysed transcript",
              method: str = METHOD_LOGREG) -> dict[str, Any]:
    analysis = analyze_transcript(transcript)
    prediction = predict_with_method(transcript, method)
    if prediction and "predicted_label" in prediction:
        ml_label = prediction["predicted_label"]
        ml_score = f"{prediction['confidence'] * 100:.0f}%"
    else:
        ml_label = "Model unavailable"
        ml_score = "—"
    return {
        "case_id": case_id,
        "transcript": transcript,
        "model": method,
        "ml_label": ml_label,
        "ml_score": ml_score,
        "rule_cue": analysis["urgency"],
        "threat": analysis["threat"],
        "location": ", ".join(analysis["location_clues"]) if analysis["location_clues"] else "Not identified",
        "suggested_queue": analysis["suggested_queue"],
        "created_at": created_at,
        "review_status": "Pending review",
        "operator_notes": "",
        "source": source,
    }


SEED_SAMPLE_KEYS = ["Following near metro", "Being followed — help", "Repeated harassment", "Information request"]


def seed_cases() -> list[dict[str, Any]]:
    # Seeds always use the fast Logistic Regression baseline so startup never
    # depends on the heavy Sentence Transformer download.
    return [
        make_case(f"DEMO-{i:03d}", SAMPLE_TRANSCRIPTS[key], "Fictional example",
                  source="Built-in fictional example", method=METHOD_LOGREG)
        for i, key in enumerate(SEED_SAMPLE_KEYS, start=1)
    ]


for key, default in [
    ("cases", None), ("last_analysis", None), ("last_prediction", None),
    ("last_transcript", ""), ("last_case_id", None), ("case_counter", 1),
    ("last_detected_language", None),
]:
    if key not in st.session_state:
        st.session_state[key] = seed_cases() if key == "cases" else default


# -----------------------------------------------------------------------------
# Related-complaint matching: lexical similarity with a justified threshold
# -----------------------------------------------------------------------------
RELATED_THRESHOLD = 0.30
STRONG_THRESHOLD = 0.55


def _strength_band(score: float) -> str:
    if score >= STRONG_THRESHOLD:
        return "Strong"
    if score >= RELATED_THRESHOLD:
        return "Moderate"
    return "Weak"


def find_related_complaints(
    transcript: str,
    cases: list[dict[str, Any]],
    exclude_case_id: str | None = None,
    top_n: int = 5,
) -> tuple[pd.DataFrame, float]:
    columns = ["Case ID", "Match strength", "Lexical overlap", "Previous transcript", "ML label"]
    candidates = [
        case for case in cases
        if case.get("case_id") != exclude_case_id and str(case.get("transcript", "")).strip()
    ]
    if not transcript.strip() or not candidates:
        return pd.DataFrame(columns=columns), 0.0

    documents = [transcript] + [case["transcript"] for case in candidates]
    try:
        word_matrix = TfidfVectorizer(
            ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True, max_features=30000,
        ).fit_transform(documents)
        char_matrix = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), strip_accents="unicode",
            sublinear_tf=True, max_features=40000,
        ).fit_transform(documents)
        word_sim = cosine_similarity(word_matrix[0:1], word_matrix[1:]).ravel()
        char_sim = cosine_similarity(char_matrix[0:1], char_matrix[1:]).ravel()
        scores = (0.5 * word_sim) + (0.5 * char_sim)
    except ValueError:
        return pd.DataFrame(columns=columns), 0.0

    best = float(scores.max()) if len(scores) else 0.0
    rows = []
    for case, score in zip(candidates, scores):
        s = float(score)
        if s < RELATED_THRESHOLD:
            continue
        rows.append({
            "Case ID": case.get("case_id", "Unknown"),
            "Match strength": _strength_band(s),
            "Lexical overlap": f"{s * 100:.0f}%",
            "_s": s,
            "Previous transcript": case.get("transcript", ""),
            "ML label": ml_meta(case.get("ml_label", "")).get("title", case.get("ml_label", "")),
        })
    rows.sort(key=lambda r: r["_s"], reverse=True)
    frame = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "_s"} for r in rows[:top_n]], columns=columns
    )
    return frame, best


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🛡️ Operator Console")
st.sidebar.caption("Fictional examples only.")

sample_choice = st.sidebar.selectbox(
    "Load an example transcript",
    ["Custom transcript", *SAMPLE_TRANSCRIPTS.keys()],
    key="sample_choice",
)
if st.session_state.get("_last_sample_choice") != sample_choice:
    st.session_state.transcript_input = SAMPLE_TRANSCRIPTS.get(sample_choice, "")
    st.session_state._last_sample_choice = sample_choice

st.sidebar.divider()

# Model selector: Logistic Regression (default), SEV, and Evidence Fusion.
_sev_ok = sev_available()
_fusion_ok = fusion_available()
_method_options = [METHOD_LOGREG]
if _sev_ok:
    _method_options.append(METHOD_SEV)
if _fusion_ok:
    _method_options.append(METHOD_FUSION)
triage_method = st.sidebar.radio(
    "Triage model",
    _method_options,
    index=0,
    help=(
        "Logistic Regression (word+char TF-IDF) is the default. Semantic Evidence Voting "
        "is an experimental multilingual-embedding method. Evidence Fusion blends the two "
        "class-score vectors with a validation-tuned weight and flags disagreement / "
        "low-confidence cases for human review."
    ),
)
if not _sev_ok:
    st.sidebar.caption("Semantic Evidence Voting / Fusion unavailable (needs sentence-transformers + built index).")
st.sidebar.caption(
    "Model metrics are read from saved evaluation reports below; no performance values are hard-coded."
)

if ml_predict("status check") is not None:
    st.sidebar.success("ML triage model: loaded", icon="✅")
else:
    st.sidebar.error("ML model unavailable — keyword rules only.", icon="⚠️")
if get_sarvam_api_key():
    st.sidebar.success("Sarvam API key: configured", icon="🎙️")
else:
    st.sidebar.info("Sarvam key not set — typed transcripts still work.", icon="🎙️")

st.sidebar.markdown("**Pipeline (v2.1)**")
st.sidebar.markdown(
    "- ML classifier · word+char TF-IDF → Logistic Regression (`models2.0`)\n"
    "- Experimental · Semantic Evidence Voting (multilingual embeddings)\n"
    "- Transparent keyword evidence (`triage_rules`)\n"
    "- Optional Sarvam AI speech-to-text\n"
    "- Location & missing-detail flags\n"
    "- Lexical similarity (thresholded)\n"
    "- Session-only case history"
)
st.sidebar.caption(
    "Class scores / vote proportions are uncalibrated model outputs, not probabilities "
    "of real-world correctness."
)


# -----------------------------------------------------------------------------
# 1. Incoming complaint (audio → Sarvam → transcript, or typed)
# -----------------------------------------------------------------------------
def clear_transcript() -> None:
    """Reset input + results. Runs as a callback (before the widget re-instantiates)."""
    st.session_state.transcript_input = ""
    st.session_state.last_analysis = None
    st.session_state.last_prediction = None
    st.session_state.last_transcript = ""
    st.session_state.last_case_id = None
    st.session_state.last_detected_language = None


with st.container(border=True):
    card_header("Step 1", "Incoming complaint")
    st.markdown(
        '<div class="muted">Upload a short audio clip (transcribed via Sarvam AI) or type a transcript directly.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    audio_col, mode_col = st.columns([2, 1])
    with audio_col:
        audio_file = st.file_uploader(
            "Audio recording (short clips, ~30s, for synchronous transcription)",
            type=AUDIO_TYPES,
            help="Use a fictional or specifically authorised clip. Audio is sent to Sarvam AI only when you request transcription.",
        )
    with mode_col:
        sarvam_mode = st.selectbox(
            "Transcription style",
            ["transcribe", "codemix", "translit"],
            index=0,
            help="Try transcribe first; compare codemix/translit on fictional Hinglish examples if needed.",
        )

    consent = st.checkbox(
        "I confirm this is fictional or authorised audio and may be sent to Sarvam AI for transcription.",
        key="audio_consent",
    )

    t_col, _ = st.columns([1.3, 2.4])
    with t_col:
        transcribe_clicked = st.button(
            "🎙️ Transcribe with Sarvam",
            use_container_width=True,
            disabled=audio_file is None,
        )

    if transcribe_clicked:
        if not consent:
            st.error("Please confirm the audio is fictional or authorised before sending it to Sarvam.")
        elif not get_sarvam_api_key():
            st.error("Sarvam API key is missing. Configure SARVAM_API_KEY in Streamlit Secrets or the environment.")
        elif audio_file is not None:
            try:
                with st.spinner("Transcribing audio with Sarvam…"):
                    transcript_result, detected_language = transcribe_audio(
                        audio_file.getvalue(), audio_file.name, sarvam_mode
                    )
                st.session_state.transcript_input = transcript_result
                st.session_state.last_detected_language = detected_language
                st.success("Transcription complete. Review and correct the text below before analysis.")
                st.rerun()
            except Exception as exc:
                st.error(f"Transcription failed: {type(exc).__name__}: {exc}")
                st.info("You can still enter or paste a transcript manually below.")

    if st.session_state.get("last_detected_language"):
        st.caption(f"Sarvam detected language: {st.session_state.last_detected_language}")

    transcript = st.text_area(
        "Call transcript — review before analysis",
        key="transcript_input",
        height=140,
        placeholder="Enter a fictional transcript in English, Hindi, or Romanized Hinglish…",
        help="Original text is kept for operator review. ML classifies routing; keyword rules add transparent evidence.",
        label_visibility="collapsed",
    )
    col_analyse, col_clear, _ = st.columns([1.1, 1, 2.4])
    with col_analyse:
        analyze_clicked = st.button("🔍 Analyse transcript", type="primary", use_container_width=True)
    with col_clear:
        st.button("Clear", use_container_width=True, on_click=clear_transcript)

if analyze_clicked:
    if not transcript.strip():
        st.warning("Enter a fictional transcript before analysing.")
    else:
        result = analyze_transcript(transcript)
        with st.spinner(f"Analysing with {triage_method}…"):
            prediction = predict_with_method(transcript.strip(), triage_method)
        case_id = f"HL-{st.session_state.case_counter:04d}"
        st.session_state.case_counter += 1
        created_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        st.session_state.cases.append(
            make_case(case_id, transcript.strip(), created_at, method=triage_method)
        )
        st.session_state.last_analysis = result
        st.session_state.last_prediction = prediction
        st.session_state.last_transcript = transcript.strip()
        st.session_state.last_case_id = case_id


# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
def render_score_bars(class_scores: dict[str, float]) -> None:
    rows_html = []
    for label in ["PRIORITY_REVIEW", "SUPPORT_REVIEW", "UNCLEAR_REVIEW"]:
        if label not in class_scores:
            continue
        meta = ml_meta(label)
        pct = class_scores[label] * 100
        rows_html.append(
            f'<div class="score-row"><div class="score-label">'
            f'<span>{meta["emoji"]} {meta["title"]}</span><span>{pct:.0f}%</span></div>'
            f'<div class="score-track"><div class="score-fill {meta["fill"]}" '
            f'style="width:{pct:.0f}%"></div></div></div>'
        )
    st.markdown("".join(rows_html), unsafe_allow_html=True)


if st.session_state.last_analysis is not None:
    result = st.session_state.last_analysis
    prediction = st.session_state.last_prediction
    case_id = st.session_state.last_case_id
    current_case = next((c for c in st.session_state.cases if c["case_id"] == case_id), None)
    review_status = current_case.get("review_status", "Pending review") if current_case else "Pending review"

    method_used = prediction.get("method", METHOD_LOGREG) if prediction else METHOD_LOGREG
    score_label = prediction.get("score_label", "model score") if prediction else "model score"
    with st.container(border=True):
        card_header("Step 2", f"Triage result · case {case_id}")
        st.markdown(
            f'<div class="muted">Model: <b>{method_used}</b> · routing label with rule-based evidence · '
            f'uncalibrated {score_label}, not a validated risk score</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        head_l, head_r = st.columns([1.5, 1])
        with head_l:
            if prediction and "predicted_label" in prediction:
                meta = ml_meta(prediction["predicted_label"])
                conf = prediction["confidence"] * 100
                st.markdown(
                    f'<span class="triage-badge {meta["badge"]}">{meta["emoji"]} {meta["title"]}</span>'
                    f'<span class="muted" style="margin-left:0.6rem;">{score_label} {conf:.0f}%</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'<div class="muted" style="margin-top:0.45rem;">{prediction.get("description","")}</div>', unsafe_allow_html=True)
                if result["urgency"] == "Critical" and prediction["predicted_label"] != "PRIORITY_REVIEW":
                    st.error(
                        "⚠️ Keyword rules flagged possible CRITICAL indicators while the ML label is "
                        f"'{meta['title']}'. Treat as priority and verify the transcript immediately."
                    )
            else:
                st.error("Selected model unavailable — showing keyword-rule evidence only.")
        with head_r:
            st.markdown(
                '<div class="eyebrow">Operator review</div>'
                f'<span class="triage-badge b-neutral badge-sm">📋 {review_status}</span>',
                unsafe_allow_html=True,
            )

        # Evidence Fusion: show component agreement and the human-review flag.
        if prediction and prediction.get("fusion"):
            fz = prediction["fusion"]
            st.write("")
            lr_m, sev_m = ml_meta(fz["logreg_label"]), ml_meta(fz["sev_label"])
            agree_badge = ("b-unclear", "✅ Models agree") if fz["models_agree"] else ("b-support", "⚠️ Models disagree")
            st.markdown(
                '<div class="eyebrow">Component models</div>'
                f'<div class="kv"><span class="chip">Logistic Regression → {lr_m["emoji"]} {lr_m["title"]}</span>'
                f'<span class="chip">SEV → {sev_m["emoji"]} {sev_m["title"]}</span>'
                f'<span class="triage-badge {agree_badge[0]} badge-sm">{agree_badge[1]}</span></div>',
                unsafe_allow_html=True,
            )
            if fz["review_flag"]:
                st.warning("🚩 Flagged for human review: " + " ".join(fz["review_reasons"]))
            else:
                st.caption("No fusion review flag raised (models agree and the fused score is confident). Human verification is still required.")

        if prediction and prediction.get("scores"):
            st.write("")
            if method_used == METHOD_SEV:
                bars_label = "Vote proportions"
            elif method_used == METHOD_FUSION:
                bars_label = "Fused class scores"
            else:
                bars_label = "Model class scores (uncalibrated)"
            st.markdown(f'<div class="eyebrow">{bars_label}</div>', unsafe_allow_html=True)
            render_score_bars(prediction["scores"])
            st.caption(prediction.get("score_note", "Uncalibrated output; not a probability of real-world correctness."))

        # Semantic Evidence Voting / Fusion: show the nearest training examples transparently.
        if prediction and prediction.get("supporting_examples"):
            st.write("")
            st.markdown('<div class="eyebrow">Supporting training examples (nearest neighbours)</div>', unsafe_allow_html=True)
            for ex in prediction["supporting_examples"]:
                ex_meta = ml_meta(ex["label"])
                snippet = ex["transcript"] if len(ex["transcript"]) <= 110 else ex["transcript"][:107] + "…"
                st.markdown(
                    f'<div class="kv"><span class="triage-badge {ex_meta["badge"]} badge-sm">{ex_meta["emoji"]} {ex_meta["title"]}</span>'
                    f'<span class="muted" style="margin-left:0.5rem;">similarity {ex["similarity"]*100:.0f}%</span><br>'
                    f'<span style="font-size:0.88rem;">“{snippet}”</span></div>',
                    unsafe_allow_html=True,
                )
            st.caption("These are the labelled training transcripts most similar to this input. Similarity is not proof of a shared caller or incident; a human must verify.")

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            card_header("Evidence", "🔎 Detected cues (rules)")
            st.markdown(f'<div class="kv"><b>Possible concern:</b> {result["threat"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kv"><b>Suggested queue:</b> {result["suggested_queue"]}</div>', unsafe_allow_html=True)
            loc = ", ".join(result["location_clues"]) if result["location_clues"] else "Not identified"
            st.markdown(f'<div class="kv"><b>Location clue(s):</b> {loc}</div>', unsafe_allow_html=True)
            st.caption("A location clue is unverified text, not a confirmed current location or address.")
            st.markdown('<div class="eyebrow" style="margin-top:0.4rem;">Matched text indicators</div>', unsafe_allow_html=True)
            if result["evidence"]:
                st.markdown("".join(f'<span class="chip chip-sage">{p}</span>' for p in result["evidence"]), unsafe_allow_html=True)
            else:
                st.markdown('<span class="muted">No configured indicator phrase matched.</span>', unsafe_allow_html=True)
            st.caption("A matched phrase is a review cue, not proof that an event occurred.")
    with c2:
        with st.container(border=True):
            card_header("Verify", "✅ Information to confirm")
            for item in result["missing"]:
                st.markdown(f'<div class="verify-item">{item}</div>', unsafe_allow_html=True)
            st.markdown('<div class="eyebrow" style="margin-top:0.4rem;">Suggested operator next step</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kv">{result["next_step"]}</div>', unsafe_allow_html=True)
            if prediction and "human_review_note" in prediction:
                st.caption(prediction["human_review_note"])

    with st.container(border=True):
        card_header("Step 3", "Operator review")
        st.markdown(
            '<div class="muted">Distinguish model/rule output above from operator-confirmed findings below. '
            'Saving does not contact services or dispatch help.</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        if current_case is not None:
            status_key = f"review_status_{case_id}"
            notes_key = f"review_notes_{case_id}"
            st.session_state.setdefault(status_key, current_case.get("review_status", "Pending review"))
            st.session_state.setdefault(notes_key, current_case.get("operator_notes", ""))
            rc1, rc2 = st.columns([1, 1.4])
            with rc1:
                status = st.selectbox(
                    "Operator-confirmed outcome",
                    [
                        "Pending review",
                        "Reviewed — more information needed",
                        "Reviewed — follow-up required",
                        "Reviewed — no further action recorded",
                        "Escalated according to local protocol",
                    ],
                    key=status_key,
                )
            with rc2:
                notes = st.text_area(
                    "Operator notes (fictional demo only)",
                    key=notes_key,
                    placeholder="Record what was verified, without adding real personal data.",
                    height=90,
                )
            if st.button("💾 Save operator review", key=f"save_review_{case_id}"):
                for case in st.session_state.cases:
                    if case["case_id"] == case_id:
                        case["review_status"] = status
                        case["operator_notes"] = notes.strip()
                        break
                st.success("Operator review saved to this browser session.")

    with st.container(border=True):
        card_header("Step 4", "Related transcripts")
        matches, best = find_related_complaints(
            st.session_state.last_transcript, st.session_state.cases, exclude_case_id=case_id
        )
        if matches.empty:
            st.info(
                f"No strong matches found (highest lexical overlap was {best * 100:.0f}%, "
                f"below the {RELATED_THRESHOLD * 100:.0f}% threshold for a related candidate).",
                icon="🔗",
            )
        else:
            st.dataframe(matches, use_container_width=True, hide_index=True)
        st.caption(
            "Lexical overlap is a rough word/character comparison, not proof of a shared caller, "
            "person, or incident. Any potential link must be checked by an authorised human."
        )


# -----------------------------------------------------------------------------
# Model performance: read saved evaluation artifacts; never invent metrics.
# Expected canonical format: models2.0/model_metrics.json with a "models" map.
# The loader also attempts to read the existing LogReg-vs-SEV evaluation file.
# -----------------------------------------------------------------------------
_METRIC_FILE_CANDIDATES = [
    _MODELS2_DIR / "model_metrics.json",
    _MODELS2_DIR / "evaluation_evidence_fusion.json",
    _MODELS2_DIR / "evaluation_fusion.json",
    _MODELS2_DIR / "evaluation_semantic_vs_logreg.json",
]

_MODEL_ALIASES = {
    "logistic regression": METHOD_LOGREG,
    "logreg": METHOD_LOGREG,
    "lr": METHOD_LOGREG,
    "semantic evidence voting": METHOD_SEV,
    "semantic voting": METHOD_SEV,
    "sev": METHOD_SEV,
    "semantic": METHOD_SEV,
    "evidence fusion": METHOD_FUSION,
    "fusion": METHOD_FUSION,
}
_METRIC_ALIASES = {
    "accuracy": ("accuracy", "test_accuracy", "acc"),
    "macro_f1": ("macro_f1", "f1_macro", "macro avg f1-score", "f1_macro_score"),
    "macro_precision": ("macro_precision", "precision_macro", "macro avg precision", "precision_macro_score"),
    "macro_recall": ("macro_recall", "recall_macro", "macro avg recall", "recall_macro_score"),
    "priority_recall": ("priority_recall", "priority_review_recall", "recall_priority", "priority_class_recall"),
    "test_samples": ("test_samples", "n_test", "test_size", "support_total"),
}


def _canonical_model_name(name: str) -> str | None:
    key = str(name).strip().lower().replace("_", " ").replace("-", " ")
    return _MODEL_ALIASES.get(key)


def _first_numeric(mapping: dict, aliases: tuple[str, ...]):
    lowered = {str(k).strip().lower().replace("_", " ").replace("-", " "): v for k, v in mapping.items()}
    for alias in aliases:
        key = alias.lower().replace("_", " ").replace("-", " ")
        if key in lowered:
            try:
                value = float(lowered[key])
                if pd.notna(value):
                    return value
            except (TypeError, ValueError):
                continue
    return None


def _extract_metrics_from_model_blob(blob: dict) -> dict[str, Any]:
    """Extract common metric names from a report without fabricating missing values."""
    result: dict[str, Any] = {}
    for metric, aliases in _METRIC_ALIASES.items():
        value = _first_numeric(blob, aliases)
        if value is None:
            # Common sklearn classification_report shape: {"macro avg": {"f1-score": ...}}
            macro = blob.get("macro avg") or blob.get("macro_avg") or blob.get("macro average")
            if isinstance(macro, dict):
                nested_aliases = {
                    "macro_f1": ("f1-score", "f1 score", "f1"),
                    "macro_precision": ("precision",),
                    "macro_recall": ("recall",),
                }.get(metric, ())
                if nested_aliases:
                    value = _first_numeric(macro, nested_aliases)
            if value is None and metric == "priority_recall":
                for label_key in ("PRIORITY_REVIEW", "priority_review", "Priority review"):
                    class_metrics = blob.get(label_key)
                    if isinstance(class_metrics, dict):
                        value = _first_numeric(class_metrics, ("recall",))
                        if value is not None:
                            break
        if value is not None:
            result[metric] = value

    # Accept common sklearn output keys.
    cm = blob.get("confusion_matrix", blob.get("confusion matrix"))
    labels = blob.get("labels", blob.get("class_labels"))
    if isinstance(cm, list) and cm and all(isinstance(row, (list, tuple)) for row in cm):
        result["confusion_matrix"] = cm
        if isinstance(labels, list):
            result["labels"] = labels
    return result


def _collect_model_reports(payload: Any, collected: dict[str, dict[str, Any]]) -> None:
    """Recursively find model-named dictionaries in common evaluation JSON layouts."""
    if not isinstance(payload, dict):
        return
    for key, value in payload.items():
        model_name = _canonical_model_name(str(key))
        if model_name and isinstance(value, dict):
            metrics = _extract_metrics_from_model_blob(value)
            # Some reports nest metrics under a "metrics" or "test" object.
            for nested_key in ("metrics", "test", "test_metrics", "evaluation", "results"):
                nested = value.get(nested_key)
                if isinstance(nested, dict):
                    metrics.update({k: v for k, v in _extract_metrics_from_model_blob(nested).items() if k not in metrics})
            if metrics:
                collected[model_name] = {**collected.get(model_name, {}), **metrics}
        _collect_model_reports(value, collected)


def load_model_metrics() -> tuple[dict[str, dict[str, Any]], list[str], list[dict[str, Any]]]:
    reports: dict[str, dict[str, Any]] = {}
    loaded_files: list[str] = []
    raw_payloads: list[dict[str, Any]] = []
    for path in _METRIC_FILE_CANDIDATES:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        loaded_files.append(path.name)
        raw_payloads.append({"file": path.name, "payload": payload})
        # Preferred canonical structure: {"models": {"Logistic Regression": {...}, ...}}
        model_map = payload.get("models") if isinstance(payload, dict) else None
        if isinstance(model_map, dict):
            _collect_model_reports(model_map, reports)
        _collect_model_reports(payload, reports)
    return reports, loaded_files, raw_payloads


def _format_metric(value: Any, metric: str) -> str:
    if value is None:
        return "—"
    if metric == "test_samples":
        return str(int(value))
    # Metrics from JSON may be 0–1 fractions or 0–100 percentages.
    numeric = float(value)
    if metric != "test_samples" and 0 <= numeric <= 1:
        return f"{numeric:.3f} ({numeric * 100:.1f}%)"
    return f"{numeric:.3f}" if metric != "test_samples" else str(int(numeric))


with st.container(border=True):
    card_header("Model evaluation", "📊 Model performance & comparison")
    st.markdown(
        '<div class="muted">Held-out evaluation results for Logistic Regression, Semantic Evidence Voting, and Evidence Fusion. '
        'These metrics describe the labelled evaluation dataset, not real-world emergency performance.</div>',
        unsafe_allow_html=True,
    )
    metric_reports, metric_files, raw_metric_payloads = load_model_metrics()
    if not metric_reports:
        st.info(
            "No readable model-comparison metrics were found. Add a saved evaluation report under "
            "`models2.0/model_metrics.json` (format shown below) after running evaluation. "
            "The dashboard intentionally does not display placeholder scores."
        )
        with st.expander("Expected evaluation JSON format"):
            st.code(json.dumps({
                "evaluation_split": "held-out test set",
                "test_samples": None,
                "models": {
                    "Logistic Regression": {"accuracy": None, "macro_f1": None, "macro_precision": None, "macro_recall": None, "priority_recall": None},
                    "Semantic Evidence Voting": {"accuracy": None, "macro_f1": None, "macro_precision": None, "macro_recall": None, "priority_recall": None},
                    "Evidence Fusion": {"accuracy": None, "macro_f1": None, "macro_precision": None, "macro_recall": None, "priority_recall": None}
                }
            }, indent=2), language="json")
            st.caption("Template only. Replace null values with metrics measured on the same held-out test set; never enter illustrative scores.")
    else:
        st.caption("Loaded evaluation file(s): " + ", ".join(metric_files))
        display_names = [METHOD_LOGREG, METHOD_SEV, METHOD_FUSION]
        metric_order = ["accuracy", "macro_f1", "macro_precision", "macro_recall", "priority_recall", "test_samples"]
        display_labels = {
            "accuracy": "Accuracy", "macro_f1": "Macro-F1", "macro_precision": "Macro precision",
            "macro_recall": "Macro recall", "priority_recall": "Priority recall", "test_samples": "Test samples",
        }
        table_rows = []
        for model_name in display_names:
            metrics = metric_reports.get(model_name)
            if not metrics:
                continue
            row = {"Model": model_name}
            for metric in metric_order:
                row[display_labels[metric]] = _format_metric(metrics.get(metric), metric)
            table_rows.append(row)
        if table_rows:
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
            f1_candidates = [(name, vals.get("macro_f1")) for name, vals in metric_reports.items() if vals.get("macro_f1") is not None]
            priority_candidates = [(name, vals.get("priority_recall")) for name, vals in metric_reports.items() if vals.get("priority_recall") is not None]
            insights = []
            if f1_candidates:
                best_f1 = max(f1_candidates, key=lambda item: item[1])
                insights.append(f"Highest reported Macro-F1: **{best_f1[0]}** ({_format_metric(best_f1[1], 'macro_f1')}).")
            if priority_candidates:
                best_priority = max(priority_candidates, key=lambda item: item[1])
                insights.append(f"Highest reported Priority Recall: **{best_priority[0]}** ({_format_metric(best_priority[1], 'priority_recall')}).")
            if insights:
                st.markdown("\n\n".join(insights))
            if len(metric_reports) > 1:
                st.caption("There is no single universally best model: consider Macro-F1, priority-class recall, and the cost of missed cases together. Highest metrics do not establish real-world safety.")
        else:
            st.warning("Evaluation JSON files were found, but their metric structure was not recognised. See raw reports below and adapt the evaluation output to the expected schema.")
        with st.expander("Evaluation report details"):
            for report in raw_metric_payloads:
                st.markdown(f"**{report['file']}**")
                st.json(report["payload"])
        # Show a confusion matrix only when it is actually present in a report.
        matrix_found = False
        for model_name in display_names:
            metrics = metric_reports.get(model_name, {})
            cm = metrics.get("confusion_matrix")
            if cm:
                matrix_found = True
                labels = metrics.get("labels") or ["Class 1", "Class 2", "Class 3"]
                st.markdown(f"**{model_name} — confusion matrix**")
                try:
                    st.dataframe(pd.DataFrame(cm, index=labels, columns=labels), use_container_width=True)
                except (ValueError, TypeError):
                    st.write(cm)
        if not matrix_found:
            st.caption("Confusion matrices will appear here when included in the saved evaluation report.")
    st.caption(
        "For a fair comparison, evaluate all three methods on the same held-out test examples. "
        "Tune fusion weights on validation data only; keep the test set out of semantic retrieval and tuning. "
        "Scores shown here are evaluation metrics, not per-call confidence scores."
    )


# -----------------------------------------------------------------------------
# 5. Emergency assistance — nearby resources + conservative safety cues
# -----------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def _geocode_place(query: str) -> dict[str, Any] | None:
    """Resolve an operator-entered place using OpenStreetMap Nominatim."""
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "jsonv2", "limit": 1},
        headers={"User-Agent": "HelplineTriageDemo/1.0 (emergency-resource lookup)"},
        timeout=15,
    )
    response.raise_for_status()
    rows = response.json()
    if not rows:
        return None
    return {"lat": float(rows[0]["lat"]), "lon": float(rows[0]["lon"]), "display_name": rows[0].get("display_name", query)}


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometres."""
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))


@st.cache_data(ttl=600, show_spinner=False)
def _nearby_emergency_resources(lat: float, lon: float, radius_m: int) -> list[dict[str, Any]]:
    """Query OpenStreetMap via Overpass; results are map listings, not availability."""
    query = f"""[out:json][timeout:25];(
      node["amenity"="police"](around:{radius_m},{lat},{lon});
      way["amenity"="police"](around:{radius_m},{lat},{lon});
      relation["amenity"="police"](around:{radius_m},{lat},{lon});
      node["amenity"="hospital"](around:{radius_m},{lat},{lon});
      way["amenity"="hospital"](around:{radius_m},{lat},{lon});
      relation["amenity"="hospital"](around:{radius_m},{lat},{lon});
      node["amenity"="clinic"](around:{radius_m},{lat},{lon});
      way["amenity"="clinic"](around:{radius_m},{lat},{lon});
      relation["amenity"="clinic"](around:{radius_m},{lat},{lon});
      node["emergency"="ambulance_station"](around:{radius_m},{lat},{lon});
      way["emergency"="ambulance_station"](around:{radius_m},{lat},{lon});
      relation["emergency"="ambulance_station"](around:{radius_m},{lat},{lon});
    );out center tags;"""
    response = requests.post(
        "https://overpass-api.de/api/interpreter", data={"data": query},
        headers={"User-Agent": "HelplineTriageDemo/1.0"}, timeout=35,
    )
    response.raise_for_status()
    elements = response.json().get("elements", [])
    results: list[dict[str, Any]] = []
    seen = set()
    for el in elements:
        tags = el.get("tags", {})
        if "lat" in el and "lon" in el:
            rlat, rlon = float(el["lat"]), float(el["lon"])
        elif el.get("center"):
            rlat, rlon = float(el["center"]["lat"]), float(el["center"]["lon"])
        else:
            continue
        amenity = tags.get("amenity", "")
        emergency = tags.get("emergency", "")
        if amenity == "police":
            kind = "Police station"
        elif amenity == "hospital":
            kind = "Hospital"
        elif amenity == "clinic":
            kind = "Clinic"
        elif emergency == "ambulance_station":
            kind = "Ambulance station"
        else:
            continue
        name = tags.get("name") or tags.get("operator") or kind
        dedupe = (kind, name.lower(), round(rlat, 4), round(rlon, 4))
        if dedupe in seen:
            continue
        seen.add(dedupe)
        results.append({
            "Resource": name, "Type": kind,
            "Distance (km)": round(_distance_km(lat, lon, rlat, rlon), 2),
            "Address": tags.get("addr:full") or ", ".join(x for x in [tags.get("addr:street"), tags.get("addr:city")] if x) or "Address not listed",
            "lat": rlat, "lon": rlon,
            "Phone": tags.get("phone") or tags.get("contact:phone") or "Not listed",
            "Website": tags.get("website") or tags.get("contact:website") or "",
        })
    return sorted(results, key=lambda row: row["Distance (km)"])


with st.container(border=True):
    card_header("Step 5", "🚨 Emergency assistance & nearby resources")
    st.markdown(
        '<div class="muted">Find mapped police stations, hospitals, clinics and ambulance stations near an operator-entered location.</div>',
        unsafe_allow_html=True,
    )
    st.warning(
        "Demo resource finder, not an emergency dispatch service. OpenStreetMap listings may be missing, outdated or incomplete; "
        "a listed ambulance station does not confirm that an ambulance is available. In an active emergency, follow local emergency protocol immediately.",
        icon="⚠️",
    )
    loc_col, radius_col = st.columns([3, 1])
    with loc_col:
        emergency_location = st.text_input(
            "Caller location (enter a nearby landmark, address or locality)",
            placeholder="e.g., Connaught Place, New Delhi",
            key="emergency_location_query",
            help="Location is sent to OpenStreetMap only when you press Search. Use fictional/demo locations unless you have authorization.",
        )
    with radius_col:
        radius_km = st.selectbox("Search radius", [2, 5, 10, 20], index=1, format_func=lambda x: f"{x} km", key="emergency_radius_km")
    search_resources = st.button("📍 Find nearby emergency resources", type="primary", use_container_width=True)
    if search_resources:
        if not emergency_location.strip():
            st.error("Enter a location or landmark first.")
        else:
            try:
                with st.spinner("Resolving location and searching mapped emergency resources…"):
                    geocoded = _geocode_place(emergency_location.strip())
                    if not geocoded:
                        st.session_state.emergency_lookup_result = {"error": "Location not found. Try a more specific address or landmark."}
                    else:
                        places = _nearby_emergency_resources(geocoded["lat"], geocoded["lon"], int(radius_km * 1000))
                        st.session_state.emergency_lookup_result = {"location": geocoded, "places": places, "radius_km": radius_km}
            except requests.RequestException as exc:
                st.session_state.emergency_lookup_result = {"error": f"The map service could not be reached ({type(exc).__name__}). Please retry later or use an official local directory."}
            except (ValueError, KeyError) as exc:
                st.session_state.emergency_lookup_result = {"error": f"Could not process the map response ({type(exc).__name__}). Please retry later."}
    lookup = st.session_state.get("emergency_lookup_result")
    if lookup:
        if lookup.get("error"):
            st.error(lookup["error"])
        else:
            resolved = lookup["location"]
            places = lookup["places"]
            st.caption(f"Resolved location: {resolved['display_name']} · Search radius: {lookup['radius_km']} km")
            if not places:
                st.info("No matching resources were found in the mapped data within this radius. This does not mean no services exist; verify using official local sources.")
            else:
                resource_df = pd.DataFrame(places)
                type_counts = resource_df["Type"].value_counts().to_dict()
                count_cols = st.columns(4)
                for idx, label in enumerate(["Police station", "Hospital", "Clinic", "Ambulance station"]):
                    count_cols[idx].metric(label, int(type_counts.get(label, 0)))
                map_df = pd.DataFrame([{"lat": resolved["lat"], "lon": resolved["lon"], "Map point": "Entered location"}] + [
                    {"lat": item["lat"], "lon": item["lon"], "Map point": item["Resource"]} for item in places
                ])
                st.map(map_df, latitude="lat", longitude="lon", size=90, color=None, use_container_width=True)
                display_df = resource_df[["Resource", "Type", "Distance (km)", "Address", "Phone"]].copy()
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                st.markdown("**Directions and contact links**")
                for item in places[:12]:
                    directions_url = f"https://www.google.com/maps/dir/?api=1&destination={item['lat']},{item['lon']}"
                    maps_url = f"https://www.openstreetmap.org/?mlat={item['lat']}&mlon={item['lon']}#map=17/{item['lat']}/{item['lon']}"
                    phone_text = f" · Phone: {item['Phone']}" if item["Phone"] != "Not listed" else " · Phone not listed"
                    st.markdown(
                        f"- **{item['Resource']}** — {item['Type']}, {item['Distance (km)']} km{phone_text} · "
                        f"[Directions](<{directions_url}>) · [OpenStreetMap](<{maps_url}>)"
                    )
                st.caption("Data source: OpenStreetMap contributors via Nominatim and Overpass. Distances are straight-line estimates, not travel times. Confirm the facility and phone number independently before relying on them.")

    # Simple, transparent safety cues from the latest transcript; not a risk classifier.
    latest_transcript = str(st.session_state.get("last_transcript") or st.session_state.get("transcript_input") or "").strip()
    if latest_transcript:
        text_lower = latest_transcript.lower()
        safety_cues = {
            "Possible immediate threat / violence": ["he is here", "he's here", "attacking me", "trying to kill", "has a weapon", "gun", "knife", "following me right now", "help me now", "being attacked", "mujhe maar", "maar raha", "ghar mein ghus", "peecha kar raha"],
            "Possible serious medical emergency": ["not breathing", "can't breathe", "cannot breathe", "unconscious", "not responding", "severe bleeding", "heavy bleeding", "chest pain", "collapsed", "saans nahi", "behosh", "khoon beh", "saans nahin aa"],
        }
        matched_cues = [(category, [phrase for phrase in phrases if phrase in text_lower]) for category, phrases in safety_cues.items()]
        matched_cues = [(category, phrases) for category, phrases in matched_cues if phrases]
        st.markdown("**Transcript safety-check cues (experimental)**")
        if matched_cues:
            for category, phrases in matched_cues:
                st.error(f"Review promptly: {category}. Matched phrase(s): {', '.join(phrases)}")
            st.caption("Keyword checks can miss emergencies and can flag harmless or quoted text. A trained operator must verify context and follow local protocol.")
        else:
            st.info("No configured safety phrase matched the latest transcript. This is not evidence that the situation is safe.")
    else:
        st.caption("Analyse a fictional/authorized transcript above to run the experimental transcript safety checks.")


# -----------------------------------------------------------------------------
# 6. Session case history
# -----------------------------------------------------------------------------
with st.container(border=True):
    card_header("Step 6", "Session case history")
    st.markdown('<div class="muted">Kept in Streamlit session state only — not a permanent database.</div>', unsafe_allow_html=True)
    st.write("")
    if st.session_state.cases:
        history = pd.DataFrame(st.session_state.cases)
        display_columns = [
            "case_id", "model", "ml_label", "ml_score", "rule_cue", "threat", "location",
            "suggested_queue", "review_status", "created_at", "source",
        ]
        export_columns = [
            "case_id", "transcript", "model", "ml_label", "ml_score", "rule_cue", "threat", "location",
            "suggested_queue", "created_at", "review_status", "operator_notes", "source",
        ]
        for col in set(display_columns + export_columns):
            if col not in history.columns:
                history[col] = ""
        st.dataframe(history[display_columns], use_container_width=True, hide_index=True)

        csv_bytes = history[export_columns].to_csv(index=False).encode("utf-8-sig")
        hc1, hc2 = st.columns([1, 1])
        with hc1:
            st.download_button(
                "⬇️ Download session history (CSV)", data=csv_bytes,
                file_name="helpline_triage_session_history.csv", mime="text/csv",
                use_container_width=True,
            )
        with hc2:
            if st.button("↺ Reset to built-in examples", use_container_width=True):
                st.session_state.cases = seed_cases()
                st.session_state.last_analysis = None
                st.session_state.last_prediction = None
                st.session_state.last_transcript = ""
                st.session_state.last_case_id = None
                st.session_state.last_detected_language = None
                st.session_state.case_counter = 1
                st.rerun()
    else:
        st.info("No cases recorded in this session yet.")

st.caption(
    "Prototype v2.3 — Logistic Regression, optional Semantic Evidence Voting and Evidence Fusion, "
    "transparent keyword-rule evidence, optional Sarvam AI speech-to-text, emergency-resource lookup, location clues, and temporary "
    "session history. This is a decision-support prototype using synthetic/demo data; model scores are "
    "not validated real-world risk probabilities and must be reviewed by a trained human."
)
