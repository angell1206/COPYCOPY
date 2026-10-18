"""AI-Based Help Line Triage — Hinglish helpline transcript demo (v2.1).

Operator-support dashboard that pairs a trained ML triage classifier
(``models2.0/``) with a transparent, rule-based evidence layer
(``triage_rules.py``). The ML model gives the headline routing label with an
uncalibrated model score; the keyword rules give human-readable evidence,
location clues, and information still to verify. They are kept separate so an
operator can cross-check them.

Fictional/demo transcripts only. This is decision support, not a validated
emergency-risk model, and it must never dispatch services.
"""
from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from triage_rules import analyze_transcript

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

      /* Header band */
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

      /* Elevated cards = st.container(border=True) */
      [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--cream); border: 1px solid var(--border) !important;
        border-radius: 0.9rem; box-shadow: var(--shadow);
        padding: 0.3rem 0.25rem;
      }
      /* avoid double-card look for nested columns' inner wrappers */
      [data-testid="column"] [data-testid="stVerticalBlockBorderWrapper"] { box-shadow: var(--shadow-sm); }

      .eyebrow {
        display: inline-block; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: var(--sage-deep); margin-bottom: 0.1rem;
      }
      .card-title { font-size: 1.05rem; font-weight: 700; color: var(--ink); margin: 0 0 0.3rem; }

      /* Status badges */
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

      /* Score bars */
      .score-row { margin: 0.4rem 0; }
      .score-label { font-size: 0.86rem; color: var(--ink); display: flex; justify-content: space-between; margin-bottom: 0.22rem; font-weight: 600; }
      .score-track { background: var(--champagne); border-radius: 7px; height: 0.55rem; overflow: hidden; }
      .score-fill { height: 100%; border-radius: 7px; }
      .f-priority { background: linear-gradient(90deg, #c07558, var(--priority)); }
      .f-support  { background: linear-gradient(90deg, #c2a05a, var(--support)); }
      .f-unclear  { background: linear-gradient(90deg, #8aa79c, var(--unclear)); }

      /* Chips */
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

      /* Buttons */
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

      /* Inputs */
      textarea, [data-baseweb="select"] > div, .stTextArea textarea {
        background: var(--cream) !important; border-radius: 0.6rem !important;
        border-color: var(--border) !important; color: var(--ink) !important;
      }
      /* Dataframe + alerts soften */
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
    "**Demo only — use fictional transcripts.** The ML label and keyword rules are decision "
    "support, not a validated urgency model; they can miss danger or flag harmless text. A "
    "trained operator must verify the transcript and follow local protocol. This tool must "
    "not independently dispatch emergency services.",
    icon="⚠️",
)


# -----------------------------------------------------------------------------
# models2.0: trained ML triage classifier (loaded by file path)  [unchanged logic]
# -----------------------------------------------------------------------------
_MODELS2_DIR = Path(__file__).resolve().parent / "models2.0"
_TRIAGE_SPEC = importlib.util.spec_from_file_location(
    "triage_model_v2", _MODELS2_DIR / "triage_model.py"
)
triage_model_v2 = importlib.util.module_from_spec(_TRIAGE_SPEC)
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


def card_header(eyebrow: str, title: str) -> None:
    st.markdown(f'<div class="eyebrow">{eyebrow}</div><div class="card-title">{title}</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sample data and case records  [unchanged logic]
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


def make_case(case_id: str, transcript: str, created_at: str, source: str = "Analysed transcript") -> dict[str, Any]:
    analysis = analyze_transcript(transcript)
    prediction = ml_predict(transcript)
    if prediction and "predicted_label" in prediction:
        ml_label = prediction["predicted_label"]
        ml_score = f"{prediction['model_confidence'] * 100:.0f}%"
    else:
        ml_label = "Model unavailable"
        ml_score = "—"
    return {
        "case_id": case_id,
        "transcript": transcript,
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
    return [
        make_case(f"DEMO-{i:03d}", SAMPLE_TRANSCRIPTS[key], "Fictional example", source="Built-in fictional example")
        for i, key in enumerate(SEED_SAMPLE_KEYS, start=1)
    ]


for key, default in [
    ("cases", None), ("last_analysis", None), ("last_prediction", None),
    ("last_transcript", ""), ("last_case_id", None), ("case_counter", 1),
]:
    if key not in st.session_state:
        st.session_state[key] = seed_cases() if key == "cases" else default


# -----------------------------------------------------------------------------
# Related-complaint matching: lexical similarity with a justified threshold  [unchanged logic]
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
if ml_predict("status check") is not None:
    st.sidebar.success("ML triage model: loaded", icon="✅")
else:
    st.sidebar.error("ML model unavailable — keyword rules only.", icon="⚠️")

st.sidebar.markdown("**Pipeline (v2.1)**")
st.sidebar.markdown(
    "- ML classifier · word+char TF-IDF → Logistic Regression (`models2.0`)\n"
    "- Transparent keyword evidence (`triage_rules`)\n"
    "- Location & missing-detail flags\n"
    "- Lexical similarity (thresholded)\n"
    "- Session-only case history"
)
st.sidebar.caption(
    "Class scores are uncalibrated model outputs, not probabilities of real-world "
    "correctness. No speech-to-text or BHASHINI API is connected."
)


# -----------------------------------------------------------------------------
# 1. Incoming complaint
# -----------------------------------------------------------------------------
def clear_transcript() -> None:
    """Reset input + results. Runs as a callback (before the widget re-instantiates)."""
    st.session_state.transcript_input = ""
    st.session_state.last_analysis = None
    st.session_state.last_prediction = None
    st.session_state.last_transcript = ""
    st.session_state.last_case_id = None


with st.container(border=True):
    card_header("Step 1", "Incoming complaint")
    transcript = st.text_area(
        "Call transcript",
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
        prediction = ml_predict(transcript.strip())
        case_id = f"HL-{st.session_state.case_counter:04d}"
        st.session_state.case_counter += 1
        created_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        st.session_state.cases.append(make_case(case_id, transcript.strip(), created_at))
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

    # ---- Card: triage result + class scores ----
    with st.container(border=True):
        card_header("Step 2", f"Triage result · case {case_id}")
        st.markdown(
            '<div class="muted">ML routing label with rule-based evidence · uncalibrated model score, not a validated risk score</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        head_l, head_r = st.columns([1.5, 1])
        with head_l:
            if prediction and "predicted_label" in prediction:
                meta = ml_meta(prediction["predicted_label"])
                conf = prediction["model_confidence"] * 100
                st.markdown(
                    f'<span class="triage-badge {meta["badge"]}">{meta["emoji"]} {meta["title"]}</span>'
                    f'<span class="muted" style="margin-left:0.6rem;">model score {conf:.0f}%</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'<div class="muted" style="margin-top:0.45rem;">{prediction.get("description","")}</div>', unsafe_allow_html=True)
                if result["urgency"] == "Critical" and prediction["predicted_label"] != "PRIORITY_REVIEW":
                    st.error(
                        "⚠️ Keyword rules flagged possible CRITICAL indicators while the ML label is "
                        f"'{meta['title']}'. Treat as priority and verify the transcript immediately."
                    )
            else:
                st.error("ML model unavailable — showing keyword-rule evidence only.")
        with head_r:
            st.markdown(
                '<div class="eyebrow">Operator review</div>'
                f'<span class="triage-badge b-neutral badge-sm">📋 {review_status}</span>',
                unsafe_allow_html=True,
            )

        if prediction and "class_scores" in prediction:
            st.write("")
            st.markdown('<div class="eyebrow">Model class scores (uncalibrated)</div>', unsafe_allow_html=True)
            render_score_bars(prediction["class_scores"])
            st.caption(prediction.get("confidence_note", "Uncalibrated model score; not a probability of real-world correctness."))

    # ---- Two cards: evidence | information to verify ----
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

    # ---- Card: operator review ----
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

    # ---- Card: related transcripts ----
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
# 5. Session case history
# -----------------------------------------------------------------------------
with st.container(border=True):
    card_header("Step 5", "Session case history")
    st.markdown('<div class="muted">Kept in Streamlit session state only — not a permanent database.</div>', unsafe_allow_html=True)
    st.write("")
    if st.session_state.cases:
        history = pd.DataFrame(st.session_state.cases)
        display_columns = [
            "case_id", "ml_label", "ml_score", "rule_cue", "threat", "location",
            "suggested_queue", "review_status", "created_at", "source",
        ]
        export_columns = [
            "case_id", "transcript", "ml_label", "ml_score", "rule_cue", "threat", "location",
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
                st.session_state.case_counter = 1
                st.rerun()
    else:
        st.info("No cases recorded in this session yet.")

st.caption(
    "Prototype v2.1 — trained ML triage (models2.0: word+char TF-IDF → Logistic Regression) with a "
    "transparent keyword-rule evidence layer, basic location clues, thresholded lexical similarity, "
    "and temporary session history. Scores are uncalibrated model outputs on synthetic data. "
    "BHASHINI, speech-to-text, semantic embeddings, calibrated probabilities, and real-world "
    "validation are future work."
)
