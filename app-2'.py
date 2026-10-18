"""Helpline Triage Console: ML-assisted transcript review demo.

For fictional/authorized demo inputs only. The model is not a validated
emergency-risk assessment system and must never dispatch services.
"""

from __future__ import annotations

import html
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "triage_pipeline.joblib"
LABELS = {"PRIORITY_REVIEW", "SUPPORT_REVIEW", "UNCLEAR_REVIEW"}
AUDIO_TYPES = ["wav", "mp3", "aac", "aiff", "ogg", "opus", "flac", "mp4", "m4a", "amr", "wma", "webm"]

# Display names and colours per label (colour is never the only signal: the text is always shown).
LABEL_STYLE = {
    "PRIORITY_REVIEW": ("Priority review", "#9F2D2D", "#F6E7E6"),
    "SUPPORT_REVIEW": ("Support review", "#1F5C7A", "#E4EEF3"),
    "UNCLEAR_REVIEW": ("Unclear, needs follow-up", "#76591A", "#F3ECDC"),
}

st.set_page_config(
    page_title="Helpline Triage Console",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
      :root {
        --ink:#1B2530; --muted:#5C6773; --line:#D9DEE3; --panel:#FFFFFF;
        --bg:#F3F5F7; --accent:#1F4E6B;
      }
      html, body, [class*="css"], .stApp, button, input, textarea, select {
        font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
      }
      .stApp {background: var(--bg); color: var(--ink);}
      #MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden; height: 0;}
      .block-container {padding: 0 2.5rem 3rem; max-width: 1280px;}

      .topbar {
        background: var(--ink); color: #fff; margin: 0 -2.5rem 1.5rem;
        padding: .9rem 2.5rem; display: flex; justify-content: space-between; align-items: baseline;
      }
      .topbar .name {font-size: 1.1rem; font-weight: 600; letter-spacing: .01em;}
      .topbar .sub {font-size: .85rem; color: #AEB8C2;}

      .notice {
        border-left: 3px solid #76591A; background: #F3ECDC; color: #4A3910;
        padding: .6rem .9rem; font-size: .85rem; line-height: 1.45; margin-bottom: 1.5rem;
      }
      h2, h3 {color: var(--ink); font-weight: 600; letter-spacing: 0;}
      h2 {font-size: 1.15rem !important; padding-bottom: .4rem; border-bottom: 1px solid var(--line);}
      h3 {font-size: 1rem !important;}

      div[data-testid="stVerticalBlockBorderWrapper"] {background: var(--panel); border-radius: 4px;}
      .stButton > button, .stDownloadButton > button {
        border-radius: 3px; font-weight: 500; border: 1px solid var(--line);
      }
      .stButton > button[kind="primary"] {background: var(--accent); border-color: var(--accent); color: #fff;}
      .stButton > button:focus-visible {outline: 2px solid var(--accent); outline-offset: 2px;}

      .result-head {display:flex; align-items:center; gap:.9rem; margin-bottom:.3rem;}
      .badge {padding:.25rem .7rem; border-radius:3px; font-weight:600; font-size:.95rem;}
      .case-id {color: var(--muted); font-size:.85rem;}
      .field {margin: .9rem 0 .2rem; font-weight:600; font-size:.9rem;}
      .value {font-size:.95rem; line-height:1.5;}
      .note {color: var(--muted); font-size:.8rem; margin-top:.2rem;}
      ul.plain {margin:.2rem 0 0 1.1rem; padding:0; font-size:.93rem; line-height:1.55;}
      .quote {
        background:#F3F5F7; border:1px solid var(--line); padding:.6rem .8rem;
        font-size:.92rem; line-height:1.5; border-radius:3px;
      }
      .empty {color: var(--muted); padding: 2.5rem 0; text-align:center; font-size:.95rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading triage model...")
def load_model(model_path: str):
    """Load the team's trusted trained pipeline once per app process."""
    path = Path(model_path)
    if not path.exists():
        return None
    return joblib.load(path)


try:
    MODEL = load_model(str(MODEL_PATH))
    MODEL_ERROR = None
except Exception as exc:  # show a helpful UI message instead of hiding load errors
    MODEL = None
    MODEL_ERROR = f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Rule-based evidence and detail extraction. This is separate from the ML label.
# ---------------------------------------------------------------------------
def normalize_for_matching(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("’", "'")).strip()


def extract_details(transcript: str) -> dict[str, Any]:
    text = normalize_for_matching(transcript)
    evidence_patterns = [
        (r"\b(?:has a weapon|weapon in (?:his|her) hand|gun pointed|knife out)\b|चाकू दिखा|बंदूक दिखा", "Possible weapon indicator"),
        (r"\b(?:trying to kill me|threatened to kill me|attacking me|being attacked|stabbed me)\b|जान से मार|मुझ पर हमला", "Possible serious physical danger indicator"),
        (r"\b(?:threatening me|threatening to|threats? against me)\b|धमकी दे रहा|धमकी दे रही|धमकी भरे संदेश", "Possible threat indicator"),
        (r"\b(?:following me|stalking me|keeps following)\b|मेरा पीछा कर रहा|मेरा पीछा कर रही|पीछा कर रहा", "Possible following/stalking indicator"),
        (r"\b(?:blackmail|doxxing|fake profile|unwanted messages)\b|ब्लैकमेल|फर्जी प्रोफाइल|निजी जानकारी", "Possible harassment/blackmail indicator"),
        (r"\b(?:crash|accident|unconscious|heavy bleeding|overdose|swallowed tablets)\b|दुर्घटना|बेहोश|खून बह|गोलियां खा", "Possible medical/accident indicator"),
    ]
    # Basic local negation guard: avoid flagging phrases such as
    # "he is not following me" as an active indicator. This is not full
    # natural-language negation/context understanding, so human review remains required.
    negation_before_re = re.compile(
        r"(?:\bnot\b|\bnever\b|\bno\b|\bdidn't\b|\bdid not\b|"
        r"\bdoesn't\b|\bdoes not\b|\bisn't\b|\bis not\b|"
        r"\bwasn't\b|\bwas not\b|\bhaven't\b|\bhave not\b|"
        r"\bhasn't\b|\bhas not\b|\bnahi\b|\bnahin\b|नहीं|नही)"
        r"(?:\s+[\w’'-]+){0,4}\s*$",
        re.IGNORECASE,
    )

    def has_unnegated_match(pattern: str, value: str) -> bool:
        for match in re.finditer(pattern, value, flags=re.IGNORECASE):
            prefix = value[max(0, match.start() - 90):match.start()]
            if negation_before_re.search(prefix) is None:
                return True
        return False

    evidence = [label for pattern, label in evidence_patterns if has_unnegated_match(pattern, text)]

    location_patterns = [
        (r"\b(?:near|at|outside|inside|by) (?:the )?metro station\b|मेट्रो स्टेशन के (?:पास|बाहर)", "metro station"),
        (r"\b(?:near|at|outside|inside|by) (?:the )?bus stop\b|बस स्टॉप के (?:पास|बाहर)", "bus stop"),
        (r"\b(?:near|at|outside|inside) (?:the )?college\b|कॉलेज के (?:पास|बाहर)", "college"),
        (r"\b(?:at|near|outside) (?:my )?home\b|\bhome ke paas\b|घर (?:पर|के पास|के बाहर)", "home"),
        (r"\b(?:near|at|outside) (?:the )?market\b|बाज़ार के पास|बाजार के पास|मार्केट में", "market"),
        (r"\b(?:near|at|outside) (?:the )?hospital\b|अस्पताल के पास", "hospital"),
        (r"\b(?:near|at|outside) (?:the )?hostel\b|हॉस्टल के पास|हॉस्टल में", "hostel"),
    ]
    locations = [name for pattern, name in location_patterns if re.search(pattern, text)]

    missing = []
    if not locations:
        missing.append("Current location is not clearly identified; ask and verify it.")
    else:
        missing.append("Any location clue is unverified; confirm the caller's current location.")
    if not re.search(r"\b(now|right now|currently|at present|abhi|iss waqt|is waqt)\b|अभी|इस वक्त", text):
        missing.append("Whether the situation is happening right now is unclear.")
    if len(text.split()) < 7:
        missing.append("Transcript is short; additional context may be needed.")
    if not re.search(r"\b(safe|unsafe|danger|help|scared|afraid|hurt|injured)\b|सुरक्षित|डर|मदद|घायल|खतरा", text):
        missing.append("Immediate safety status needs clarification.")

    return {
        "evidence": evidence,
        "location_clues": list(dict.fromkeys(locations)),
        "missing_information": missing,
    }


def suggested_queue(label: str) -> tuple[str, str]:
    if label == "PRIORITY_REVIEW":
        return (
            "Priority human review",
            "Promptly review the original transcript, verify current safety and location, and follow the organisation's established protocol if warranted.",
        )
    if label == "SUPPORT_REVIEW":
        return (
            "Support review",
            "Clarify the caller's needs, timing, available evidence, and whether the situation has escalated.",
        )
    return (
        "General intake / clarification",
        "The transcript may be incomplete or ambiguous. Ask follow-up questions; this label does not mean the caller is safe.",
    )


def get_sarvam_api_key() -> str | None:
    # Streamlit Community Cloud secrets first; environment variable as fallback.
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
        raise RuntimeError("Sarvam API key is not configured. Add SARVAM_API_KEY to Streamlit Secrets.")

    from sarvamai import SarvamAI

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


def esc(value: str) -> str:
    return html.escape(str(value))


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
st.session_state.setdefault("cases", [])
st.session_state.setdefault("transcript_input", "")
st.session_state.setdefault("last_result", None)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
model_state = "Model loaded" if MODEL is not None else "Model not loaded"
stt_state = "Speech-to-text ready" if get_sarvam_api_key() else "Speech-to-text off (typed transcripts only)"
st.markdown(
    f"""
    <div class="topbar">
      <span class="name">Helpline Triage Console</span>
      <span class="sub">{model_state} &nbsp;|&nbsp; {stt_state}</span>
    </div>
    <div class="notice">
      Demonstration only. Use fictional or authorised recordings and text. The classifier is trained on a
      small synthetic dataset and is not a validated risk model. An operator must verify every result,
      and the system never dispatches services.
    </div>
    """,
    unsafe_allow_html=True,
)

if MODEL is None:
    st.error(MODEL_ERROR or "Model file not found. Add models/triage_pipeline.joblib to run the classifier.")

intake_col, result_col = st.columns([5, 6], gap="large")

# ---------------------------------------------------------------------------
# Intake
# ---------------------------------------------------------------------------
with intake_col:
    st.header("Call intake")

    with st.expander("Add a recording (optional)", expanded=False):
        audio_file = st.file_uploader(
            "Audio file, up to 30 seconds",
            type=AUDIO_TYPES,
            help="The clip is sent to Sarvam AI when you select Transcribe.",
        )
        sarvam_mode = st.selectbox(
            "Transcription style",
            ["transcribe", "codemix", "translit"],
            help="Start with transcribe. Try codemix or translit if Hinglish text comes out poorly.",
        )
        consent = st.checkbox("This audio is fictional or authorised, and may be sent to Sarvam AI.")
        transcribe_clicked = st.button("Transcribe", disabled=audio_file is None, use_container_width=True)

        if transcribe_clicked:
            if not consent:
                st.error("Confirm the audio is fictional or authorised before transcribing.")
            elif not get_sarvam_api_key():
                st.error("SARVAM_API_KEY is missing. Add it to Streamlit Secrets.")
            else:
                try:
                    with st.spinner("Transcribing..."):
                        text_out, lang = transcribe_audio(audio_file.getvalue(), audio_file.name, sarvam_mode)
                    st.session_state.transcript_input = text_out
                    st.session_state.last_detected_language = lang
                    st.rerun()
                except Exception as exc:
                    st.error(f"Transcription failed ({type(exc).__name__}): {exc}")
                    st.caption("You can still type or paste the transcript below.")

    transcript = st.text_area(
        "Transcript",
        key="transcript_input",
        height=200,
        placeholder="Type or paste what the caller said. Example: Woh mera peecha kar raha hai aur mujhe darr lag raha hai.",
        help="Correct any speech-recognition errors before analysing.",
    )
    if st.session_state.get("last_detected_language"):
        st.caption(f"Detected language: {st.session_state.last_detected_language}. Check the text before analysing.")

    run_col, clear_col = st.columns([3, 1])
    analyse_clicked = run_col.button(
        "Analyse transcript", type="primary", use_container_width=True,
        disabled=MODEL is None or not transcript.strip(),
    )
    if clear_col.button("Clear", use_container_width=True):
        st.session_state.transcript_input = ""
        st.session_state.last_result = None
        st.rerun()

    if analyse_clicked:
        try:
            prediction = str(MODEL.predict([transcript.strip()])[0])
            if prediction not in LABELS:
                st.error(f"The model returned an unexpected label: {prediction}")
            else:
                details = extract_details(transcript)
                queue, next_step = suggested_queue(prediction)
                result = {
                    "case_id": f"HL-{len(st.session_state.cases) + 1:04d}",
                    "transcript": transcript.strip(),
                    "label": prediction,
                    "queue": queue,
                    "next_step": next_step,
                    **details,
                    "review_status": "Pending review",
                    "operator_notes": "",
                    "created_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
                }
                st.session_state.last_result = result
                st.session_state.cases.append(result.copy())
        except Exception as exc:
            st.error(f"Prediction failed ({type(exc).__name__}): {exc}")

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
with result_col:
    st.header("Triage result")
    result = st.session_state.last_result
    if not result:
        st.markdown('<div class="empty">Enter a transcript and select Analyse transcript to see a result here.</div>', unsafe_allow_html=True)
    else:
        name, fg, bg = LABEL_STYLE[result["label"]]
        evidence_html = (
            "<ul class='plain'>" + "".join(f"<li>{esc(e)}</li>" for e in result["evidence"]) + "</ul>"
            if result["evidence"]
            else "<div class='value'>No listed phrase matched. This does not rule out danger.</div>"
        )
        locations = esc(", ".join(result["location_clues"])) if result["location_clues"] else "None found"
        verify_html = "<ul class='plain'>" + "".join(f"<li>{esc(m)}</li>" for m in result["missing_information"]) + "</ul>"
        st.markdown(
            f"""
            <div class="result-head">
              <span class="badge" style="color:{fg};background:{bg};border:1px solid {fg}33">{name}</span>
              <span class="case-id">Case {esc(result['case_id'])} &nbsp;|&nbsp; {esc(result['created_at'])}</span>
            </div>
            <div class="note">This label prompts a human assessment. It is not an emergency decision.</div>

            <div class="field">Transcript</div>
            <div class="quote">{esc(result['transcript'])}</div>

            <div class="field">Suggested queue</div>
            <div class="value">{esc(result['queue'])}</div>
            <div class="note">{esc(result['next_step'])}</div>

            <div class="field">Matched indicators</div>
            {evidence_html}
            <div class="note">Matched against a limited phrase list. Not proof that an event occurred.</div>

            <div class="field">Location clues</div>
            <div class="value">{locations}</div>
            <div class="note">Taken from the text only. Confirm the caller's current location.</div>

            <div class="field">To verify with the caller</div>
            {verify_html}
            """,
            unsafe_allow_html=True,
        )

        st.markdown("&nbsp;")
        status = st.selectbox(
            "Review status",
            [
                "Pending review",
                "Reviewed: more information needed",
                "Reviewed: follow-up required",
                "Reviewed: escalate per local protocol",
                "Reviewed: no further action",
            ],
            key=f"status_{result['case_id']}",
        )
        notes = st.text_area(
            "Operator notes",
            key=f"notes_{result['case_id']}",
            placeholder="Record what you verified. Do not enter real personal information.",
            height=90,
        )
        if st.button("Save review", key=f"save_{result['case_id']}"):
            for case in st.session_state.cases:
                if case["case_id"] == result["case_id"]:
                    case["review_status"] = status
                    case["operator_notes"] = notes.strip()
                    break
            st.success("Review saved for this session.")

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
st.markdown("&nbsp;")
st.header("Session history")
if st.session_state.cases:
    history = pd.DataFrame(st.session_state.cases)
    history["label"] = history["label"].map(lambda x: LABEL_STYLE[x][0])
    history["location_clues"] = history["location_clues"].map(lambda x: ", ".join(x) if x else "")
    cols = {
        "case_id": "Case", "created_at": "Created", "label": "Label",
        "queue": "Queue", "location_clues": "Location clues", "review_status": "Status",
    }
    st.dataframe(history[list(cols)].rename(columns=cols), use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV",
        data=pd.DataFrame(st.session_state.cases).to_csv(index=False).encode("utf-8-sig"),
        file_name="helpline_triage_session_history.csv",
        mime="text/csv",
    )
    st.caption("History lives only in this browser session and is lost when the session ends.")
else:
    st.markdown('<div class="empty">No cases yet. Analysed transcripts will be listed here.</div>', unsafe_allow_html=True)

st.caption(
    "Limitations: small synthetic training set, limited evidence and location rules, possible transcription "
    "errors, and no text-similarity or COMI-LINGUA NER integration. Every result needs human review."
)
