"""AI-Based Help Line Triage: ML-assisted transcript review demo.

For fictional/authorized demo inputs only. The model is not a validated
emergency-risk assessment system and must never dispatch services.
"""

from __future__ import annotations

import html
import importlib.util
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

REVIEW_STATUS_OPTIONS = [
    "Pending review",
    "Reviewed — more information needed",
    "Reviewed — follow-up required",
    "Reviewed — escalation according to local protocol",
    "Reviewed — no further action recorded",
]

# Fictional UI test transcript. It is only loaded into the editor; the
# results are always produced by the real model and rules below.
DEMO_TRANSCRIPT = (
    "Hello, mujhe help chahiye. Main abhi metro station ke bahar hoon. "
    "Ek aadmi kaafi der se mera peecha kar raha hai aur ab mujhe dhamki de raha hai. "
    "Woh mere paas aa raha hai aur mujhe bahut darr lag raha hai. "
    "Mujhe samajh nahi aa raha ki main kya karun. Please meri help kijiye."
)

# Display wording only. The label codes themselves are unchanged.
LABEL_META = {
    "PRIORITY_REVIEW": {
        "tone": "priority",
        "title": "Priority review",
        "short": "priority review",
        "message": "Prompt human review is recommended. This is a prompt for assessment, not an automatic dispatch instruction.",
    },
    "SUPPORT_REVIEW": {
        "tone": "support",
        "title": "Support review",
        "short": "support review",
        "message": "A human operator should confirm the caller's needs, timing, and context.",
    },
    "UNCLEAR_REVIEW": {
        "tone": "unclear",
        "title": "Unclear — human assessment needed",
        "short": "unclear review",
        "message": "Unclear does not mean safe. Follow up to establish context and immediate safety.",
    },
}


def compact_html(markup: str) -> str:
    """Strip indentation and blank lines so Markdown never treats HTML as a code block."""
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())


st.set_page_config(
    page_title="AI-Based Help Line Triage",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_CSS = """
:root{--bg:#F7F6F3;--border:#E6E4DF;--text:#101828;--body:#344054;--muted:#667085;
--accent:#0E7490;--accent-dark:#0B5C73;--accent-soft:#ECFEFF;--accent-line:#A5E3EE;
--red:#B42318;--red-soft:#FEF3F2;--red-line:#FECDCA;
--amber:#B54708;--amber-soft:#FFFAEB;--amber-line:#FEDF89;}
html{color-scheme:light;}
.stApp{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu,footer{visibility:hidden;}
.block-container{max-width:1180px;padding-top:3.75rem;padding-bottom:2.5rem;}
.stButton>button,.stDownloadButton>button{border-radius:8px;font-weight:550;}
.stButton>button[kind="primary"],button[data-testid="stBaseButton-primary"]{background:var(--accent);border-color:var(--accent);color:#fff;}
.stButton>button[kind="primary"]:hover,button[data-testid="stBaseButton-primary"]:hover{background:var(--accent-dark);border-color:var(--accent-dark);color:#fff;}
.stButton>button:disabled{opacity:.5;}
button[data-baseweb="tab"]{font-weight:550;}
[data-baseweb="tab-highlight"]{background-color:var(--accent);}
.stTextArea textarea{background:#fff;font-size:.95rem;line-height:1.55;}
[data-testid="stVerticalBlockBorderWrapper"]{background:#fff;border-color:var(--border);border-radius:12px;}
[data-testid="stExpander"] details{background:#fff;border-color:var(--border);border-radius:12px;}
.hl-header{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;padding:0 0 14px;}
.hl-brand{display:flex;align-items:center;gap:12px;}
.hl-mark{width:38px;height:38px;border-radius:10px;background:var(--accent);display:flex;align-items:center;justify-content:center;gap:3px;}
.hl-mark span{display:block;width:3px;border-radius:2px;background:#fff;}
.hl-title{font-size:1.2rem;font-weight:650;letter-spacing:-.01em;line-height:1.2;}
.hl-tag{font-size:.86rem;color:var(--muted);}
.hl-status{display:flex;gap:8px;flex-wrap:wrap;}
.hl-pill{display:inline-flex;align-items:center;gap:7px;font-size:.78rem;color:var(--body);background:#fff;border:1px solid var(--border);border-radius:999px;padding:4px 11px;}
.hl-dot{width:7px;height:7px;border-radius:50%;background:#98A2B3;}
.hl-dot.ok{background:#12B76A;}
.hl-dot.warn{background:#F79009;}
.hl-dot.bad{background:#D92D20;}
.hl-notice{background:#fff;border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:10px;padding:10px 14px;font-size:.84rem;line-height:1.5;color:var(--body);margin-bottom:6px;}
.hl-label{font-size:.72rem;font-weight:650;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);margin:2px 0 2px;}
.hl-reco{border:1px solid var(--border);border-left-width:4px;border-radius:12px;padding:16px 18px;background:#fff;}
.hl-reco.priority{border-color:var(--red-line);border-left-color:var(--red);background:var(--red-soft);}
.hl-reco.support{border-color:var(--accent-line);border-left-color:var(--accent);background:var(--accent-soft);}
.hl-reco.unclear{border-color:var(--amber-line);border-left-color:var(--amber);background:var(--amber-soft);}
.hl-reco-top{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;}
.hl-reco-title{font-size:1.25rem;font-weight:650;letter-spacing:-.01em;}
.hl-reco p{margin:6px 0 0;font-size:.9rem;color:var(--body);}
.hl-code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.76rem;padding:2px 8px;border-radius:6px;background:rgba(16,24,40,.07);color:var(--body);}
.hl-compare{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:4px;}
.hl-cell{background:#fff;border:1px solid var(--border);border-radius:10px;padding:10px 12px;}
.hl-cell.changed{border-color:var(--red-line);}
.hl-cell small{display:block;font-size:.7rem;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);margin-bottom:3px;}
.hl-cell code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.85rem;font-weight:600;color:var(--text);background:none;padding:0;}
.hl-rows{background:#fff;border:1px solid var(--border);border-radius:12px;overflow:hidden;}
.hl-row{display:grid;grid-template-columns:150px 1fr;gap:12px;padding:11px 14px;border-top:1px solid #EEEDE9;font-size:.88rem;line-height:1.5;}
.hl-row:first-child{border-top:none;}
.hl-k{color:var(--muted);font-weight:550;}
.hl-v{color:var(--text);}
.hl-v ul,.hl-summary ul{margin:0;padding-left:1.1rem;}
.hl-v li,.hl-summary li{margin:0 0 3px;}
.hl-none{color:var(--muted);}
.hl-chip{display:inline-block;font-size:.7rem;font-weight:600;color:var(--amber);background:var(--amber-soft);border:1px solid var(--amber-line);border-radius:999px;padding:0 8px;margin-left:6px;}
.hl-summary{background:#fff;border:1px solid var(--border);border-radius:12px;padding:14px 16px;font-size:.9rem;line-height:1.55;color:var(--body);}
.hl-summary .hl-label{margin-bottom:8px;}
.hl-next{margin-top:10px;padding-top:10px;border-top:1px solid #EEEDE9;}
.hl-flow{display:flex;flex-wrap:wrap;align-items:center;gap:6px;font-size:.8rem;color:var(--muted);}
.hl-flow b{font-weight:500;color:var(--body);background:#fff;border:1px solid var(--border);border-radius:999px;padding:3px 10px;}
.hl-flow i{font-style:normal;color:#98A2B3;}
.hl-empty{border:1px dashed #D0D5DD;border-radius:12px;padding:26px 22px;background:rgba(255,255,255,.6);}
.hl-empty-title{font-size:1rem;font-weight:650;margin-bottom:4px;}
.hl-empty p{margin:0 0 14px;font-size:.88rem;line-height:1.5;color:var(--muted);}
@media (max-width:900px){.hl-row{grid-template-columns:1fr;gap:2px;}}
@media (max-width:640px){.hl-compare{grid-template-columns:1fr;}}
"""
st.markdown("<style>" + compact_html(APP_CSS) + "</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the trained triage model...")
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
        (r"\b(?:has a weapon|weapon in (?:his|her) hand|gun pointed|knife out|has a knife out|bandook dikha raha|bandook dikha rahi|chaku dikha raha|chaku dikha rahi)\b|चाकू दिखा|बंदूक दिखा|उसके हाथ में चाकू|हाथ में बंदूक", "Possible weapon indicator"),
        (r"\b(?:trying to kill me|threatened to kill me|attacking me|being attacked|stabbed me|someone is attacking me)\b|जान से मार|मुझ पर हमला|मुझपर हमला|हमला हो रहा", "Possible serious physical danger indicator"),
        (r"\b(?:threatening me|threatening to|threats? against me|dhamki de raha|dhamki de rahi|dhamki de raha hai|dhamki de rahi hai|mujhe dhamki de raha|mujhe dhamki de rahi|dhamki de raha hai)\b|धमकी दे रहा|धमकी दे रही|धमकी भरे संदेश|मुझे धमकी दे रहा|मुझे धमकी दे रही", "Possible threat indicator"),
        (r"\b(?:following me|stalking me|keeps following|mera peecha kar raha|mera peecha kar rahi|mujhe follow kar raha|mujhe follow kar rahi|peecha kar raha|peecha kar rahi|mera peecha kar raha hai|mera peecha kar rahi hai)\b|मेरा पीछा कर रहा|मेरा पीछा कर रही|मेरा पीछा कर रहा है|मेरा पीछा कर रही है|पीछा कर रहा|पीछा कर रही", "Possible following/stalking indicator"),
        (r"\b(?:blackmail|doxxing|fake profile|unwanted messages|baar baar dhamki|dhamki wale messages|threatening messages)\b|ब्लैकमेल|फर्जी प्रोफाइल|निजी जानकारी|धमकी भरे संदेश", "Possible harassment/blackmail indicator"),
        (r"\b(?:crash|accident|unconscious|heavy bleeding|overdose|swallowed tablets)\b|दुर्घटना|बेहोश|खून बह|गोलियां खा|गोलियाँ खा", "Possible medical/accident indicator"),
    ]
    # Basic local negation guard: avoid flagging some phrases such as
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

    evidence = [
        label for pattern, label in evidence_patterns
        if has_unnegated_match(pattern, text)
    ]

    location_patterns = [
        (r"\b(?:near|at|outside|inside|by) (?:the )?metro station\b|\bmetro station ke (?:paas|bahar)\b|मेट्रो स्टेशन के (?:पास|बाहर)", "metro station"),
        (r"\b(?:near|at|outside|inside|by) (?:the )?bus stop\b|\bbus stop ke (?:paas|bahar)\b|बस स्टॉप के (?:पास|बाहर)", "bus stop"),
        (r"\b(?:near|at|outside|inside) (?:the )?college\b|\bcollege ke (?:paas|bahar)\b|कॉलेज के (?:पास|बाहर)", "college"),
        (r"\b(?:at|near|outside) (?:my )?home\b|\bhome ke paas\b|\bghar (?:par|pe|ke paas|ke bahar)\b|घर (?:पर|के पास|के बाहर)", "home"),
        (r"\b(?:near|at|outside) (?:the )?market\b|\bmarket ke (?:paas|bahar)\b|बाज़ार के पास|बाजार के पास|मार्केट में", "market"),
        (r"\b(?:near|at|outside) (?:the )?hospital\b|\bhospital ke (?:paas|bahar)\b|अस्पताल के पास", "hospital"),
        (r"\b(?:near|at|outside) (?:the )?hostel\b|\bhostel ke (?:paas|bahar)\b|हॉस्टल के पास|हॉस्टल में", "hostel"),
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
    if not re.search(r"\b(safe|unsafe|danger|help|scared|afraid|hurt|injured|darr|dar|darr lag|darr lag raha|bahut darr|mere paas aa raha|mere paas aa rahi|coming towards me|approaching me)\b|सुरक्षित|डर|मदद|घायल|खतरा", text):
        missing.append("Immediate safety status needs clarification.")

    return {
        "evidence": evidence,
        "location_clues": list(dict.fromkeys(locations)),
        "missing_information": missing,
    }


def assess_combined_safety_cues(transcript: str, details: dict[str, Any]) -> dict[str, Any]:
    """Conservative cue-based escalation for human review, separate from ML.

    This is not another trained model or an emergency decision. It escalates only
    a specific combination of supported current-risk indicators for human review.
    """
    text = normalize_for_matching(transcript)

    # Very small negation and historical guards. These are intentionally limited;
    # context is complex, so the original transcript still needs human review.
    negated_or_hypothetical = bool(re.search(
        r"(?:\bnot\s+(?:following|stalking|threatening)|"
        r"\bnever\s+(?:followed|threatened)|"
        r"\bno one is (?:following|threatening)|"
        r"\bused to (?:follow|threaten)|"
        r"\bin the past\b|\byears ago\b|\blast week\b|"
        r"\bfor my (?:essay|assignment|novel|story)\b|"
        r"\bin a (?:movie|film|play|drama|scene)\b|"
        r"\bwhat does .*\bmean\b|\bif somebody\b|\bhypothetically\b|"
        r"नहीं .* पीछा|नहीं .* धमकी|पिछले सप्ताह|कुछ साल पहले)",
        text,
        flags=re.IGNORECASE,
    ))

    following = bool(re.search(
        r"\b(following me|stalking me|keeps following|"
        r"mera peecha kar raha(?: hai)?|mera peecha kar rahi(?: hai)?|"
        r"mujhe follow kar raha(?: hai)?|mujhe follow kar rahi(?: hai)?|"
        r"peecha kar raha(?: hai)?|peecha kar rahi(?: hai)?)\b|"
        r"मेरा पीछा कर रहा|मेरा पीछा कर रही|पीछा कर रहा|पीछा कर रही",
        text,
        flags=re.IGNORECASE,
    ))
    threat = bool(re.search(
        r"\b(threatening me|threatening to|threats? against me|"
        r"dhamki de raha(?: hai)?|dhamki de rahi(?: hai)?|"
        r"mujhe dhamki|dhamki wale messages)\b|"
        r"धमकी दे रहा|धमकी दे रही|धमकी भरे संदेश|मुझे धमकी",
        text,
        flags=re.IGNORECASE,
    ))
    fear_or_approach = bool(re.search(
        r"\b(scared|afraid|frightened|darr lag raha|darr lag rahi|"
        r"mujhe darr|bahut darr|coming towards me|walking towards me|"
        r"approaching me|comes closer|moving towards me|"
        r"mere paas aa raha|mere paas aa rahi|mere kareeb aa raha|"
        r"mere kareeb aa rahi|abhi)\b|डर लग रहा|मुझे डर|अभी",
        text,
        flags=re.IGNORECASE,
    ))

    # Escalate to prompt human review when multiple independently observable
    # current-safety cues co-occur. This keeps the original ML prediction visible.
    should_escalate = following and threat and fear_or_approach and not negated_or_hypothetical
    reasons = []
    if following:
        reasons.append("possible following/stalking cue")
    if threat:
        reasons.append("possible threat cue")
    if fear_or_approach:
        reasons.append("fear, approach, or current-time cue")

    return {
        "rule_escalation": should_escalate,
        "rule_escalation_reasons": reasons if should_escalate else [],
        "review_recommendation": "PRIORITY_REVIEW" if should_escalate else None,
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
            response = client.speech_to_text.transcribe(
                file=audio_file,
                model="saaras:v4",
                mode=mode,
            )
        transcript = getattr(response, "transcript", None)
        language_code = getattr(response, "language_code", None)
        if not transcript or not str(transcript).strip():
            raise RuntimeError("Sarvam returned an empty transcript. Try a clearer short clip.")
        return str(transcript).strip(), language_code
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)



# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
class UnexpectedLabelError(ValueError):
    """Raised when the trained pipeline returns a label outside LABELS."""


def esc(value: Any) -> str:
    """Escape dynamic text for safe use inside HTML (and neutralise LaTeX '$')."""
    return html.escape(str(value), quote=True).replace("$", "&#36;")


def render_html(markup: str) -> None:
    st.markdown(compact_html(markup), unsafe_allow_html=True)


def stretch_button(label: str, **kwargs: Any) -> bool:
    """Full-width button that works across Streamlit versions."""
    try:
        return st.button(label, width="stretch", **kwargs)
    except Exception:
        return st.button(label, use_container_width=True, **kwargs)


def show_table(frame: pd.DataFrame) -> None:
    try:
        st.dataframe(frame, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(frame, use_container_width=True, hide_index=True)


def safe_error(exc: Exception) -> str:
    """Describe an exception without ever echoing the API key."""
    message = f"{type(exc).__name__}: {exc}"
    key = get_sarvam_api_key()
    if key:
        message = message.replace(key, "[hidden]")
    return message


def list_html(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{esc(item)}</li>" for item in items) + "</ul>"


def clear_transcript() -> None:
    st.session_state.transcript_input = ""
    st.session_state.last_result = None
    st.session_state.last_detected_language = None


def load_demo_transcript() -> None:
    st.session_state.transcript_input = DEMO_TRANSCRIPT
    st.session_state.last_result = None
    st.session_state.last_detected_language = None


def run_analysis(text: str) -> dict[str, Any]:
    """Same flow as the original app: ML prediction + separate rule-based checks."""
    prediction = str(MODEL.predict([text])[0])
    if prediction not in LABELS:
        raise UnexpectedLabelError(
            f"The model returned an unexpected label ({prediction}). No result is shown."
        )
    details = extract_details(text)
    safety_check = assess_combined_safety_cues(text, details)
    # Keep the model output transparent. A separate, conservative cue check
    # can recommend priority human review without rewriting the ML prediction.
    review_recommendation = safety_check["review_recommendation"] or prediction
    queue, next_step = suggested_queue(review_recommendation)
    if safety_check["rule_escalation"]:
        next_step = (
            "Multiple configured safety cues were found. Promptly review the original transcript, "
            "verify current safety and location, and follow established organisational procedures. "
            "This rule flag is not proof of an emergency and does not dispatch services."
        )
    case_id = f"HL-{len(st.session_state.cases) + 1:04d}"
    return {
        "case_id": case_id,
        "transcript": text,
        "label": prediction,
        "model_label": prediction,
        "review_recommendation": review_recommendation,
        "rule_escalation": safety_check["rule_escalation"],
        "rule_escalation_reasons": safety_check["rule_escalation_reasons"],
        "queue": queue,
        "next_step": next_step,
        **details,
        "review_status": "Pending review",
        "operator_notes": "",
        "created_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
    }


def build_status_pills() -> list[tuple[str, str]]:
    """Status is derived from real checks: model loaded, API key present, SDK importable."""
    pills: list[tuple[str, str]] = []
    if MODEL is not None:
        pills.append(("ok", "Triage model loaded"))
    else:
        pills.append(("bad", "Triage model unavailable"))
    key_found = bool(get_sarvam_api_key())
    try:
        sdk_found = importlib.util.find_spec("sarvamai") is not None
    except (ImportError, ValueError):
        sdk_found = False
    if key_found and sdk_found:
        pills.append(("ok", "Transcription: API key configured"))
    elif key_found:
        pills.append(("warn", "Transcription: SDK not installed"))
    else:
        pills.append(("neutral", "Transcription: not configured"))
    return pills


def render_header() -> None:
    bars = "".join(f'<span style="height:{h}px"></span>' for h in (10, 20, 26, 16, 8))
    pills = "".join(
        f'<span class="hl-pill"><span class="hl-dot {tone}"></span>{esc(text)}</span>'
        for tone, text in build_status_pills()
    )
    render_html(
        f"""
        <div class="hl-header">
          <div class="hl-brand">
            <div class="hl-mark">{bars}</div>
            <div>
              <div class="hl-title">AI-Based Help Line Triage</div>
              <div class="hl-tag">Understand the Call. Guide the Response.</div>
            </div>
          </div>
          <div class="hl-status">{pills}</div>
        </div>
        """
    )
    render_html(
        """
        <div class="hl-notice">
          <b>Decision support for a human operator.</b>
          Review a caller's transcript, see the evidence behind a suggested priority, and decide what happens next.
          This prototype does not contact authorities, dispatch services, or decide that a caller is safe.
          The model can make mistakes, so verify every transcript and result. Use fictional or authorised data only.
        </div>
        """
    )


def render_flow() -> None:
    steps = ["Audio or text", "Transcription (audio only)", "ML prediction", "Safety-rule checks", "Human review"]
    render_html('<div class="hl-flow">' + "<i>→</i>".join(f"<b>{esc(s)}</b>" for s in steps) + "</div>")


def build_summary_points(result: dict[str, Any]) -> list[str]:
    label = result["review_recommendation"]
    model_label = result["model_label"]
    reasons = ", ".join(result["rule_escalation_reasons"])
    points = [f"Recommended for {LABEL_META[label]['short']} ({label})."]
    if result["rule_escalation"] and model_label != label:
        points.append(
            f"The ML model predicted {model_label}. Safety-rule cues raised the recommendation to {label}: {reasons}."
        )
    elif result["rule_escalation"]:
        points.append(
            f"The ML model predicted {model_label}, and safety-rule cues ({reasons}) also point to priority review."
        )
    else:
        points.append(f"The ML model predicted {model_label}. No safety-rule escalation applied.")
    if result["evidence"]:
        points.append("Matched indicators: " + "; ".join(result["evidence"]) + ".")
    else:
        points.append("No configured evidence phrase matched. This does not rule out danger.")
    if result["location_clues"]:
        points.append("Possible location clue: " + ", ".join(result["location_clues"]) + " (unverified).")
    else:
        points.append("No location clue identified. Ask for the caller's current location.")
    if result["missing_information"]:
        points.append("Verify first: " + " ".join(result["missing_information"][:3]))
    return points


def render_result(result: dict[str, Any]) -> None:
    label = result["review_recommendation"]
    model_label = result["model_label"]
    meta = LABEL_META[label]
    changed = model_label != label

    # 1. Human-review recommendation
    render_html(
        f"""
        <div class="hl-label">Review recommendation · Case {esc(result['case_id'])}</div>
        <div class="hl-reco {meta['tone']}">
          <div class="hl-reco-top">
            <div class="hl-reco-title">{esc(meta['title'])}</div>
            <span class="hl-code">{esc(label)}</span>
          </div>
          <p>{esc(meta['message'])}</p>
        </div>
        """
    )

    # 2. ML prediction shown separately from the final recommendation
    render_html(
        f"""
        <div class="hl-compare">
          <div class="hl-cell"><small>ML model prediction</small><code>{esc(model_label)}</code></div>
          <div class="hl-cell{' changed' if changed else ''}"><small>After safety-rule checks</small><code>{esc(label)}</code></div>
        </div>
        """
    )
    if result["rule_escalation"]:
        st.caption(
            "Safety-rule cues were detected together and recommend priority human review. "
            "The original ML prediction is shown unchanged; additional safety cues may elevate a case above it."
        )
    else:
        st.caption("No safety-rule escalation applied, so the recommendation follows the ML prediction.")

    # 3. Evidence and extracted information
    evidence_html = (
        list_html(result["evidence"])
        if result["evidence"]
        else '<span class="hl-none">Could not identify. This does not rule out danger.</span>'
    )
    cues_html = (
        list_html(result["rule_escalation_reasons"])
        if result["rule_escalation"]
        else '<span class="hl-none">No escalation rule triggered</span>'
    )
    if result["location_clues"]:
        location_html = esc(", ".join(result["location_clues"])) + '<span class="hl-chip">unverified</span>'
    else:
        location_html = '<span class="hl-none">Could not identify</span>'
    verify_html = (
        list_html(result["missing_information"])
        if result["missing_information"]
        else '<span class="hl-none">Not provided</span>'
    )
    render_html(
        f"""
        <div class="hl-label">Evidence and extracted information</div>
        <div class="hl-rows">
          <div class="hl-row"><div class="hl-k">Evidence indicators</div><div class="hl-v">{evidence_html}</div></div>
          <div class="hl-row"><div class="hl-k">Safety-rule cues</div><div class="hl-v">{cues_html}</div></div>
          <div class="hl-row"><div class="hl-k">Location clues</div><div class="hl-v">{location_html}</div></div>
          <div class="hl-row"><div class="hl-k">Information to verify</div><div class="hl-v">{verify_html}</div></div>
        </div>
        """
    )
    st.caption("Evidence uses a limited phrase list and is not proof that an event occurred.")

    # 4. Human-review summary (built only from the actual outputs above)
    render_html(
        f"""
        <div class="hl-summary">
          <div class="hl-label">Human-review summary</div>
          {list_html(build_summary_points(result))}
          <div class="hl-next">
            <b>Suggested queue:</b> {esc(result['queue'])}<br>
            <b>Suggested next step:</b> {esc(result['next_step'])}
          </div>
        </div>
        """
    )
    st.caption(
        "Verify the transcript and evidence before acting. The model can make mistakes, especially with "
        "Hinglish, code words, dialects, and noisy transcription."
    )

    # Operator review (existing functionality, preserved)
    with st.expander("Record operator review"):
        status = st.selectbox("Review status", REVIEW_STATUS_OPTIONS, key=f"status_{result['case_id']}")
        notes = st.text_area(
            "Operator notes (fictional demo only)",
            key=f"notes_{result['case_id']}",
            placeholder="Record what was verified; do not enter real personal information.",
            height=90,
        )
        if st.button("Save review", key=f"save_{result['case_id']}"):
            for case in st.session_state.cases:
                if case["case_id"] == result["case_id"]:
                    case["review_status"] = status
                    case["operator_notes"] = notes.strip()
                    break
            st.success("Review saved in this browser session.")


def render_empty_state(message: str) -> None:
    render_html(
        f"""
        <div class="hl-label">Triage review</div>
        <div class="hl-empty">
          <div class="hl-empty-title">{esc(message)}</div>
          <p>You will see the ML prediction, the recommendation after safety-rule checks,
          the evidence found, and what a human operator should verify.</p>
        </div>
        """
    )
    render_flow()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "cases" not in st.session_state:
    st.session_state.cases = []
if "transcript_input" not in st.session_state:
    st.session_state.transcript_input = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_detected_language" not in st.session_state:
    st.session_state.last_detected_language = None


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
render_header()

if MODEL is None:
    st.error(
        "The triage model could not be loaded, so analysis is unavailable. "
        + (MODEL_ERROR or "Expected file: models/triage_pipeline.joblib")
    )

left, right = st.columns([5, 6], gap="large")

with left:
    render_html('<div class="hl-label">Call input</div>')
    with st.container(border=True):
        tab_audio, tab_text = st.tabs(["Upload audio", "Type or paste"])

        with tab_audio:
            audio_file = st.file_uploader(
                "Audio recording (short clips up to 30 seconds for synchronous transcription)",
                type=AUDIO_TYPES,
                help="Use a fictional or specifically authorised clip. Audio will be sent to Sarvam AI when you request transcription.",
            )
            if audio_file is not None:
                try:
                    st.audio(audio_file)
                except Exception:
                    pass
            sarvam_mode = st.selectbox(
                "Transcription style",
                ["transcribe", "codemix", "translit"],
                index=0,
                help="Try transcribe first. Compare codemix/translit on fictional Hinglish examples if needed.",
            )
            consent = st.checkbox(
                "I confirm this is fictional or authorised audio and may be sent to Sarvam AI for transcription.",
                key="audio_consent",
            )
            transcribe_clicked = stretch_button(
                "Transcribe with Sarvam",
                key="transcribe_btn",
                disabled=audio_file is None,
            )
            if transcribe_clicked:
                if not consent:
                    st.error("Please confirm the audio is fictional or authorised before sending it to Sarvam.")
                elif not get_sarvam_api_key():
                    st.error("Sarvam API key is missing. Add SARVAM_API_KEY to Streamlit Secrets. You can still type a transcript.")
                elif audio_file is not None:
                    try:
                        with st.spinner("Transcribing audio with Sarvam..."):
                            transcript_result, detected_language = transcribe_audio(
                                audio_file.getvalue(), audio_file.name, sarvam_mode
                            )
                        # Set before the transcript editor is created in this run.
                        st.session_state.transcript_input = transcript_result
                        st.session_state.last_detected_language = detected_language
                        st.session_state.last_result = None
                        st.success("Transcription complete. Review and correct the text below before analysis.")
                    except ImportError:
                        st.error("The sarvamai package is not installed in this environment. Check requirements.txt.")
                    except Exception as exc:
                        st.error(f"Transcription failed. {safe_error(exc)}")
                        st.info("Nothing was transcribed. You can still type or paste a transcript.")

        with tab_text:
            st.caption(
                "Type or paste the transcript into the editor below. Typed transcripts do not use the Sarvam API."
            )
            st.button("Use fictional example", key="demo_btn", on_click=load_demo_transcript)

    render_html('<div class="hl-label">Transcript</div>')
    if st.session_state.last_detected_language:
        st.caption(f"Sarvam detected language: {st.session_state.last_detected_language}")
    transcript = st.text_area(
        "Transcript — review and edit before analysis",
        key="transcript_input",
        height=180,
        placeholder="Example: Woh mera peecha kar raha hai aur mujhe darr lag raha hai.",
        help="You may edit the transcript if speech recognition made a mistake.",
    )

    analyse_col, clear_col = st.columns([3, 1])
    with analyse_col:
        # Not disabled for an empty editor: a disabled state can swallow the first click
        # right after typing. Empty input is handled with a message instead.
        analyse_clicked = stretch_button(
            "Analyse transcript", type="primary", key="analyse_btn", disabled=MODEL is None
        )
    with clear_col:
        stretch_button("Clear", key="clear_btn", on_click=clear_transcript)

    if analyse_clicked:
        text = transcript.strip()
        if not text:
            st.warning("Enter or transcribe a transcript before analysing.")
        else:
            try:
                with st.spinner("Analysing transcript..."):
                    new_result = run_analysis(text)
                st.session_state.last_result = new_result
                st.session_state.cases.append(new_result.copy())
            except UnexpectedLabelError as exc:
                st.session_state.last_result = None
                st.error(str(exc))
            except Exception as exc:
                st.session_state.last_result = None
                st.error(f"Analysis failed. {safe_error(exc)}")

with right:
    current_result = st.session_state.last_result
    if current_result is None:
        render_empty_state("Results appear here after analysis")
    elif current_result["transcript"] != transcript.strip():
        # Never show a result next to a transcript it was not produced from.
        render_empty_state("The transcript changed since the last analysis. Analyse again to refresh the results")
    else:
        render_result(current_result)

st.write("")
with st.expander(f"Session case history ({len(st.session_state.cases)})"):
    st.caption("Temporary: this history resets when the Streamlit session is lost. Do not use real caller data.")
    if st.session_state.cases:
        history = pd.DataFrame(st.session_state.cases)
        for col in ("evidence", "location_clues", "missing_information", "rule_escalation_reasons"):
            if col in history.columns:
                history[col] = history[col].apply(lambda v: "; ".join(v) if isinstance(v, list) else v)
        display_cols = ["case_id", "model_label", "review_recommendation", "queue", "location_clues", "review_status", "created_at"]
        for col in display_cols:
            if col not in history.columns:
                history[col] = ""
        show_table(history[display_cols])
        st.download_button(
            "Download session history CSV",
            data=history.to_csv(index=False).encode("utf-8-sig"),
            file_name="helpline_triage_session_history.csv",
            mime="text/csv",
        )
    else:
        st.info("No cases have been analysed in this session yet.")

with st.expander("How it works and limitations"):
    render_flow()
    st.markdown(
        "- Small synthetic training dataset; not a validated emergency-risk model.\n"
        "- Evidence and location rules are limited phrase lists, not proof that an event occurred.\n"
        "- Sarvam transcription can make errors, especially with noise, dialects, and mixed languages.\n"
        "- Text similarity and COMI-LINGUA NER are not integrated.\n"
        "- Every result requires human review. No autonomous dispatch is performed."
    )
