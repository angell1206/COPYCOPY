"""AI-Based Help Line Triage: ML-assisted transcript review demo.

For fictional/authorized demo inputs only. The model is not a validated
emergency-risk assessment system and must never dispatch services.
"""

from __future__ import annotations

import io
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

st.set_page_config(
    page_title="AI-Based Help Line Triage",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
      div[data-testid="stMetric"] {
        background: rgba(128,128,128,.07); padding: .75rem 1rem;
        border-radius: .75rem; border: 1px solid rgba(128,128,128,.15);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


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
# Page content
# ---------------------------------------------------------------------------
st.title("🛡️ AI-Based Help Line Triage")
st.caption("Hinglish helpline transcript analysis · ML prediction plus separate safety-cue review")
st.warning(
    "DEMO ONLY — use fictional or appropriately authorised audio/text. The classifier is trained on a small synthetic dataset and is not a validated emergency-risk model. It can make mistakes. A human operator must verify every result; the system must not dispatch services."
)

with st.sidebar:
    st.header("System status")
    if MODEL is not None:
        st.success("Trained ML model loaded")
    else:
        st.error("Trained ML model is not loaded")
        if MODEL_ERROR:
            st.caption(MODEL_ERROR)
        else:
            st.caption("Expected file: models/triage_pipeline.joblib")
    if get_sarvam_api_key():
        st.success("Sarvam API key configured")
    else:
        st.info("Sarvam API key not configured — typed transcripts still work")
    st.caption("Case history is session-only and is not a permanent database.")

if "cases" not in st.session_state:
    st.session_state.cases = []
if "transcript_input" not in st.session_state:
    st.session_state.transcript_input = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None

st.header("1. Incoming call")
st.write("Upload a short audio clip or enter a transcript manually.")

audio_col, mode_col = st.columns([2, 1])
with audio_col:
    audio_file = st.file_uploader(
        "Audio recording (short clips up to 30 seconds for synchronous transcription)",
        type=AUDIO_TYPES,
        help="Use a fictional or specifically authorised clip. Audio will be sent to Sarvam AI when you request transcription.",
    )
with mode_col:
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

transcribe_col, clear_col = st.columns(2)
with transcribe_col:
    transcribe_clicked = st.button(
        "Transcribe with Sarvam",
        type="secondary",
        use_container_width=True,
        disabled=audio_file is None,
    )
with clear_col:
    clear_clicked = st.button("Clear transcript", use_container_width=True)

if transcribe_clicked:
    if not consent:
        st.error("Please confirm the audio is fictional or authorised before sending it to Sarvam.")
    elif not get_sarvam_api_key():
        st.error("Sarvam API key is missing. Configure SARVAM_API_KEY in Streamlit Secrets.")
    elif audio_file is not None:
        try:
            with st.spinner("Transcribing audio with Sarvam..."):
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

if clear_clicked:
    st.session_state.transcript_input = ""
    st.session_state.last_result = None
    st.rerun()

if st.session_state.get("last_detected_language"):
    st.caption(f"Sarvam detected language: {st.session_state.last_detected_language}")

transcript = st.text_area(
    "Transcript — review this text before analysis",
    key="transcript_input",
    height=160,
    placeholder="Example: Woh mera peecha kar raha hai aur mujhe darr lag raha hai.",
    help="You may edit the transcript if speech recognition made a mistake.",
)

analyse_clicked = st.button(
    "Analyse transcript",
    type="primary",
    use_container_width=True,
    disabled=MODEL is None or not transcript.strip(),
)

if MODEL is None:
    st.warning("The text classifier cannot run until models/triage_pipeline.joblib is uploaded and loads correctly.")

if analyse_clicked:
    try:
        prediction = str(MODEL.predict([transcript.strip()])[0])
        if prediction not in LABELS:
            st.error(f"Unexpected label returned by model: {prediction}")
        else:
            details = extract_details(transcript)
            safety_check = assess_combined_safety_cues(transcript, details)
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
            result = {
                "case_id": case_id,
                "transcript": transcript.strip(),
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
            st.session_state.last_result = result
            st.session_state.cases.append(result.copy())
    except Exception as exc:
        st.error(f"Model prediction failed: {type(exc).__name__}: {exc}")

result = st.session_state.last_result
if result:
    st.divider()
    st.header("2. Triage review card")
    st.caption(f"Case ID: {result['case_id']} · ML label + separate rule-based evidence extraction")

    m1, m2, m3 = st.columns(3)
    m1.metric("ML model prediction", result.get("model_label", result["label"]))
    m2.metric("Review recommendation", result.get("review_recommendation", result["label"]))
    m3.metric("Evidence indicators", len(result["evidence"]))

    if result.get("rule_escalation"):
        st.error(
            "PRIORITY HUMAN REVIEW RECOMMENDED: multiple configured cues for possible following, "
            "threat, and fear/approach were detected. The ML model's own prediction is shown separately. "
            "Verify the original transcript and current safety; this rule flag does not dispatch services."
        )
        st.write("**Rule cues detected:**", ", ".join(result.get("rule_escalation_reasons", [])))
    elif result.get("review_recommendation", result["label"]) == "PRIORITY_REVIEW":
        st.warning("Priority review recommended — a prompt for human assessment, not an automatic emergency decision.")
    elif result.get("review_recommendation", result["label"]) == "SUPPORT_REVIEW":
        st.info("Support review recommended — confirm needs, timing, and context with a human operator.")
    else:
        st.warning("Unclear review — unclear does not mean safe. Follow up to establish context and immediate safety.")

    left, right = st.columns(2)
    with left:
        st.subheader("Possible concern and routing")
        st.write("**Suggested queue:**", result["queue"])
        st.write("**Suggested next step:**", result["next_step"])
        st.markdown("**Matched evidence indicators**")
        if result["evidence"]:
            for item in result["evidence"]:
                st.markdown(f"- {item}")
        else:
            st.write("No configured evidence phrase matched. This does not rule out danger.")
        st.caption("Evidence indicators use a limited rule list; they are not proof that an event occurred.")
    with right:
        st.subheader("Location clues")
        if result["location_clues"]:
            st.write(", ".join(result["location_clues"]))
            st.caption("Unverified text clues only; confirm the caller's current location.")
        else:
            st.write("No supported location clue found.")
        st.subheader("Information to verify")
        for item in result["missing_information"]:
            st.warning(item)

    st.subheader("Operator review")
    status = st.selectbox(
        "Review status",
        ["Pending review", "Reviewed — more information needed", "Reviewed — follow-up required", "Reviewed — escalation according to local protocol", "Reviewed — no further action recorded"],
        key=f"status_{result['case_id']}",
    )
    notes = st.text_area(
        "Operator notes (fictional demo only)",
        key=f"notes_{result['case_id']}",
        placeholder="Record what was verified; do not enter real personal information.",
    )
    if st.button("Save review", key=f"save_{result['case_id']}"):
        for case in st.session_state.cases:
            if case["case_id"] == result["case_id"]:
                case["review_status"] = status
                case["operator_notes"] = notes.strip()
                break
        st.success("Review saved in this browser session.")

st.divider()
st.header("3. Session case history")
st.caption("This history is temporary and resets when the Streamlit session is lost. Do not use real caller data.")
if st.session_state.cases:
    history = pd.DataFrame(st.session_state.cases)
    display_cols = ["case_id", "model_label", "review_recommendation", "queue", "location_clues", "review_status", "created_at"]
    for col in display_cols:
        if col not in history.columns:
            history[col] = ""
    st.dataframe(history[display_cols], use_container_width=True, hide_index=True)
    csv_data = history.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download session history CSV",
        data=csv_data,
        file_name="helpline_triage_session_history.csv",
        mime="text/csv",
    )
else:
    st.info("No cases have been analysed in this session yet.")

st.divider()
st.caption(
    "Limitations: small synthetic training dataset; limited evidence/location rules; "
    "Sarvam transcription can make errors; text similarity and COMI-LINGUA NER are not integrated. "
    "Every result requires human review. No autonomous dispatch is performed."
)
