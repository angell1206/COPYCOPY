"""Sahaayak Triage — transparent, rules-based helpline transcript demo.

This prototype is intended for fictional/demo transcripts and human review only.
It is not a validated emergency-risk model and must not dispatch services.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------------------------------------------------
# Page setup and styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sahaayak Triage",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
      div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.07);
        padding: 0.8rem 1rem;
        border-radius: 0.75rem;
        border: 1px solid rgba(128, 128, 128, 0.16);
      }
      .small-muted { font-size: 0.88rem; opacity: 0.8; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛡️ Sahaayak Triage")
st.caption("AI-assisted Hinglish helpline transcript triage · operator-support prototype")
st.warning(
    "DEMO ONLY — use fictional transcripts. This version uses transparent keyword rules, "
    "not a trained urgency model. It can miss danger or flag harmless text. A trained human "
    "operator must verify the original transcript and follow the organisation's protocol. "
    "It must not independently dispatch emergency services."
)


# -----------------------------------------------------------------------------
# Transparent triage rules
# -----------------------------------------------------------------------------
SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Light normalization for matching; the original transcript is preserved."""
    text = (text or "").replace("’", "'").replace("`", "'")
    return SPACE_RE.sub(" ", text.lower()).strip()


def phrase_regex(phrase: str) -> re.Pattern[str]:
    """Compile a phrase with flexible whitespace between words."""
    parts = [re.escape(part) for part in phrase.split()]
    return re.compile(r"(?<!\w)" + r"\s+".join(parts) + r"(?!\w)", re.IGNORECASE)


# Each item is a phrase, not a probability or confidence score. These lists are
# intentionally visible and easy for the team to review/extend.
CRITICAL_PHRASES = [
    "attacking me",
    "being attacked",
    "stabbed me",
    "trying to kill me",
    "threatened to kill me",
    "has a weapon",
    "weapon in his hand",
    "weapon in her hand",
    "chaku dikha raha",
    "chaku dikha rahi",
    "bandook dikha raha",
    "bandook dikha rahi",
    "jaan se maarne",
    "jaan se maar dega",
    "jaan se maar degi",
    "help me now",
    "मुझ पर हमला हो रहा",
    "मेरे ऊपर हमला",
    "जान से मार",
    "चाकू दिखा",
    "बंदूक दिखा",
    "मुझे अभी बचाओ",
]

HIGH_PHRASES = [
    "threatening me",
    "he is threatening me",
    "she is threatening me",
    "following me",
    "stalking me",
    "trapped here",
    "not safe",
    "don't feel safe",
    "do not feel safe",
    "afraid for my life",
    "forced me",
    "dhamki de raha",
    "dhamki de rahi",
    "peecha kar raha",
    "peecha kar rahi",
    "darr lag raha",
    "safe nahi",
    "zabardasti",
    "mujhe bachao",
    "धमकी दे रहा",
    "धमकी दे रही",
    "मेरा पीछा कर रहा",
    "मेरा पीछा कर रही",
    "डर लग रहा",
    "सुरक्षित महसूस नहीं",
    "जबरदस्ती",
]

MEDIUM_PHRASES = [
    "keeps messaging",
    "unwanted messages",
    "harassing me",
    "blackmail",
    "being monitored",
    "baar baar message",
    "pareshan kar raha",
    "pareshan kar rahi",
    "tang kar raha",
    "tang kar rahi",
    "बार बार मैसेज",
    "परेशान कर रहा",
    "परेशान कर रही",
]

INFORMATION_PHRASES = [
    "need information",
    "want information",
    "how to report",
    "information about reporting",
    "reporting harassment",
    "legal advice",
]

NEGATION_RE = re.compile(
    r"(?:\bnot\b|\bnever\b|\bno\b|\bwithout\b|"
    r"\bdon't\b|\bdont\b|\bdo not\b|\bdidn't\b|\bdidnt\b|"
    r"\bdid not\b|\bdoesn't\b|\bdoesnt\b|\bdoes not\b|"
    r"\bisn't\b|\bisnt\b|\bis not\b|\bwasn't\b|\bwasnt\b|\bwas not\b|"
    r"\bhaven't\b|\bhavent\b|\bhave not\b|\bhasn't\b|\bhasnt\b|\bhas not\b|"
    r"\bnahi\b|\bnahin\b|\bmat\b|नहीं|नही)"
    r"(?:\s+\w+){0,3}\s*$",
    re.IGNORECASE,
)

# These negative phrases themselves indicate a safety concern, so they are not
# removed as negation of danger (e.g. "I do not feel safe").
SAFETY_NEGATIVE_PHRASES = {"not safe", "don't feel safe", "do not feel safe", "safe nahi", "सुरक्षित महसूस नहीं"}


def phrase_is_negated(text: str, start: int, phrase: str) -> bool:
    if phrase in SAFETY_NEGATIVE_PHRASES:
        return False
    prefix = text[max(0, start - 70) : start]
    return NEGATION_RE.search(prefix) is not None


def find_active_phrases(text: str, phrases: list[str]) -> list[str]:
    """Return matched phrases unless a nearby simple negation precedes them."""
    found: list[tuple[int, str]] = []
    for phrase in phrases:
        pattern = phrase_regex(phrase)
        for match in pattern.finditer(text):
            if not phrase_is_negated(text, match.start(), phrase):
                found.append((match.start(), phrase))
                break
    # Keep transcript order while removing duplicates.
    found.sort(key=lambda item: item[0])
    output: list[str] = []
    for _, phrase in found:
        if phrase not in output:
            output.append(phrase)
    return output


def extract_location_clues(text: str) -> list[str]:
    """Extract only supported place clues, not exact or verified locations."""
    patterns = [
        (r"\b(?:near|at|outside|inside|by)\s+(?:the\s+)?metro station\b", "metro station"),
        (r"\bmetro station\s+ke\s+(?:paas|bahar)\b", "metro station"),
        (r"\b(?:near|at|outside|inside|by)\s+(?:the\s+)?bus stop\b", "bus stop"),
        (r"\bbus stop\s+ke\s+(?:paas|bahar)\b", "bus stop"),
        (r"\b(?:at|inside|outside|near)\s+(?:the\s+)?college\b", "college"),
        (r"\bcollege\s+ke\s+(?:paas|bahar)\b", "college"),
        (r"\b(?:at|inside|outside|near|in)\s+(?:the\s+)?hostel\b", "hostel"),
        (r"\bhostel\s+mein\b", "hostel"),
        (r"\b(?:at|near|outside)\s+(?:my\s+)?home\b|\bhome\s+ke\s+paas\b", "home"),
        (r"\bghar\s+(?:par|pe|ke paas|ke bahar)\b", "home"),
        (r"\b(?:at|inside|outside|near)\s+(?:the\s+)?market\b", "market"),
        (r"\bmarket\s+mein\b", "market"),
        (r"\b(?:at|inside|outside|near)\s+(?:the\s+)?hospital\b", "hospital"),
        (r"\b(?:on|at)\s+campus\b", "campus"),
        (r"मेट्रो स्टेशन\s+के\s+(?:पास|बाहर)", "metro station"),
        (r"बस स्टॉप\s+के\s+(?:पास|बाहर)", "bus stop"),
        (r"कॉलेज\s+के\s+(?:पास|बाहर)", "college"),
        (r"हॉस्टल\s+में", "hostel"),
        (r"घर\s+(?:पर|पे|के पास|के बाहर)", "home"),
        (r"मार्केट\s+में", "market"),
    ]
    normalized = normalize_text(text)
    clues: list[str] = []
    for pattern, label in patterns:
        if re.search(pattern, normalized) and label not in clues:
            clues.append(label)
    return clues


def analyze_transcript(text: str) -> dict[str, Any]:
    """Rule-based baseline. This function does not produce ML probabilities."""
    original = (text or "").strip()
    normalized = normalize_text(original)

    if not normalized:
        return {
            "urgency": "Unclear",
            "threat": "No transcript provided",
            "location_clues": [],
            "evidence": [],
            "missing": ["A transcript is required before analysis."],
            "suggested_queue": "Intake review",
            "next_step": "Enter a fictional transcript and ask an operator to review it.",
            "matched_phrases": [],
            "information_request": False,
        }

    critical = find_active_phrases(normalized, CRITICAL_PHRASES)
    high = find_active_phrases(normalized, HIGH_PHRASES)
    medium = find_active_phrases(normalized, MEDIUM_PHRASES)

    # Include a few single-term variants, but do not treat every isolated mention
    # of a dangerous object as proof of an ongoing emergency.
    if not critical:
        for pattern, label in [
            (r"\b(?:weapon|gun|knife)\s+(?:is\s+)?(?:pointed|aimed)\s+at\s+me\b", "weapon pointed at me"),
            (r"\b(?:chaku|bandook)\s+(?:hai|nikala|nikal liya)\b", "possible weapon mention in Hinglish"),
        ]:
            if re.search(pattern, normalized):
                critical.append(label)

    all_active = critical + high + medium
    location_clues = extract_location_clues(normalized)

    if critical:
        urgency = "Critical"
        threat = "Possible immediate physical danger"
        suggested_queue = "Priority safety review"
        next_step = (
            "Promptly review the original transcript and current safety status; "
            "follow the organisation's established emergency protocol if warranted."
        )
    elif high:
        urgency = "High"
        if any(term in high for term in ["following me", "stalking me", "peecha kar raha", "peecha kar rahi"]):
            threat = "Possible stalking / following"
        elif any(term in high for term in ["dhamki de raha", "dhamki de rahi", "threatening me", "he is threatening me", "she is threatening me"]):
            threat = "Possible threat / coercion"
        else:
            threat = "Possible immediate safety concern"
        suggested_queue = "Priority safety review"
        next_step = "Verify whether danger is ongoing, establish the caller's current location, and let the operator decide the appropriate response."
    elif medium:
        urgency = "Medium"
        threat = "Possible harassment / blackmail"
        suggested_queue = "Harassment / support review"
        next_step = "Clarify frequency, timing, evidence, current safety, and whether the situation has escalated."
    else:
        information_request = bool(find_active_phrases(normalized, INFORMATION_PHRASES))
        urgency = "Unclear"
        threat = "Information request / no configured indicator detected" if information_request else "No configured threat phrase detected"
        suggested_queue = "General intake — operator review"
        next_step = "Review the original transcript and ask follow-up questions as needed. No matched phrase does not mean the caller is safe."

    # Evidence is a list of configured indicators, not proof that an event occurred.
    evidence = list(dict.fromkeys(all_active))
    if not evidence and re.search(r"\b(?:not safe|don't feel safe|do not feel safe|safe nahi)\b", normalized):
        evidence = [phrase for phrase in SAFETY_NEGATIVE_PHRASES if phrase in normalized]

    missing: list[str] = []
    if not location_clues:
        missing.append("Current location is not identified in the transcript; ask and verify it.")
    else:
        missing.append("The location clue is unverified; confirm the caller's current location.")

    if not re.search(r"\b(?:now|right now|currently|abhi|iss waqt|is waqt)\b", normalized):
        missing.append("Whether the concern is happening right now is unclear.")

    if len(normalized.split()) < 7:
        missing.append("The transcript is short; additional context may be needed.")

    if not re.search(r"\b(?:safe|unsafe|danger|help|threat|scared|afraid|bachao|darr|surakshit)\b", normalized):
        missing.append("The caller's immediate safety status needs clarification.")

    return {
        "urgency": urgency,
        "threat": threat,
        "location_clues": location_clues,
        "evidence": evidence,
        "missing": missing,
        "suggested_queue": suggested_queue,
        "next_step": next_step,
        "matched_phrases": evidence,
        "information_request": bool(find_active_phrases(normalized, INFORMATION_PHRASES)),
    }


# -----------------------------------------------------------------------------
# Sample data: explicitly fictional examples
# -----------------------------------------------------------------------------
SAMPLE_TRANSCRIPTS = {
    "Following near metro": "A man is following me near the metro station. He is threatening me and I am scared right now.",
    "Threat at home": "My partner is threatening me at home. I do not feel safe right now.",
    "Repeated harassment": "Someone keeps sending unwanted messages and following me after college.",
    "Information request": "I need information about reporting harassment. There is no immediate danger that I know of.",
    "Hinglish example": "Woh mera peecha kar raha hai aur dhamki de raha hai. Mujhe darr lag raha hai abhi.",
    "Negation test": "He is not following me and he has not threatened me. I am asking how to report an old incident.",
    "Short / ambiguous": "Help please. I am scared.",
}


def make_case(case_id: str, transcript: str, created_at: str, source: str = "Analysed transcript") -> dict[str, Any]:
    analysis = analyze_transcript(transcript)
    return {
        "case_id": case_id,
        "transcript": transcript,
        "urgency": analysis["urgency"],
        "threat": analysis["threat"],
        "location": ", ".join(analysis["location_clues"]) if analysis["location_clues"] else "Not identified",
        "suggested_queue": analysis["suggested_queue"],
        "created_at": created_at,
        "review_status": "Pending review",
        "operator_notes": "",
        "source": source,
    }


if "cases" not in st.session_state:
    seed_texts = [
        SAMPLE_TRANSCRIPTS["Following near metro"],
        SAMPLE_TRANSCRIPTS["Threat at home"],
        SAMPLE_TRANSCRIPTS["Repeated harassment"],
        SAMPLE_TRANSCRIPTS["Information request"],
    ]
    st.session_state.cases = [
        make_case(f"DEMO-{i:03d}", text, "Fictional example", source="Built-in fictional example")
        for i, text in enumerate(seed_texts, start=1)
    ]
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""
if "last_case_id" not in st.session_state:
    st.session_state.last_case_id = None
if "case_counter" not in st.session_state:
    st.session_state.case_counter = 1


# -----------------------------------------------------------------------------
# Related-complaint matching: lexical similarity only
# -----------------------------------------------------------------------------
def find_related_complaints(
    transcript: str,
    cases: list[dict[str, Any]],
    exclude_case_id: str | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    columns = ["Case ID", "Text similarity", "Previous transcript", "Previous triage label"]
    candidates = [
        case for case in cases
        if case.get("case_id") != exclude_case_id and str(case.get("transcript", "")).strip()
    ]
    if not transcript.strip() or not candidates:
        return pd.DataFrame(columns=columns)

    documents = [transcript] + [case["transcript"] for case in candidates]
    try:
        word_matrix = TfidfVectorizer(
            ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True,
            max_features=30000,
        ).fit_transform(documents)
        char_matrix = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), strip_accents="unicode",
            sublinear_tf=True, max_features=40000,
        ).fit_transform(documents)
        word_sim = cosine_similarity(word_matrix[0:1], word_matrix[1:]).ravel()
        char_sim = cosine_similarity(char_matrix[0:1], char_matrix[1:]).ravel()
        # Simple blended lexical similarity; it is not semantic identity matching.
        scores = (0.5 * word_sim) + (0.5 * char_sim)
    except ValueError:
        return pd.DataFrame(columns=columns)

    rows = []
    for case, score in zip(candidates, scores):
        rows.append({
            "Case ID": case.get("case_id", "Unknown"),
            "Text similarity": f"{float(score) * 100:.1f}%",
            "_similarity": float(score),
            "Previous transcript": case.get("transcript", ""),
            "Previous triage label": case.get("urgency", "Unclear"),
        })
    rows.sort(key=lambda row: row["_similarity"], reverse=True)
    return pd.DataFrame([
        {key: value for key, value in row.items() if key != "_similarity"}
        for row in rows[:top_n]
    ], columns=columns)


# -----------------------------------------------------------------------------
# Sidebar controls and sample transcript loader
# -----------------------------------------------------------------------------
st.sidebar.title("Operator Console")
st.sidebar.caption("Use fictional examples only.")

sample_choice = st.sidebar.selectbox(
    "Load an example transcript",
    ["Custom transcript", *SAMPLE_TRANSCRIPTS.keys()],
    key="sample_choice",
)
if st.session_state.get("_last_sample_choice") != sample_choice:
    st.session_state.transcript_input = SAMPLE_TRANSCRIPTS.get(sample_choice, "")
    st.session_state._last_sample_choice = sample_choice

st.sidebar.divider()
st.sidebar.markdown("**Current prototype**")
st.sidebar.write("• Transparent keyword rules")
st.sidebar.write("• Evidence and missing-detail flags")
st.sidebar.write("• TF-IDF text-similarity baseline")
st.sidebar.write("• Session-only case history")
st.sidebar.caption("No speech-to-text, BHASHINI API, trained ML classifier, or semantic embedding model is connected yet.")


# -----------------------------------------------------------------------------
# Main input form
# -----------------------------------------------------------------------------
st.header("1. Incoming complaint")
transcript = st.text_area(
    "Call transcript",
    key="transcript_input",
    height=155,
    placeholder="Enter a fictional transcript in English or Hinglish...",
    help="Original text is kept for operator review. Matching is based on a limited configured phrase list.",
)

col_analyse, col_clear = st.columns([1, 1])
with col_analyse:
    analyze_clicked = st.button("Analyse transcript", type="primary", use_container_width=True)
with col_clear:
    clear_current = st.button("Clear transcript", use_container_width=True)

if clear_current:
    st.session_state.transcript_input = ""
    st.session_state.last_analysis = None
    st.session_state.last_transcript = ""
    st.session_state.last_case_id = None
    st.rerun()

if analyze_clicked:
    if not transcript.strip():
        st.warning("Enter a fictional transcript before analysing.")
    else:
        result = analyze_transcript(transcript)
        case_id = f"HL-{st.session_state.case_counter:04d}"
        st.session_state.case_counter += 1
        created_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        new_case = make_case(case_id, transcript.strip(), created_at)
        # Store the analysis from this exact input in the case record as well.
        new_case["urgency"] = result["urgency"]
        new_case["threat"] = result["threat"]
        new_case["location"] = ", ".join(result["location_clues"]) if result["location_clues"] else "Not identified"
        new_case["suggested_queue"] = result["suggested_queue"]
        st.session_state.cases.append(new_case)
        st.session_state.last_analysis = result
        st.session_state.last_transcript = transcript.strip()
        st.session_state.last_case_id = case_id


# -----------------------------------------------------------------------------
# Results for latest analysed transcript
# -----------------------------------------------------------------------------
if st.session_state.last_analysis is not None:
    result = st.session_state.last_analysis
    case_id = st.session_state.last_case_id
    st.divider()
    st.header("2. Triage summary")
    st.caption(f"Case ID: {case_id} · Rule-based baseline · Not an ML confidence score")

    m1, m2, m3 = st.columns(3)
    m1.metric("Review priority", result["urgency"])
    m2.metric("Configured indicators", len(result["matched_phrases"]))
    m3.metric("Location clues", len(result["location_clues"]))

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Detected information")
        st.write("**Possible concern:**", result["threat"])
        st.write("**Suggested review queue:**", result["suggested_queue"])
        st.write("**Location clue(s):**", ", ".join(result["location_clues"]) if result["location_clues"] else "Not identified")
        st.caption("Any location shown is a text clue, not a verified current location or an address.")

        st.markdown("**Matched text indicators**")
        if result["evidence"]:
            for phrase in result["evidence"]:
                st.markdown(f"- `{phrase}`")
        else:
            st.write("No configured indicator phrase matched.")
        st.caption("A matched phrase is a cue for human review, not proof that an event occurred.")

    with c2:
        st.subheader("Information to verify")
        for item in result["missing"]:
            st.warning(item)
        st.markdown("**Suggested operator next step**")
        st.write(result["next_step"])
        st.caption("The operator should assess the full context and follow local procedures.")

    st.divider()
    st.header("3. Operator review")
    st.write("Record a review outcome for this demo case. This does not contact services or dispatch help.")
    current_case = next((case for case in st.session_state.cases if case["case_id"] == case_id), None)
    if current_case is not None:
        status_key = f"review_status_{case_id}"
        notes_key = f"review_notes_{case_id}"
        if status_key not in st.session_state:
            st.session_state[status_key] = current_case.get("review_status", "Pending review")
        if notes_key not in st.session_state:
            st.session_state[notes_key] = current_case.get("operator_notes", "")
        status = st.selectbox(
            "Review outcome",
            [
                "Pending review",
                "Reviewed — more information needed",
                "Reviewed — follow-up required",
                "Reviewed — no further action recorded",
                "Escalated according to local protocol",
            ],
            key=status_key,
        )
        notes = st.text_area(
            "Operator notes (fictional demo only)",
            key=notes_key,
            placeholder="Record what was verified, without adding real personal data.",
        )
        if st.button("Save review outcome", key=f"save_review_{case_id}"):
            for case in st.session_state.cases:
                if case["case_id"] == case_id:
                    case["review_status"] = status
                    case["operator_notes"] = notes.strip()
                    break
            st.success("Review outcome saved to this browser session.")

    st.divider()
    st.header("4. Related complaint matching")
    st.write("These are textually similar transcripts from the current session and built-in fictional examples.")
    matches = find_related_complaints(
        st.session_state.last_transcript,
        st.session_state.cases,
        exclude_case_id=st.session_state.last_case_id,
    )
    if matches.empty:
        st.info("No comparable transcripts are available yet.")
    else:
        st.dataframe(matches, use_container_width=True, hide_index=True)
        st.caption(
            "Text similarity is a rough lexical comparison, not proof of a shared caller, person, or incident. "
            "Any potential link must be checked by an authorised human."
        )


# -----------------------------------------------------------------------------
# Case history and export
# -----------------------------------------------------------------------------
st.divider()
st.header("5. Session case history")
st.caption("History is kept in Streamlit session state only. It is not a permanent database.")

if st.session_state.cases:
    history = pd.DataFrame(st.session_state.cases)
    display_columns = [
        "case_id", "urgency", "threat", "location", "suggested_queue",
        "review_status", "created_at", "source",
    ]
    for col in display_columns:
        if col not in history.columns:
            history[col] = ""
    st.dataframe(history[display_columns], use_container_width=True, hide_index=True)

    export_columns = [
        "case_id", "transcript", "urgency", "threat", "location",
        "suggested_queue", "created_at", "review_status", "operator_notes", "source",
    ]
    for col in export_columns:
        if col not in history.columns:
            history[col] = ""
    csv_bytes = history[export_columns].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download session history as CSV",
        data=csv_bytes,
        file_name="sahaayak_session_history.csv",
        mime="text/csv",
        help="Exports the current browser session, including fictional transcripts and review notes.",
    )

    if st.button("Reset session history to built-in examples"):
        seed_texts = [
            SAMPLE_TRANSCRIPTS["Following near metro"],
            SAMPLE_TRANSCRIPTS["Threat at home"],
            SAMPLE_TRANSCRIPTS["Repeated harassment"],
            SAMPLE_TRANSCRIPTS["Information request"],
        ]
        st.session_state.cases = [
            make_case(f"DEMO-{i:03d}", text, "Fictional example", source="Built-in fictional example")
            for i, text in enumerate(seed_texts, start=1)
        ]
        st.session_state.last_analysis = None
        st.session_state.last_transcript = ""
        st.session_state.last_case_id = None
        st.session_state.case_counter = 1
        st.rerun()
else:
    st.info("No cases are recorded in this session yet.")

st.divider()
st.caption(
    "Prototype limitations: rules-based triage, limited phrase coverage, basic location clues, "
    "TF-IDF similarity, and temporary session history. Trained ML urgency classification, "
    "BHASHINI, speech-to-text, Sentence Transformers, multilingual evaluation, and offline validation "
    "are not implemented in this version."
)
