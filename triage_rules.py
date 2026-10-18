"""Transparent, rule-based evidence layer for helpline transcripts.

This module is deliberately independent of the ML classifier in ``models2.0/``.
It surfaces human-readable *evidence* (matched safety-cue phrases, location
clues, and information that still needs verifying). It does NOT produce
probabilities and must never be presented as a validated risk model. Keeping it
separate from the ML prediction lets an operator cross-check the two.

Phrase lists are intentionally visible and easy to review/extend, and cover
English, Devanagari Hindi, and Romanized Hinglish safety cues.
"""
from __future__ import annotations

import re
from typing import Any

SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Light normalization for matching; the original transcript is preserved."""
    text = (text or "").replace("’", "'").replace("`", "'")
    return SPACE_RE.sub(" ", text.lower()).strip()


def phrase_regex(phrase: str) -> re.Pattern[str]:
    """Compile a phrase with flexible whitespace between words."""
    parts = [re.escape(part) for part in phrase.split()]
    return re.compile(r"(?<!\w)" + r"\s+".join(parts) + r"(?!\w)", re.IGNORECASE)


# Each item is a phrase, not a probability or confidence score.
CRITICAL_PHRASES = [
    "attacking me",
    "being attacked",
    "stabbed me",
    "trying to kill me",
    "threatened to kill me",
    "has a weapon",
    "weapon in his hand",
    "weapon in her hand",
    "trying to grab me",
    "trying to grab",
    "chaku dikha raha",
    "chaku dikha rahi",
    "bandook dikha raha",
    "bandook dikha rahi",
    "jaan se maarne",
    "jaan se maar dega",
    "jaan se maar degi",
    "pakadne ki koshish",
    "hamla karne wala",
    "help me now",
    "मुझ पर हमला हो रहा",
    "मेरे ऊपर हमला",
    "जान से मार",
    "चाकू दिखा",
    "बंदूक दिखा",
    "मुझे अभी बचाओ",
    "हमला करने वाला",
]

HIGH_PHRASES = [
    "threatening me",
    "he is threatening me",
    "she is threatening me",
    # Following / stalking variants (English)
    "following me",
    "is following me",
    "following me right now",
    "being followed",
    "followed by",
    "someone is following",
    "a man is following",
    "keeps following me",
    "stalking me",
    "chasing me",
    "cannot get away",
    "trapped here",
    # Feeling unsafe (English)
    "not safe",
    "don't feel safe",
    "do not feel safe",
    "feel unsafe",
    "feeling unsafe",
    "i feel unsafe",
    "afraid for my life",
    "forced me",
    # Help requests (English)
    "send help",
    "please send help",
    "please help me",
    "i need help",
    "need help now",
    # Romanized Hinglish
    "dhamki de raha",
    "dhamki de rahi",
    "peecha kar raha",
    "peecha kar rahi",
    "follow kar raha",
    "follow kar rahi",
    "follow ho raha",
    "follow ho rahi",
    "darr lag raha",
    "safe nahi",
    "zabardasti",
    "mujhe bachao",
    "madad bhejo",
    "madad karo",
    "madad chahiye",
    "help chahiye",
    # Devanagari Hindi
    "धमकी दे रहा",
    "धमकी दे रही",
    "मेरा पीछा कर रहा",
    "मेरा पीछा कर रही",
    "फॉलो कर रहा",
    "फॉलो कर रही",
    "डर लग रहा",
    "सुरक्षित महसूस नहीं",
    "सुरक्षित नहीं",
    "मदद भेज",
    "मदद चाहिए",
    "मदद करो",
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
SAFETY_NEGATIVE_PHRASES = {
    "not safe", "don't feel safe", "do not feel safe", "feel unsafe", "feeling unsafe",
    "i feel unsafe", "safe nahi", "सुरक्षित महसूस नहीं", "सुरक्षित नहीं",
}


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
        (r"\b(?:on|at|near)\s+(?:the\s+)?(?:road|street)\b", "road / street"),
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
    """Rule-based evidence layer. Does NOT produce ML probabilities."""
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

    following_terms = {
        "following me", "is following me", "following me right now", "being followed",
        "followed by", "someone is following", "a man is following", "keeps following me",
        "stalking me", "chasing me", "peecha kar raha", "peecha kar rahi",
        "follow kar raha", "follow kar rahi", "follow ho raha", "follow ho rahi",
        "मेरा पीछा कर रहा", "मेरा पीछा कर रही", "फॉलो कर रहा", "फॉलो कर रही",
    }
    help_terms = {
        "send help", "please send help", "please help me", "i need help", "need help now",
        "madad bhejo", "madad karo", "madad chahiye", "help chahiye", "मदद भेज", "मदद चाहिए", "मदद करो",
    }
    threat_terms = {
        "dhamki de raha", "dhamki de rahi", "threatening me",
        "he is threatening me", "she is threatening me", "धमकी दे रहा", "धमकी दे रही",
    }

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
        if any(term in high for term in following_terms):
            threat = "Possible stalking / following"
        elif any(term in high for term in threat_terms):
            threat = "Possible threat / coercion"
        elif any(term in high for term in help_terms):
            threat = "Possible request for urgent help"
        else:
            threat = "Possible immediate safety concern"
        suggested_queue = "Priority safety review"
        next_step = (
            "Verify whether danger is ongoing, establish the caller's current location, "
            "and let the operator decide the appropriate response."
        )
    elif medium:
        urgency = "Medium"
        threat = "Possible harassment / blackmail"
        suggested_queue = "Harassment / support review"
        next_step = "Clarify frequency, timing, evidence, current safety, and whether the situation has escalated."
    else:
        information_request = bool(find_active_phrases(normalized, INFORMATION_PHRASES))
        urgency = "Unclear"
        threat = (
            "Information request / no configured indicator detected"
            if information_request else "No configured threat phrase detected"
        )
        suggested_queue = "General intake — operator review"
        next_step = (
            "Review the original transcript and ask follow-up questions as needed. "
            "No matched phrase does not mean the caller is safe."
        )

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

    if not re.search(r"\b(?:safe|unsafe|danger|help|threat|scared|afraid|bachao|darr|surakshit|madad)\b", normalized):
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
