# AI-Based Help Line Triage
### Ten Seconds to Read a Call: Helpline Triage in Hinglish

**Team:** Angel's Angels  
**Institution:** Bharati Vidyapeeth's College of Engineering, New Delhi

## Problem Statement

Helpline calls may contain incomplete information, noisy transcripts, and a mixture of Hindi and English. These challenges can make it difficult for operators to quickly identify potentially urgent situations and extract the information needed to respond appropriately.

## Project Overview

AI-Based Help Line Triage is an operator-support prototype that analyses a transcript and presents a structured review summary. The current version uses transparent, manually configured keyword rules for triage cues and TF-IDF text similarity to compare transcripts.

It is a decision-support demonstration—not a validated emergency-risk model. All results require human verification.

## Current Features

- **Transcript input:** Analyse fictional English or Hinglish text through a Streamlit interface.
- **Rule-based triage indicators:** Apply configured phrase patterns to flag possible safety concerns.
- **Evidence phrases:** Show which configured phrases matched the transcript.
- **Location clues:** Identify a limited set of supported place mentions; these are not verified addresses or confirmed current locations.
- **Missing-information prompts:** Highlight details an operator may need to verify, such as current location or immediate safety status.
- **Related transcript matching:** Compare current-session and example transcripts using word-level and character-level TF-IDF similarity.
- **Human review notes:** Record a review status and notes in the current Streamlit session.
- **Case history export:** Download the current session history as a CSV file.
- **Fictional examples:** Load built-in examples to demonstrate different inputs and edge cases.

## Workflow

1. Enter a fictional call transcript or load a built-in example.
2. Apply lightweight text normalization for phrase matching while retaining the original transcript.
3. Detect configured urgency-related indicators and possible concern categories.
4. Display evidence phrases, location clues, and information that needs verification.
5. Present a suggested review queue and a suggested next step for the operator.
6. Allow a human reviewer to record a review outcome and inspect textually similar examples.

## Technology Stack

- **Python**
- **Streamlit** for the web interface
- **pandas** for case-history tables and CSV export
- **scikit-learn** for TF-IDF vectorization and cosine similarity
- **Regular expressions** for the current transparent keyword-rule baseline
- **GitHub** for version control

Install the packages in `requirements.txt`, then launch the app with:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Known Limitations

- Urgency triage is **rule-based**; there is no trained ML urgency classifier in this version.
- Phrase coverage is limited, so the rules can miss risks or flag harmless text. Negation and context handling are not comprehensive.
- Location extraction detects only a limited set of configured patterns and does not verify caller location.
- TF-IDF measures lexical similarity; it cannot establish that reports refer to the same person or incident.
- Case history and review notes are stored in Streamlit session state, not a permanent database.
- Speech-to-text, BHASHINI APIs, Sentence Transformers, persistent storage, and a full labelled-data evaluation pipeline are not integrated in this version.
- Offline operation and performance have not yet been validated.

## Safety

- Use fictional transcripts for demonstrations and testing.
- The prototype assists review and must not replace a trained operator or local procedures.
- It must not independently dispatch emergency services.
- Do not enter real caller recordings, personally identifying details, or confidential case information into a public demo.
- Report only evaluation metrics actually measured on an appropriate test set. Do not describe rule indicators as model confidence or validated risk probabilities.

## Project Status

The current repository contains an interactive, rules-based prototype. Training and evaluating an urgency classifier, expanding Hinglish coverage, improving extraction, and validating offline operation are future work.

## License

A license has not yet been selected. Confirm the team's and hackathon's requirements before choosing one.
