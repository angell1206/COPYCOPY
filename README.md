# helpline-traige
### AI-Assisted Hinglish Helpline Call Triage

**Project:** Ten Seconds to Read a Call: Helpline Triage in Hinglish  
**Team:** Angel's Angels  
**Institution:** Bharati Vidyapeeth's College of Engineering, New Delhi

## Problem Statement

Helpline calls may contain incomplete information, noisy transcripts, and a mixture of Hindi and English. These challenges can make it difficult for operators to quickly identify potentially urgent situations and extract the information needed to respond appropriately.

## Proposed Solution

Sahaayak Triage is a prototype designed to assist helpline operators by analysing call transcripts and presenting relevant information in a structured review interface.

The proposed system aims to:

- Classify calls into urgency-related review categories.
- Identify incident details and relevant evidence from transcripts.
- Highlight missing information that may require follow-up questions.
- Support multilingual and Hinglish input.
- Present structured recommendations for human operator review.

## Planned Workflow

1. Receive a call transcript.
2. Preprocess the text while preserving important context.
3. Estimate urgency and identify the incident type.
4. Extract available details and flag missing information.
5. Display the results in an operator-facing dashboard.
6. Leave final decisions and emergency actions to authorised human operators.

## Planned Technology Stack

- **Language:** Python
- **Machine Learning:** scikit-learn
- **Data Processing:** pandas
- **Dashboard:** Streamlit
- **Development Environment:** Google Colab and, if required, a local environment
- **Version Control:** Git and GitHub

These are planned technologies; the final stack will depend on implementation progress and hackathon rules.

## Evaluation Plan

The prototype will be evaluated using labelled test examples and relevant metrics, including:

- Urgent-class recall and false-positive rate.
- Macro F1-score for classification.
- Accuracy of extracted incident details.
- Inference time and reliability across Hinglish variations.

Results will be reported from actual tests rather than assumed in advance.

## Safety and Limitations

- The prototype is intended to assist human review, not replace trained operators.
- It will not autonomously dispatch emergency services.
- Test examples will be fictional or appropriately licensed and prepared data.
- Predictions may be incorrect, especially for ambiguous or incomplete transcripts.
- Any claimed multilingual, offline, or real-time capability must be verified through testing.

## Project Status

**Current stage:** Repository and project planning.

Implementation, datasets, model performance, and dashboard functionality are not yet confirmed.

## License

To be decided in accordance with the project requirements and hackathon rules.
