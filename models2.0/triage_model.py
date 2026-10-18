"""Inference helper for Streamlit. Load the saved model once and reuse it."""
from pathlib import Path
import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "helpline_triage_model.joblib"
LABEL_DESCRIPTIONS = {
    "PRIORITY_REVIEW": "Potentially urgent concern — prioritize prompt human assessment.",
    "SUPPORT_REVIEW": "Support/reporting need — assess context and route to suitable support.",
    "UNCLEAR_REVIEW": "Insufficient or ambiguous context — clarify with the caller; do not assume safe.",
}

_model = None

def load_model(model_path=None):
    """Load the trained pipeline once. Call from Streamlit's st.cache_resource."""
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run `python train_model.py` first."
        )
    return joblib.load(path)

def predict_transcript(transcript, model=None):
    """Return prediction, class scores, and a conservative review flag."""
    text = (transcript or "").strip()
    if not text:
        raise ValueError("Enter a transcript before analysing the call.")
    active_model = model or load_model()
    predicted = str(active_model.predict([text])[0])
    probs = active_model.predict_proba([text])[0]
    classes = list(active_model.named_steps["classifier"].classes_)
    scores = {str(label): float(score) for label, score in zip(classes, probs)}
    confidence = float(np.max(probs))

    # These are routing labels, not a medical or emergency severity prediction.
    # All predictions remain decision support; never auto-dispatch or close a case.
    review_required = True
    return {
        "predicted_label": predicted,
        "description": LABEL_DESCRIPTIONS.get(predicted, "Human review required."),
        "class_scores": scores,
        "model_confidence": confidence,
        "confidence_note": "Uncalibrated model score; not a probability of real-world correctness.",
        "review_required": review_required,
        "human_review_note": "Verify the transcript, current danger, caller location, and appropriate response with a trained operator.",
    }
