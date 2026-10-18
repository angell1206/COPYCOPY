"""Inference helper for Streamlit. Load the saved model once and reuse it.

The pipeline (``helpline_triage_model.joblib``) is a FeatureUnion of word
(1-2 gram) and character_wb (3-5 gram) TF-IDF features feeding a class-balanced
Logistic Regression. It is trained by ``train_model.py`` on
``data/calls_train_v2.csv`` plus the curated ``augmentation_v2_1.csv`` set.
Honest, leakage-free metrics are produced by ``evaluate_model.py``.

The three outputs are routing labels only, with uncalibrated model scores. This
is decision support, not a validated emergency-risk model, and it must never
auto-dispatch, escalate, or close a case without human verification.
"""
from pathlib import Path
import warnings
import joblib
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning

MODEL_PATH = Path(__file__).resolve().parent / "helpline_triage_model.joblib"
# Retrain with train_model.py whenever the local scikit-learn differs from the
# version the committed artefact was pickled with. load_model() silences the
# resulting InconsistentVersionWarning defensively.
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
            f"Trained model not found at {path}. Expected the committed artefact "
            f"'helpline_triage_model.joblib' inside the models2.0 folder."
        )
    # The artefact may have been pickled with a slightly different scikit-learn
    # version; predictions are still valid for this demo, so silence the noisy
    # version warning rather than surfacing it to operators.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        return joblib.load(path)

def _model_classes(active_model):
    """Return class labels, whatever the final estimator step is named."""
    if hasattr(active_model, "classes_"):
        return list(active_model.classes_)
    return list(active_model.steps[-1][1].classes_)

def predict_transcript(transcript, model=None):
    """Return prediction, class scores, and a conservative review flag."""
    text = (transcript or "").strip()
    if not text:
        raise ValueError("Enter a transcript before analysing the call.")
    active_model = model or load_model()
    predicted = str(active_model.predict([text])[0])
    probs = active_model.predict_proba([text])[0]
    classes = _model_classes(active_model)
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
