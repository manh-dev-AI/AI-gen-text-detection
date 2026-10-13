"""FastAPI service for the Pipeline 1 Word TF-IDF model."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "pipeline1"
VECTORIZER_PATH = MODEL_DIR / "word_vectorizer.joblib"
CLASSIFIER_PATH = MODEL_DIR / "best_model.joblib"
METRICS_PATH = MODEL_DIR / "pipeline1_metrics.json"

app = FastAPI(title="AI Text Detector API", version="0.1.0")

# Local prototype: lets the unpacked Chrome Extension call this localhost API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=20, max_length=100_000)


class ArtifactLoadError(RuntimeError):
    """Raised when the local Pipeline 1 artifacts are unavailable or invalid."""


@lru_cache(maxsize=1)
def load_artifacts() -> tuple[Any, Any]:
    missing = [path.name for path in (VECTORIZER_PATH, CLASSIFIER_PATH) if not path.exists()]
    if missing:
        raise ArtifactLoadError(
            f"Missing model artifact(s) in {MODEL_DIR}: {', '.join(missing)}"
        )

    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        classifier = joblib.load(CLASSIFIER_PATH)
    except Exception as error:  # joblib reports the underlying incompatible dependency
        raise ArtifactLoadError(f"Could not load Pipeline 1 artifacts: {error}") from error

    if not hasattr(vectorizer, "transform") or not hasattr(classifier, "predict"):
        raise ArtifactLoadError("Artifacts do not contain a scikit-learn vectorizer and classifier.")
    return vectorizer, classifier


def ai_score(classifier: Any, matrix: Any, predicted_label: int) -> tuple[float, str]:
    """Return an AI score and explain whether it is a calibrated probability."""
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(matrix)[0]
        classes = list(classifier.classes_)
        ai_index = classes.index(1)
        return float(probabilities[ai_index]), "probability"

    if hasattr(classifier, "decision_function"):
        # LinearSVC exposes a margin, not a calibrated probability. Sigmoid only maps it to 0-1 for UI display.
        margin = float(np.ravel(classifier.decision_function(matrix))[0])
        score = 1.0 / (1.0 + np.exp(-np.clip(margin, -30, 30)))
        return float(score), "sigmoid_decision_score"

    return float(predicted_label), "hard_label"


def model_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "model_directory": str(MODEL_DIR),
        "vectorizer_file": VECTORIZER_PATH.name,
        "classifier_file": CLASSIFIER_PATH.name,
    }
    if METRICS_PATH.exists():
        try:
            records = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            if records:
                metadata["selected_model"] = records[0].get("Model")
                metadata["validation_macro_f1"] = records[0].get("Macro-F1")
        except (json.JSONDecodeError, OSError):
            metadata["selected_model"] = "unknown"
    return metadata


@app.get("/health")
def health_check() -> dict[str, Any]:
    try:
        vectorizer, classifier = load_artifacts()
        return {
            "status": "ready",
            **model_metadata(),
            "vocabulary_size": len(getattr(vectorizer, "vocabulary_", {})),
            "classifier_type": type(classifier).__name__,
        }
    except ArtifactLoadError as error:
        return {"status": "unavailable", "detail": str(error), **model_metadata()}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    text = request.text.strip()
    try:
        vectorizer, classifier = load_artifacts()
    except ArtifactLoadError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    matrix = vectorizer.transform([text])
    if matrix.nnz == 0:
        raise HTTPException(
            status_code=422,
            detail="Text has no vocabulary terms after Pipeline 1 preprocessing.",
        )

    predicted_label = int(classifier.predict(matrix)[0])
    score, score_type = ai_score(classifier, matrix, predicted_label)
    return {
        "label": "AI-generated" if predicted_label == 1 else "Human",
        "predicted_class": predicted_label,
        "ai_score": round(score, 6),
        "human_score": round(1.0 - score, 6),
        "score_type": score_type,
        "text_length": len(text),
        "nonzero_features": int(matrix.nnz),
        **model_metadata(),
    }
