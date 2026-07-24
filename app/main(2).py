import os
import sys
from typing import Any, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.preprocessing import TextPreprocessor, nlp

sys.modules["__main__"].TextPreprocessor = TextPreprocessor
sys.modules["__main__"].nlp = nlp

MODEL_PATH = os.getenv("MODEL_PATH", "model/svm_pipeline.joblib")
REVIEW_MARGIN_THRESHOLD = float(os.getenv("REVIEW_MARGIN_THRESHOLD", "0.10"))
TOP_K_CLASSES = int(os.getenv("TOP_K_CLASSES", "3"))
TOP_K_KEYWORDS = int(os.getenv("TOP_K_KEYWORDS", "8"))

app = FastAPI(
    title="Intelligent Support Ticket Router API",
    version="2.0.0",
    description=(
        "FastAPI service for routing support tickets with a TF-IDF + Linear SVM "
        "pipeline and selective human review for low-margin predictions."
    ),
)

model = None


class TicketRequest(BaseModel):
    text: str = Field(..., min_length=1)


class BatchTicketRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)


class ClassCandidate(BaseModel):
    queue: str
    decision_score: float


class PredictionResponse(BaseModel):
    prediction: str
    score: Optional[float] = None
    margin: Optional[float] = None
    review_threshold: float
    needs_review: bool
    routing_status: str
    top_classes: List[ClassCandidate]
    keywords: List[str]
    review_message: Optional[str] = None


class BatchPredictionItem(PredictionResponse):
    text: str


class BatchPredictionResponse(BaseModel):
    predictions: List[BatchPredictionItem]


class ReviewDecisionRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model_prediction: str
    selected_queue: str
    margin: Optional[float] = None
    reviewer_note: Optional[str] = None


class ReviewDecisionResponse(BaseModel):
    status: str
    selected_queue: str
    model_prediction: str
    was_overridden: bool
    message: str


def _get_classifier(pipeline: Any) -> Any:
    if hasattr(pipeline, "steps") and pipeline.steps:
        return pipeline.steps[-1][1]
    return pipeline


def _get_vectorizer(pipeline: Any) -> Optional[Any]:
    if not hasattr(pipeline, "steps"):
        return None
    for _, step in reversed(pipeline.steps[:-1]):
        if hasattr(step, "get_feature_names_out"):
            return step
    return None


def _transform_until_vectorizer(pipeline: Any, texts: List[str]) -> Optional[Any]:
    if not hasattr(pipeline, "steps"):
        return None

    transformed: Any = texts
    found_vectorizer = False

    for _, step in pipeline.steps[:-1]:
        if not hasattr(step, "transform"):
            return None
        transformed = step.transform(transformed)
        if hasattr(step, "get_feature_names_out"):
            found_vectorizer = True
            break

    return transformed if found_vectorizer else None


def _decision_scores(texts: List[str]) -> np.ndarray:
    if model is None:
        raise RuntimeError("Model is not loaded")
    if not hasattr(model, "decision_function"):
        raise RuntimeError("Loaded model does not expose decision_function")

    scores = np.asarray(model.decision_function(texts), dtype=float)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    return scores


def _class_labels() -> np.ndarray:
    if model is None:
        raise RuntimeError("Model is not loaded")

    classifier = _get_classifier(model)
    classes = getattr(classifier, "classes_", None)
    if classes is None:
        classes = getattr(model, "classes_", None)
    if classes is None:
        raise RuntimeError("Could not read class labels from the loaded model")
    return np.asarray(classes, dtype=object)


def _top_candidates(row_scores: np.ndarray, classes: np.ndarray, top_k: int) -> List[dict]:
    top_k = max(1, min(top_k, len(classes)))
    indices = np.argsort(row_scores)[::-1][:top_k]
    return [
        {"queue": str(classes[index]), "decision_score": float(row_scores[index])}
        for index in indices
    ]


def _prediction_margin(row_scores: np.ndarray) -> Optional[float]:
    if row_scores.size < 2:
        return None
    sorted_scores = np.sort(row_scores)
    return float(sorted_scores[-1] - sorted_scores[-2])


def _extract_keywords(text: str, predicted_class_index: int, top_k: int) -> List[str]:
    if model is None:
        return []

    vectorizer = _get_vectorizer(model)
    classifier = _get_classifier(model)
    if vectorizer is None or not hasattr(classifier, "coef_"):
        return []

    try:
        features = _transform_until_vectorizer(model, [text])
        if features is None or features.shape[0] == 0:
            return []

        feature_names = np.asarray(vectorizer.get_feature_names_out(), dtype=object)
        coefficients = np.asarray(classifier.coef_, dtype=float)
        if coefficients.ndim != 2:
            return []

        class_coefficients = (
            coefficients[0]
            if coefficients.shape[0] == 1
            else coefficients[predicted_class_index]
        )

        row = features.getrow(0) if hasattr(features, "getrow") else features[0]
        indices = np.asarray(row.indices, dtype=int)
        values = np.asarray(row.data, dtype=float)
        if indices.size == 0:
            return []

        contributions = values * class_coefficients[indices]
        positive_mask = contributions > 0
        indices = indices[positive_mask]
        contributions = contributions[positive_mask]
        if indices.size == 0:
            return []

        order = np.argsort(contributions)[::-1][:top_k]
        return [str(feature_names[indices[index]]) for index in order]
    except Exception:
        return []


def _predict_one(text: str) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        classes = _class_labels()
        scores = _decision_scores([text])[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    ranked_indices = np.argsort(scores)[::-1]
    predicted_index = int(ranked_indices[0])
    prediction = str(classes[predicted_index])
    margin = _prediction_margin(scores)
    needs_review = margin is None or margin < REVIEW_MARGIN_THRESHOLD

    return {
        "prediction": prediction,
        "score": float(scores[predicted_index]),
        "margin": margin,
        "review_threshold": REVIEW_MARGIN_THRESHOLD,
        "needs_review": needs_review,
        "routing_status": "needs_review" if needs_review else "automatic",
        "top_classes": _top_candidates(scores, classes, TOP_K_CLASSES),
        "keywords": _extract_keywords(text, predicted_index, TOP_K_KEYWORDS),
        "review_message": (
            "Low-margin prediction. A reviewer should choose the correct queue "
            "from the suggested classes."
            if needs_review
            else None
        ),
    }


@app.on_event("startup")
def load_model() -> None:
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Support Ticket Router API is running",
        "review_margin_threshold": REVIEW_MARGIN_THRESHOLD,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "review_margin_threshold": REVIEW_MARGIN_THRESHOLD,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TicketRequest):
    return _predict_one(payload.text.strip())


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchTicketRequest):
    results = []
    for text in payload.texts:
        cleaned_text = text.strip() if isinstance(text, str) else ""
        results.append({"text": cleaned_text, **_predict_one(cleaned_text)})
    return {"predictions": results}


@app.post("/review/confirm", response_model=ReviewDecisionResponse)
def confirm_review(payload: ReviewDecisionRequest):
    selected_queue = payload.selected_queue.strip()
    model_prediction = payload.model_prediction.strip()

    if not selected_queue:
        raise HTTPException(status_code=422, detail="selected_queue must not be empty")

    return {
        "status": "accepted",
        "selected_queue": selected_queue,
        "model_prediction": model_prediction,
        "was_overridden": selected_queue != model_prediction,
        "message": (
            "Review decision accepted for this demo. "
            "The current endpoint does not persist decisions."
        ),
    }
