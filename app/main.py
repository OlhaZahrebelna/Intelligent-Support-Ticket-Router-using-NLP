import os
import sys
from typing import Any, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.preprocessing import TextPreprocessor, nlp


# Compatibility for joblib artifacts serialized from a notebook where
# TextPreprocessor and nlp were defined in __main__.
sys.modules["__main__"].TextPreprocessor = TextPreprocessor
sys.modules["__main__"].nlp = nlp


MODEL_PATH = os.getenv("MODEL_PATH", "model/svm_pipeline.joblib")
REVIEW_MARGIN_THRESHOLD = float(
    os.getenv("REVIEW_MARGIN_THRESHOLD", "0.10")
)
TOP_K_CLASSES = int(os.getenv("TOP_K_CLASSES", "3"))
TOP_K_KEYWORDS = int(os.getenv("TOP_K_KEYWORDS", "8"))


app = FastAPI(
    title="Intelligent Support Ticket Router API",
    version="2.0.0",
    description=(
        "Support-ticket routing with TF-IDF, Linear SVM, and selective "
        "human review for low-margin predictions."
    ),
)

model: Optional[Any] = None


class TicketRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=20_000,
        examples=["I was charged twice for my monthly subscription."],
    )


class BatchTicketRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
    )


class ClassCandidate(BaseModel):
    queue: str
    decision_score: float


class PredictionResponse(BaseModel):
    prediction: str
    score: float
    margin: float
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
    text: str = Field(..., min_length=1, max_length=20_000)
    model_prediction: str = Field(..., min_length=1)
    selected_queue: str = Field(..., min_length=1)
    margin: Optional[float] = None
    reviewer_note: Optional[str] = Field(default=None, max_length=2_000)


class ReviewDecisionResponse(BaseModel):
    status: str
    selected_queue: str
    model_prediction: str
    was_overridden: bool
    message: str


def _get_classifier(pipeline: Any) -> Any:
    """Return the final estimator from a fitted sklearn pipeline."""
    if hasattr(pipeline, "steps") and pipeline.steps:
        return pipeline.steps[-1][1]
    return pipeline


def _get_vectorizer(pipeline: Any) -> Optional[Any]:
    """Return the fitted vectorizer from a sklearn pipeline."""
    if not hasattr(pipeline, "steps"):
        return None

    for _, step in reversed(pipeline.steps[:-1]):
        if hasattr(step, "get_feature_names_out"):
            return step

    return None


def _transform_to_vectorizer(
    pipeline: Any,
    texts: List[str],
) -> Optional[Any]:
    """Transform texts through the vectorizer, excluding the classifier."""
    if not hasattr(pipeline, "steps"):
        return None

    transformed: Any = texts

    for _, step in pipeline.steps[:-1]:
        if not hasattr(step, "transform"):
            return None

        transformed = step.transform(transformed)

        if hasattr(step, "get_feature_names_out"):
            return transformed

    return None


def _get_classes() -> np.ndarray:
    if model is None:
        raise RuntimeError("Model is not loaded")

    classifier = _get_classifier(model)
    classes = getattr(classifier, "classes_", None)

    if classes is None:
        classes = getattr(model, "classes_", None)

    if classes is None:
        raise RuntimeError("Class labels are unavailable")

    return np.asarray(classes, dtype=object)


def _get_decision_scores(texts: List[str]) -> np.ndarray:
    if model is None:
        raise RuntimeError("Model is not loaded")

    if not hasattr(model, "decision_function"):
        raise RuntimeError("Model does not expose decision_function")

    scores = np.asarray(model.decision_function(texts), dtype=float)

    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])

    return scores


def _rank_classes(
    row_scores: np.ndarray,
    classes: np.ndarray,
) -> tuple[List[dict], int]:
    top_k = max(1, min(TOP_K_CLASSES, len(classes)))
    ranked_indices = np.argsort(row_scores)[::-1]
    top_indices = ranked_indices[:top_k]

    candidates = [
        {
            "queue": str(classes[index]),
            "decision_score": float(row_scores[index]),
        }
        for index in top_indices
    ]

    return candidates, int(ranked_indices[0])


def _calculate_margin(row_scores: np.ndarray) -> float:
    if row_scores.size < 2:
        return 0.0

    sorted_scores = np.sort(row_scores)
    return float(sorted_scores[-1] - sorted_scores[-2])


def _extract_keywords(
    text: str,
    predicted_class_index: int,
) -> List[str]:
    """Return TF-IDF features with the largest positive SVM contribution."""
    if model is None:
        return []

    vectorizer = _get_vectorizer(model)
    classifier = _get_classifier(model)

    if vectorizer is None or not hasattr(classifier, "coef_"):
        return []

    try:
        features = _transform_to_vectorizer(model, [text])

        if features is None or features.shape[0] == 0:
            return []

        feature_names = np.asarray(
            vectorizer.get_feature_names_out(),
            dtype=object,
        )
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

        order = np.argsort(contributions)[::-1][:TOP_K_KEYWORDS]

        return [
            str(feature_names[indices[position]])
            for position in order
        ]

    except (AttributeError, IndexError, TypeError, ValueError):
        # Explanatory metadata must never break the prediction endpoint.
        return []


def _predict_one(text: str) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        classes = _get_classes()
        row_scores = _get_decision_scores([text])[0]
        top_classes, predicted_index = _rank_classes(row_scores, classes)
        margin = _calculate_margin(row_scores)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    prediction = str(classes[predicted_index])
    decision_score = float(row_scores[predicted_index])
    needs_review = margin < REVIEW_MARGIN_THRESHOLD

    return {
        "prediction": prediction,
        # Kept for backward compatibility with the original frontend.
        "score": decision_score,
        "margin": margin,
        "review_threshold": REVIEW_MARGIN_THRESHOLD,
        "needs_review": needs_review,
        "routing_status": (
            "needs_review" if needs_review else "automatic"
        ),
        "top_classes": top_classes,
        "keywords": _extract_keywords(text, predicted_index),
        "review_message": (
            "Low-margin prediction. A reviewer should choose the correct "
            "queue from the suggested classes."
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
def root() -> dict:
    return {
        "status": "ok",
        "message": "Support Ticket Router API is running",
        "api_version": app.version,
        "review_margin_threshold": REVIEW_MARGIN_THRESHOLD,
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "api_version": app.version,
        "review_margin_threshold": REVIEW_MARGIN_THRESHOLD,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TicketRequest) -> dict:
    return _predict_one(payload.text.strip())


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchTicketRequest) -> dict:
    results = []

    for text in payload.texts:
        cleaned_text = text.strip()
        results.append(
            {
                "text": cleaned_text,
                **_predict_one(cleaned_text),
            }
        )

    return {"predictions": results}


@app.post("/review/confirm", response_model=ReviewDecisionResponse)
def confirm_review(payload: ReviewDecisionRequest) -> dict:
    """Acknowledge a human decision without persisting it yet."""
    selected_queue = payload.selected_queue.strip()
    model_prediction = payload.model_prediction.strip()

    return {
        "status": "accepted",
        "selected_queue": selected_queue,
        "model_prediction": model_prediction,
        "was_overridden": selected_queue != model_prediction,
        "message": (
            "Review decision accepted. This P0 endpoint does not persist "
            "review decisions yet."
        ),
    }
