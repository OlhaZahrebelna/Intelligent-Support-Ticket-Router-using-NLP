import os
import sys
from typing import List, Optional

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.preprocessing import TextPreprocessor, nlp


# Compatibility for models saved from a notebook where
# TextPreprocessor lived in __main__.
sys.modules["__main__"].TextPreprocessor = TextPreprocessor
sys.modules["__main__"].nlp = nlp


MODEL_PATH = os.getenv("MODEL_PATH", "model/svm_pipeline.joblib")


app = FastAPI(
    title="Intelligent Support Ticket Router API",
    version="1.0.0",
    description=(
        "FastAPI service for routing support tickets "
        "with a TF-IDF + Linear SVM pipeline."
    ),
)


model = None


class TicketRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        examples=["I was charged twice for my monthly subscription."],
    )


class BatchTicketRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    prediction: str
    score: Optional[float] = None


class BatchPredictionResponse(BaseModel):
    predictions: List[str]


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
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TicketRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    label = model.predict([payload.text])[0]

    response = {
        "prediction": str(label),
        "score": None,
    }

    if hasattr(model, "decision_function"):
        try:
            scores = model.decision_function([payload.text])
            response["score"] = float(scores.max())
        except Exception:
            response["score"] = None

    return response


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchTicketRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    cleaned_texts = [
        text if isinstance(text, str) else ""
        for text in payload.texts
    ]

    predictions = model.predict(cleaned_texts)

    return {
        "predictions": [
            str(label)
            for label in predictions
        ]
    }
