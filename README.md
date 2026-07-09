# Intelligent Support Ticket Router using NLP

An NLP-powered support ticket classification service that predicts the most likely support category for a customer message.

The project uses a traditional machine learning pipeline with text preprocessing, TF-IDF features, and a Linear SVM classifier. The trained model is served through a FastAPI API and deployed on Render.

## Live Demo

FastAPI service:

https://intelligent-support-ticket-router.onrender.com

Interactive API documentation:

https://intelligent-support-ticket-router.onrender.com/docs

Prediction endpoint:

https://intelligent-support-ticket-router.onrender.com/docs#/default/predict_predict_post

## Project Overview

Customer support teams often receive a high volume of incoming tickets. Manually routing each ticket to the correct department can be slow, inefficient, and inconsistent.

This project automates support ticket routing. The model analyzes the text of a customer message and predicts the most relevant support category. The API can be used by other applications, dashboards, or frontend interfaces.

## Tech Stack

- Python 3.11.9
- FastAPI
- Uvicorn
- scikit-learn
- spaCy
- pandas
- joblib
- Pydantic
- Render

The Python version for deployment is defined in `runtime.txt`.

The main dependencies are listed in `requirements.txt`.

## Model

The API serves a saved `joblib` model:

```text
model/svm_pipeline.joblib
```

The machine learning pipeline includes:

- Text preprocessing
- Lemmatization with spaCy
- Stopword removal
- Removal of non-alphabetic tokens
- TF-IDF vectorization
- Linear SVM classification

The text preprocessing logic is implemented in:

```text
app/preprocessing.py
```

The FastAPI application loads the model on startup from:

```text
model/svm_pipeline.joblib
```

## API Endpoints

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Single Prediction

```http
POST /predict
```

Request body:

```json
{
  "text": "I was charged twice for my monthly subscription."
}
```

Example response:

```json
{
  "prediction": "billing",
  "score": 0.84
}
```

The exact prediction label and score may vary depending on the trained model.

### Batch Prediction

```http
POST /predict_batch
```

Request body:

```json
{
  "texts": [
    "I was charged twice for my monthly subscription.",
    "I cannot login to my account.",
    "The app crashes when I open it."
  ]
}
```

Example response:

```json
{
  "predictions": [
    "billing",
    "account_access",
    "technical_support"
  ]
}
```

The exact category names depend on the classes learned by the trained model.

## Project Structure

```text
Intelligent-Support-Ticket-Router-using-NLP/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── preprocessing.py
│
├── model/
│   └── svm_pipeline.joblib
│
├── notebook/
│   ├── 01_Description_and_EDA_ENG.ipynb
│   ├── 02_Preprocessing_and_Baseline.ipynb
│   └── 03_BERT_FineTuning.ipynb
│
├── requirements.txt
├── runtime.txt
└── README.md
```

## Local Setup

Clone the repository:

```bash
git clone https://github.com/OlhaZahrebelna/Intelligent-Support-Ticket-Router-using-NLP.git
cd Intelligent-Support-Ticket-Router-using-NLP
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Run the FastAPI app locally:

```bash
uvicorn app.main:app --reload
```

Open the local API documentation:

```text
http://127.0.0.1:8000/docs
```

## Deployment

The API is deployed on Render as a Web Service.

Render configuration:

```bash
Build Command:
pip install -r requirements.txt && python -m spacy download en_core_web_sm
```

```bash
Start Command:
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Python version:

```text
python-3.11.9
```

## Example cURL Request

```bash
curl -X POST "https://intelligent-support-ticket-router.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice for my monthly subscription."}'
```

## Notes

- The model is loaded once when the FastAPI application starts.
- The `/predict` endpoint returns one prediction for a single ticket.
- The `/predict_batch` endpoint returns predictions for multiple tickets.
- The `score` value is based on the model decision function when available.
- On the Render Free plan, the first request after inactivity may take longer because the service can go into sleep mode.
