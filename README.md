# Intelligent Support Ticket Router using NLP

An end-to-end NLP project for routing English customer-support tickets to one of ten support queues.

The repository covers the complete workflow:

- exploratory data analysis;
- classical NLP baselines;
- BERT fine-tuning;
- fixed-test error analysis;
- selective human-review strategy;
- FastAPI inference service;
- Streamlit review interface;
- deployment on Render and Streamlit Community Cloud.

## Live Demo

- **FastAPI API:** https://intelligent-support-ticket-router.onrender.com
- **Swagger documentation:** https://intelligent-support-ticket-router.onrender.com/docs
- **Streamlit application:** https://intelligent-support-ticket-router-nlp.streamlit.app

> The Render free instance may take additional time to start after inactivity.

## Business Problem

Customer-support requests often need to be routed manually to the correct operational team. Manual routing is time-consuming, inconsistent, and difficult to scale.

This project formulates routing as a **10-class text-classification problem**. Given the text of a support ticket, the system predicts one of the following queues:

1. Billing and Payments
2. Customer Service
3. General Inquiry
4. Human Resources
5. IT Support
6. Product Support
7. Returns and Exchanges
8. Sales and Pre-Sales
9. Service Outages and Maintenance
10. Technical Support

## System Overview

```text
Incoming ticket
      |
      v
spaCy preprocessing
      |
      v
TF-IDF vectorization
      |
      v
Linear SVM classifier
      |
      v
Top-2 decision margin
   /              \
margin >= 0.10    margin < 0.10
      |                 |
      v                 v
Automatic routing   Human review
```

The deployed system uses a TF-IDF + Linear SVM pipeline as the primary router.

For each ticket, the API returns:

- the predicted queue;
- the Linear SVM decision score;
- the top three candidate queues;
- the difference between the two strongest class scores;
- influential TF-IDF features;
- a `needs_review` flag.

Low-margin predictions are escalated to a reviewer. The reviewer can accept the predicted queue or select another suggested queue.

## Dataset

The cleaned English dataset used in the classical-model workflow contains:

| Property | Value |
|---|---:|
| Records | 23,748 |
| Unique record IDs | 23,748 |
| Unique combined ticket texts | 23,748 |
| Target classes | 10 |

The data is divided with stratified sampling:

| Split | Records | Share |
|---|---:|---:|
| Training | 15,198 | 64% |
| Validation | 3,800 | 16% |
| Test | 4,750 | 20% |
| **Total** | **23,748** | **100%** |

The final error analysis compares models on the same fixed 4,750-record test set.

## Text Preprocessing

The deployed pipeline uses a custom scikit-learn-compatible `TextPreprocessor` defined in:

```text
app/preprocessing.py
```

Current preprocessing steps:

1. Convert non-string values to empty strings.
2. Convert text to lowercase.
3. Remove URLs.
4. Remove email addresses.
5. Remove HTML tags.
6. Normalize whitespace.
7. Tokenize and lemmatize with `en_core_web_sm`.
8. Keep alphabetic tokens only.
9. Remove spaCy stop words and number-like tokens.
10. Remove lemmas shorter than two characters.

The class inherits from `BaseEstimator` and `TransformerMixin`, allowing it to be included directly in the serialized scikit-learn pipeline.

The module also exposes the global spaCy object `nlp`. This is retained for compatibility with the current `joblib` artifact.

## Classical Baselines

The primary evaluation metric is **Macro F1**, because the classes are imbalanced and each support queue should contribute equally to model evaluation.

### Initial validation results

| Model | Train Macro F1 | Validation Macro F1 |
|---|---:|---:|
| Logistic Regression | 0.435 | 0.345 |
| Tuned Logistic Regression | 0.432 | 0.318 |
| Linear SVM | 0.704 | 0.436 |

Linear SVM achieved the strongest validation performance among the classical baselines.

### Selected configuration

The final classical search evaluated 24 parameter combinations with stratified 3-fold cross-validation.

```python
TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.95,
    max_features=30_000,
    sublinear_tf=True,
)

LinearSVC(
    C=0.2,
    class_weight="balanced",
)
```

Best mean cross-validation Macro F1:

```text
0.4801
```

## Linear SVM vs BERT

Both models were evaluated on the same fixed test set.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| **Linear SVM** | **0.5604** | **0.5427** | **0.5616** |
| BERT | 0.4288 | 0.4297 | 0.4293 |

Linear SVM was selected for deployment because it:

- achieved better fixed-test performance;
- requires less memory;
- runs efficiently on CPU;
- has lower inference latency;
- is simpler to deploy and maintain;
- supports direct feature-level explanations.

### Model agreement analysis

| Outcome | Records | Share |
|---|---:|---:|
| Both models wrong | 1,606 | 33.81% |
| Both models correct | 1,555 | 32.74% |
| Only SVM correct | 1,107 | 23.31% |
| Only BERT correct | 482 | 10.15% |

The models produced the same prediction for 47.37% of test examples.

The large group where both models failed suggests that model architecture alone does not explain all errors. Important contributing factors include:

- semantically overlapping queue definitions;
- ambiguous ticket wording;
- class imbalance;
- potentially noisy labels.

## Class-Level Results

Strongest Linear SVM classes:

| Queue | F1 |
|---|---:|
| Billing and Payments | 0.806 |
| Service Outages and Maintenance | 0.607 |
| Technical Support | 0.598 |
| Human Resources | 0.547 |

Most difficult classes:

| Queue | F1 |
|---|---:|
| Sales and Pre-Sales | 0.433 |
| General Inquiry | 0.435 |
| IT Support | 0.472 |
| Product Support | 0.507 |
| Customer Service | 0.508 |

Frequent confusion patterns include:

- Technical Support → IT Support;
- Product Support → Technical Support;
- Technical Support → Product Support;
- Technical Support → Customer Service;
- Customer Service → Product Support;
- IT Support → Technical Support.

## Selective Human Review

The deployed API uses the difference between the two highest Linear SVM decision scores:

```text
margin = highest score - second-highest score
```

A smaller margin means the classifier is less able to separate its two strongest candidate queues.

The default review threshold is:

```text
0.10
```

This can be configured through:

```text
REVIEW_MARGIN_THRESHOLD
```

### Threshold experiment

| Threshold | Review rate | Automatic coverage | Automatic accuracy | SVM errors sent to review |
|---:|---:|---:|---:|---:|
| 0.02 | 6.65% | 93.35% | 57.96% | 224 |
| 0.03 | 9.87% | 90.13% | 58.75% | 322 |
| 0.05 | 16.25% | 83.75% | 61.01% | 537 |
| **0.10** | **29.62%** | **70.38%** | **65.69%** | **941** |
| 0.15 | 41.89% | 58.11% | 70.51% | 1,274 |
| 0.20 | 51.35% | 48.65% | 75.08% | 1,512 |

At threshold `0.10`:

- 70.38% of tickets remain automatically routed;
- 29.62% are marked for review;
- automatic-route accuracy is 65.69%;
- 941 SVM errors are concentrated in the review queue.

This does **not** prove that a human or automated reviewer will correct every escalated prediction. It shows that the margin is useful for separating easier automatic cases from harder review cases.

## Explainability

For each prediction, the API can return influential text features.

For a Linear SVM class, the contribution of a TF-IDF feature is approximated as:

```text
feature contribution = TF-IDF value × class coefficient
```

Only features with a positive contribution to the selected class are returned.

This provides a lightweight explanation of which words or phrases supported the routing decision.

## API

### Root

```http
GET /
```

Example:

```json
{
  "status": "ok",
  "message": "Support Ticket Router API is running",
  "api_version": "2.0.0",
  "review_margin_threshold": 0.1
}
```

### Health check

```http
GET /health
```

Example:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "review_margin_threshold": 0.1
}
```

### Single prediction

```http
POST /predict
```

Request:

```json
{
  "text": "I was charged twice for my monthly subscription."
}
```

Response structure:

```json
{
  "prediction": "Billing and Payments",
  "score": 1.42,
  "margin": 0.36,
  "review_threshold": 0.1,
  "needs_review": false,
  "routing_status": "automatic",
  "top_classes": [
    {
      "queue": "Billing and Payments",
      "decision_score": 1.42
    },
    {
      "queue": "Customer Service",
      "decision_score": 1.06
    },
    {
      "queue": "General Inquiry",
      "decision_score": 0.41
    }
  ],
  "keywords": [
    "charged",
    "subscription"
  ],
  "review_message": null
}
```

`score` is a Linear SVM decision score. It is not a calibrated probability.

### Batch prediction

```http
POST /predict_batch
```

Request:

```json
{
  "texts": [
    "I was charged twice for my subscription.",
    "I cannot access the corporate VPN.",
    "The application crashes after the latest update."
  ]
}
```

The endpoint returns the full prediction structure for each ticket.

### Confirm review decision

```http
POST /review/confirm
```

Request:

```json
{
  "text": "I cannot access the corporate VPN.",
  "model_prediction": "Technical Support",
  "selected_queue": "IT Support",
  "margin": 0.06,
  "reviewer_note": "The issue concerns internal infrastructure."
}
```

Example response:

```json
{
  "status": "accepted",
  "selected_queue": "IT Support",
  "model_prediction": "Technical Support",
  "was_overridden": true,
  "message": "Review decision accepted for this demo. The current endpoint does not persist decisions."
}
```

The current endpoint acknowledges the review decision but does not store it in a database.

## Streamlit Interface

The frontend allows a user to:

- enter or select a sample ticket;
- call the FastAPI prediction endpoint;
- see the predicted queue and margin;
- inspect top candidate queues;
- inspect influential TF-IDF features;
- identify whether review is required;
- select a final queue for low-margin cases;
- submit the review decision to the API;
- inspect the raw JSON response.

## Project Structure

```text
Intelligent-Support-Ticket-Router-using-NLP/
├── app/
│   ├── main.py
│   └── preprocessing.py
├── frontend/
│   ├── requirements.txt
│   └── streamlit_app.py
├── model/
│   └── svm_pipeline.joblib
├── notebook/
│   ├── 01_Description_and_EDA_ENG.ipynb
│   ├── 02_Preprocessing_and_Baseline.ipynb
│   ├── 03_BERT_FineTuning.ipynb
│   └── 04_Test_Error_Analysis and Selective Agent Review Strategy.ipynb
├── .python-version
├── requirements.txt
├── runtime.txt
└── README.md
```

## Notebook Workflow

### 1. Description and EDA

`01_Description_and_EDA_ENG.ipynb`

- validates the source dataset;
- analyzes missing values and duplicates;
- examines class distribution;
- examines ticket length;
- prepares the cleaned English dataset.

### 2. Preprocessing and baselines

`02_Preprocessing_and_Baseline.ipynb`

- creates stratified train, validation, and test splits;
- evaluates Logistic Regression;
- evaluates Linear SVM;
- performs cross-validation;
- tunes TF-IDF and Linear SVM parameters;
- exports the selected scikit-learn pipeline.

### 3. BERT fine-tuning

`03_BERT_FineTuning.ipynb`

- tokenizes ticket text;
- fine-tunes BERT for 10-class classification;
- uses class-weighted training;
- evaluates validation and test performance;
- exports prediction artifacts for comparison.

### 4. Error analysis and review strategy

`04_Test_Error_Analysis and Selective Agent Review Strategy.ipynb`

- validates and joins SVM and BERT predictions by `record_id`;
- compares both models on the same test records;
- analyzes class-level errors;
- identifies dominant confusion pairs;
- examines model agreement and disagreement;
- calculates SVM margins;
- evaluates selective-review thresholds;
- identifies possible label-audit candidates.

## Local Setup

Clone the repository:

```bash
git clone https://github.com/OlhaZahrebelna/Intelligent-Support-Ticket-Router-using-NLP.git
cd Intelligent-Support-Ticket-Router-using-NLP
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install backend dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Run FastAPI:

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Run the Frontend Locally

Install frontend dependencies:

```bash
pip install -r frontend/requirements.txt
```

Set the backend URL if necessary:

```bash
export API_BASE_URL=http://127.0.0.1:8000
```

Windows PowerShell:

```powershell
$env:API_BASE_URL="http://127.0.0.1:8000"
```

Run Streamlit:

```bash
streamlit run frontend/streamlit_app.py
```

## Configuration

Backend environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `model/svm_pipeline.joblib` | Path to the serialized pipeline |
| `REVIEW_MARGIN_THRESHOLD` | `0.10` | Margin below which review is required |
| `TOP_K_CLASSES` | `3` | Number of suggested queues returned |
| `TOP_K_KEYWORDS` | `8` | Maximum number of influential features |

Frontend environment variable:

| Variable | Default | Purpose |
|---|---|---|
| `API_BASE_URL` | Render API URL | Backend used by Streamlit |

## Deployment

### Backend

The FastAPI service is deployed on Render.

Build command:

```bash
pip install -r requirements.txt && python -m spacy download en_core_web_sm
```

Start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Python version:

```text
3.11.9
```

### Frontend

The Streamlit application is deployed separately using:

```text
frontend/streamlit_app.py
```

with dependencies from:

```text
frontend/requirements.txt
```

## Technology Stack

- Python 3.11
- pandas
- NumPy
- scikit-learn
- spaCy
- Hugging Face Transformers
- PyTorch
- FastAPI
- Pydantic
- Uvicorn
- Streamlit
- joblib
- Render

## Current Limitations

- Several target classes have substantial semantic overlap.
- The dataset may contain ambiguous or noisy labels.
- Linear SVM decision scores are not calibrated probabilities.
- The selective-review threshold was derived experimentally and should be monitored after deployment.
- Review decisions are acknowledged but not persisted.
- The reviewer can choose only from the classes presented by the current frontend.
- The current API does not implement authentication, durable logging, or a retraining feedback loop.
- The serialized model depends on the current preprocessing class and spaCy compatibility objects.


This project demonstrates an end-to-end NLP workflow from exploratory analysis and model comparison to deployment, explainability, and human-in-the-loop routing.
