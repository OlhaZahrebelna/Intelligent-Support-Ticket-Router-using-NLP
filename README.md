# Intelligent Support Ticket Router using NLP

An end-to-end NLP project for automatically routing English customer-support tickets to the most appropriate support queue.

The project compares classical text-classification baselines, fine-tunes a transformer model in a separate experiment, analyzes model errors, and deploys the selected production pipeline through FastAPI and Streamlit.

## Live Applications

- **FastAPI service:** https://intelligent-support-ticket-router.onrender.com
- **Interactive API documentation:** https://intelligent-support-ticket-router.onrender.com/docs
- **Streamlit demo:** https://intelligent-support-ticket-router-nlp.streamlit.app

> The Render free instance may need additional time to start after a period of inactivity.

## Business Problem

Customer-support teams receive large volumes of incoming requests. Manual routing is slow, inconsistent, and difficult to scale.

This project treats ticket routing as a **10-class text-classification problem**. Given the subject and body of a support request, the model predicts the queue that should handle the ticket.

The target queues are:

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

## Dataset

The cleaned dataset used in the classical-baseline notebook contains:

- **23,748 records**
- **23,748 unique record IDs**
- **23,748 unique combined ticket texts**
- **10 target queues**
- English-language tickets only

The model input is the combined ticket text, while `queue` is the target variable.

The data is divided using stratified sampling:

| Split | Records | Approximate share |
|---|---:|---:|
| Training | 15,198 | 64% |
| Validation | 3,800 | 16% |
| Test | 4,750 | 20% |
| **Total** | **23,748** | **100%** |

The test set remains untouched during initial model comparison and hyperparameter selection.

## Text Preprocessing

Classical models use a custom scikit-learn-compatible `TextPreprocessor`.

The transformer:

- handles missing and non-string values;
- normalizes apostrophes and common English contractions;
- converts negative contractions such as `isn't` and `can't` into a stable negation form;
- preserves meaningful negations such as `no`, `not`, and `never`;
- removes URLs and email addresses;
- normalizes whitespace;
- tokenizes and lemmatizes text with spaCy;
- removes stop words except preserved negations;
- removes numeric, punctuation, and non-alphabetic tokens;
- removes very short tokens;
- processes texts in batches with `nlp.pipe()`.

Example:

```text
Original:  I can't log into my profile.
Processed: not log profile
```

The implementation used by the deployed pipeline is located in:

```text
app/preprocessing.py
```

Keeping preprocessing inside the scikit-learn pipeline prevents leakage and ensures that the same transformations are applied during training and inference.

## Classical Baseline Experiments

The main evaluation metric is **Macro F1**, because the target classes are imbalanced and each queue should contribute equally to the final score.

### Validation Results

| Model | Train Macro F1 | Validation Macro F1 | Notes |
|---|---:|---:|---|
| Logistic Regression | 0.435 | 0.345 | Basic TF-IDF baseline |
| Tuned Logistic Regression | 0.432 | 0.318 | Balanced weights increased minority recall but reduced precision |
| Linear SVM | 0.704 | 0.436 | Strongest initial baseline, but with a large generalization gap |
| Cross-validated Linear SVM | — | **0.480 mean CV** | Selected through stratified 3-fold cross-validation |

The tuned Logistic Regression did not outperform the default configuration. Linear SVM produced the strongest classical-model performance and was therefore selected for systematic hyperparameter tuning.

## Selected Linear SVM Configuration

Grid search evaluates 24 parameter combinations with stratified 3-fold cross-validation, for a total of 72 fits.

The selected configuration is:

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

The lower `C` value applies stronger regularization than the default Linear SVM. Unigrams and bigrams allow the model to capture phrases such as `password reset`, `billing issue`, and `service outage`.

## Why Linear SVM Was Selected

Linear SVM is well suited to sparse, high-dimensional TF-IDF features. In this project it:

- clearly outperformed both Logistic Regression baselines;
- produced better F1 scores for several minority queues;
- was considerably lighter and faster than a transformer model;
- supported low-latency CPU inference;
- was straightforward to package in a single scikit-learn pipeline.

The project also evaluates BERT separately. The production choice should be understood as an engineering trade-off between predictive quality, inference cost, latency, memory requirements, and deployment complexity.

## Known Challenges

The task remains difficult because several queues are semantically close. Commonly overlapping categories include:

- Technical Support vs IT Support;
- Technical Support vs Product Support;
- Customer Service vs Technical Support;
- General Inquiry vs Sales and Pre-Sales.

Class imbalance also makes minority categories more difficult to learn. A high training score with a substantially lower validation score indicates that regularization alone cannot completely resolve semantic overlap or possible label noise.

## Model Artifact

The deployed scikit-learn pipeline is stored as:

```text
model/svm_pipeline.joblib
```

The serialized artifact contains:

1. `TextPreprocessor`
2. `TfidfVectorizer`
3. `LinearSVC`

Because the preprocessing class is referenced by the serialized pipeline, its module path and class name must remain stable.

> After changing `app/preprocessing.py`, retrain and export the complete pipeline. Replacing only the Python preprocessing file can create a mismatch between inference-time text and the TF-IDF vocabulary learned during training.

## API

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

Request:

```json
{
  "text": "I was charged twice for my monthly subscription."
}
```

Example response:

```json
{
  "prediction": "Billing and Payments",
  "score": 0.84
}
```

The score is derived from the classifier decision function and should be interpreted as a relative model confidence signal, not as a calibrated probability.

### Batch Prediction

```http
POST /predict_batch
```

Request:

```json
{
  "texts": [
    "I was charged twice for my monthly subscription.",
    "I cannot log in to my employee account.",
    "The application crashes after the latest update."
  ]
}
```

The endpoint returns one queue prediction for each submitted text.

## Example cURL Request

```bash
curl -X POST \
  "https://intelligent-support-ticket-router.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"I was charged twice for my monthly subscription."}'
```

## Project Structure

```text
Intelligent-Support-Ticket-Router-using-NLP/
├── app/
│   ├── __init__.py
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
│   └── 04_Error_Analysis.ipynb
├── requirements.txt
├── runtime.txt
└── README.md
```

## Notebook Workflow

### `01_Description_and_EDA_ENG.ipynb`

- dataset validation;
- class distribution analysis;
- ticket-length analysis;
- duplicate and missing-value checks;
- preparation of the cleaned English dataset.

### `02_Preprocessing_and_Baseline.ipynb`

- stratified train-validation-test split;
- reusable spaCy preprocessing transformer;
- Logistic Regression baselines;
- Linear SVM baseline;
- Macro F1-based model comparison;
- stratified cross-validation;
- TF-IDF and SVM hyperparameter search;
- final classical pipeline export.

### `03_BERT_FineTuning.ipynb`

- BERT tokenization;
- fine-tuning for 10-class classification;
- class-weighted training;
- validation and test evaluation;
- comparison with the classical baseline.

### `04_Error_Analysis.ipynb`

- class-level error rates;
- confusion-pair analysis;
- ticket-length analysis;
- comparison of SVM and BERT predictions;
- identification of ambiguous or potentially noisy labels;
- confidence-based agent-review experiments.

## Local Setup

Clone the repository:

```bash
git clone https://github.com/OlhaZahrebelna/Intelligent-Support-Ticket-Router-using-NLP.git
cd Intelligent-Support-Ticket-Router-using-NLP
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies and the spaCy English model:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Run FastAPI locally:

```bash
uvicorn app.main:app --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

## Deployment

The API is deployed as a Render Web Service.

Build command:

```bash
pip install -r requirements.txt && python -m spacy download en_core_web_sm
```

Start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

The deployment Python version is defined in `runtime.txt`.

## Tech Stack

- Python
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
- Docker
- joblib
- Render

## Future Improvements

- probability calibration for more interpretable confidence scores;
- class-specific confidence thresholds;
- agent or LLM review for uncertain predictions;
- hierarchical routing for semantically overlapping queues;
- targeted relabeling of ambiguous examples;
- improved domain-specific preprocessing;
- monitoring for class drift and prediction drift;
- automated tests for preprocessing and API contracts.

This repository is a portfolio project demonstrating an end-to-end NLP workflow: data preparation, model experimentation, error analysis, deployment, and production-oriented inference.
