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

## Fixed-Test Comparison: Linear SVM vs BERT

The error-analysis notebook compares both models on exactly the same **4,750-record test set**. Prediction artifacts are joined by `record_id`, and the notebook validates that both files contain the same records, text, and ground-truth labels before any comparison.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| **Linear SVM** | **0.5604** | **0.5427** | **0.5616** |
| BERT | 0.4288 | 0.4297 | 0.4293 |

Linear SVM is therefore the stronger primary router in the current experiment. BERT is not assigned equal production weight; it is retained as a secondary disagreement and auditing signal.

The models produce the same prediction for only **47.37%** of test records.

| Outcome | Records | Share |
|---|---:|---:|
| Both models wrong | 1,606 | 33.81% |
| Both models correct | 1,555 | 32.74% |
| Only SVM correct | 1,107 | 23.31% |
| Only BERT correct | 482 | 10.15% |

The large `Both wrong` group shows that many errors are caused by task ambiguity, overlapping queue definitions, or possible label noise rather than by one specific model architecture.

### Class-Level Findings

Linear SVM achieves its strongest class-level performance on:

| Queue | SVM F1 |
|---|---:|
| Billing and Payments | 0.806 |
| Service Outages and Maintenance | 0.607 |
| Technical Support | 0.598 |
| Human Resources | 0.547 |

The most difficult SVM classes are:

| Queue | SVM F1 |
|---|---:|
| Sales and Pre-Sales | 0.433 |
| General Inquiry | 0.435 |
| IT Support | 0.472 |
| Product Support | 0.507 |
| Customer Service | 0.508 |

The largest SVM-over-BERT F1 improvements occur for `General Inquiry`, `Technical Support`, `Customer Service`, and `Product Support`.

### Dominant Error Patterns

The most frequent errors are concentrated among semantically overlapping support queues. Important confusion directions include:

- Technical Support → IT Support;
- Product Support → Technical Support;
- Technical Support → Product Support;
- Technical Support → Customer Service;
- Customer Service → Product Support;
- IT Support → Technical Support.

Both models are wrong on **1,606** tickets, and on **695** of those records they select the same incorrect label. Shared high-confidence mistakes are especially important label-audit candidates because model agreement does not guarantee that the dataset label is correct.

Ticket length is not a strong standalone explanation of errors. Median text length ranges from 53 to 64 words across model-outcome groups, so semantic ambiguity is more important than document length alone.

## Selective Agent Review Strategy

The production-oriented extension uses the Linear SVM prediction margin as an uncertainty signal. A ticket is routed automatically when the SVM margin is above a selected threshold; lower-margin tickets are sent to a human or LLM-assisted review agent.

The strategy is evaluated with the following metrics:

- **agent review rate** — share of all tickets sent for review;
- **automatic coverage** — share routed without review;
- **automatic accuracy** — accuracy on automatically routed tickets;
- **errors sent to agent** — number of SVM errors captured by review;
- **agent-case error rate** — error concentration inside the review queue;
- **error recall** — share of all SVM errors captured by review.

| Margin threshold | Agent review rate | Automatic coverage | Automatic accuracy | Errors sent to agent | Error recall |
|---:|---:|---:|---:|---:|---:|
| 0.02 | 6.65% | 93.35% | 57.96% | 224 | 10.73% |
| 0.03 | 9.87% | 90.13% | 58.75% | 322 | 15.42% |
| 0.05 | 16.25% | 83.75% | 61.01% | 537 | 25.72% |
| **0.10** | **29.62%** | **70.38%** | **65.69%** | **941** | **45.07%** |
| 0.15 | 41.89% | 58.11% | 70.51% | 1,274 | 61.02% |
| 0.20 | 51.35% | 48.65% | 75.08% | 1,512 | 72.41% |

A margin threshold of **0.10** is selected as the most balanced experimental operating point:

- about **70.4%** of tickets remain fully automatic;
- about **29.6%** are escalated for review;
- automatic-route accuracy increases from the base SVM accuracy of **56.04%** to **65.69%**;
- the review queue captures **941 model errors**, or about **45.1%** of all SVM errors;
- reviewed cases have an error rate of approximately **66.9%**, showing that the margin successfully concentrates difficult tickets.

This threshold is an experiment-derived recommendation, not a universal production constant. In a deployed system it should be selected using business costs, agent capacity, acceptable error rates, latency requirements, and monitored data drift.

### Proposed Hybrid Routing Flow

```text
Incoming ticket
      |
      v
TextPreprocessor
      |
      v
TF-IDF + Linear SVM
      |
      v
Prediction margin
   /       \
high       low
 |          |
 v          v
Automatic  Human or
routing    LLM-assisted review
```

The weaker BERT model can provide an additional disagreement signal, but the notebook does not support using a simple equal-weight SVM–BERT ensemble because BERT is materially weaker on the fixed test set.

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
│   └── 04_Test_Error_Analysis and Selective Agent Review Strategy.ipynb
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

### `04_Test_Error_Analysis and Selective Agent Review Strategy.ipynb`

- validates and merges SVM and BERT artifacts by `record_id`;
- compares both models on the same fixed 4,750-record test set;
- calculates accuracy, Macro F1, Weighted F1, and class-level metrics;
- separates both-correct, both-wrong, and model-specific outcomes;
- identifies dominant confusion pairs and shared wrong predictions;
- calculates SVM decision margins and analyzes BERT confidence;
- reviews low-margin and high-confidence disagreement cases;
- identifies potential label-review candidates;
- evaluates selective agent-review thresholds;
- recommends a 0.10 SVM-margin operating point for the experimental hybrid workflow;
- exports reusable error-analysis tables.

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
- implement the experimental 0.10-margin selective review workflow in the API;
- evaluate human and LLM reviewer accuracy on escalated tickets;
- combine margin, class-specific risk, and model disagreement in the review policy;
- hierarchical routing for semantically overlapping queues;
- targeted relabeling of ambiguous examples;
- improved domain-specific preprocessing;
- monitoring for class drift and prediction drift;
- automated tests for preprocessing and API contracts.

## Author

**Olha Zahrebelna**

This repository is a portfolio project demonstrating an end-to-end NLP workflow: data preparation, model experimentation, error analysis, deployment, and production-oriented inference.
