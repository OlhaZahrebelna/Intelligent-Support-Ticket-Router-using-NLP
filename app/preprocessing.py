"""Text preprocessing for the TF-IDF + LinearSVC routing pipeline.

Keep the module path and ``TextPreprocessor`` class name stable because the
serialized joblib pipeline may reference ``app.preprocessing.TextPreprocessor``.
"""

import re
from typing import Iterable, Optional

import numpy as np
import spacy
from sklearn.base import BaseEstimator, TransformerMixin


# Load only the components required for tokenization and lemmatization.
_NLP = spacy.load(
    "en_core_web_sm",
    disable=["parser", "ner"],
)
nlp = _NLP

NEGATION_WORDS = {
    "no",
    "not",
    "never",
    "cannot",
    "n't",
}


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocess English support-ticket text for TF-IDF models.

    Processing steps:
    1. Handle missing and non-string values safely.
    2. Normalize apostrophes and English negation contractions.
    3. Remove URLs and email addresses.
    4. Tokenize and lemmatize with spaCy.
    5. Remove stop words while preserving meaningful negations.
    6. Keep alphabetic tokens and remove very short tokens.
    """

    def __init__(
        self,
        min_token_len: int = 2,
        batch_size: int = 500,
    ) -> None:
        self.min_token_len = min_token_len
        self.batch_size = batch_size

    def fit(
        self,
        X: Iterable[object],
        y: Optional[Iterable[object]] = None,
    ) -> "TextPreprocessor":
        """Return the fitted stateless transformer."""
        return self

    @staticmethod
    def normalize_text(text: object) -> str:
        """Normalize contractions and remove non-content text patterns."""
        text = str(text)

        # Normalize typographic apostrophes before contraction processing.
        text = text.replace("’", "'").replace("`", "'")

        # Handle irregular negative contractions first.
        text = re.sub(
            r"\bcan't\b",
            "cannot",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bwon't\b",
            "will not",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bshan't\b",
            "shall not",
            text,
            flags=re.IGNORECASE,
        )

        # General negative contractions:
        # isn't -> is not, didn't -> did not, couldn't -> could not.
        text = re.sub(
            r"n['’]t\b",
            " not",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"https?://\S+|www\.\S+",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b[\w.\-+]+@[\w.\-]+\.\w+\b",
            " ",
            text,
        )

        return re.sub(r"\s+", " ", text).strip()

    def transform(self, X: Iterable[object]) -> np.ndarray:
        """Return preprocessed documents as a one-dimensional NumPy array."""
        texts = [
            self.normalize_text(text)
            if text is not None
            and not (isinstance(text, float) and np.isnan(text))
            else ""
            for text in X
        ]

        processed_texts = []

        for doc in _NLP.pipe(texts, batch_size=self.batch_size):
            tokens = []

            for token in doc:
                token_text = token.text.lower().strip()
                lemma = token.lemma_.lower().strip()

                if not lemma:
                    continue

                is_negation = (
                    token_text in NEGATION_WORDS
                    or lemma in NEGATION_WORDS
                )

                if not (token.is_alpha or is_negation):
                    continue

                if token.is_stop and not is_negation:
                    continue

                if len(lemma) < self.min_token_len and not is_negation:
                    continue

                # Map contraction fragments and "cannot" to one stable feature.
                if token_text in {"n't", "cannot"} or lemma == "cannot":
                    lemma = "not"

                tokens.append(lemma)

            processed_texts.append(" ".join(tokens))

        return np.asarray(processed_texts, dtype=object)
