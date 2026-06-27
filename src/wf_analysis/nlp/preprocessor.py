"""Text preprocessing: cleaning, tokenization, lemmatization."""

import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)


class TextPreprocessor:
    def __init__(self):
        self.stopwords = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def transform(self, texts: pd.Series) -> pd.Series:
        def _clean(text: str) -> str:
            if not isinstance(text, str):
                return ""
            text = text.lower()
            text = re.sub(r"[^\w\s]", " ", text)
            text = re.sub(r"\d+", " ", text)
            tokens = nltk.word_tokenize(text)
            tokens = [t for t in tokens if t not in self.stopwords and len(t) > 2]
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
            return " ".join(tokens)

        return texts.fillna("").astype(str).apply(_clean)
