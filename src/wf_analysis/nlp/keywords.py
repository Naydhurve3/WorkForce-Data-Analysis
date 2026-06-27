import pandas as pd
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)


class KeywordExtractor:
    def __init__(self, method: str = "rake"):
        self.method = method
        self.stopwords = set(stopwords.words("english"))

    def _rake(self, text: str) -> list[tuple[str, float]]:
        if not isinstance(text, str) or not text.strip():
            return []
        sentences = nltk.sent_tokenize(text)
        phrases = []
        for sent in sentences:
            words = nltk.word_tokenize(sent)
            candidate = []
            for word in words:
                w = word.lower().strip(".,!?;:")
                if w and w not in self.stopwords and len(w) > 2:
                    candidate.append(w)
                else:
                    if candidate:
                        phrases.append(" ".join(candidate))
                        candidate = []
            if candidate:
                phrases.append(" ".join(candidate))

        word_scores: dict[str, float] = {}
        phrase_scores: list[tuple[str, float]] = []
        for phrase in phrases:
            words = phrase.split()
            score = sum(len(w) for w in words)
            phrase_scores.append((phrase, score))
            for w in words:
                word_scores[w] = word_scores.get(w, 0) + 1

        if word_scores:
            normalized = []
            for phrase, score in phrase_scores:
                words = phrase.split()
                adj = sum(word_scores.get(w, 1) for w in words)
                normalized.append((phrase, score / adj if adj else 0))
            normalized.sort(key=lambda x: x[1], reverse=True)
            return normalized[:10]
        return phrase_scores[:10]

    def extract(
        self, texts: pd.Series
    ) -> dict[int, list[tuple[str, float]]]:
        results = {}
        for idx, text in texts.items():
            results[idx] = self._rake(str(text))
        return results
