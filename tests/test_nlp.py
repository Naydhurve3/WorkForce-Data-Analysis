import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from wf_analysis.nlp import (
    TextPreprocessor,
    SentimentAnalyzer,
    TopicModeler,
    TextClassifier,
    KeywordExtractor,
    TextEmbedder,
    NLPVisualizer,
)


class TestTextPreprocessor:
    def test_transform_cleans_text(self):
        p = TextPreprocessor()
        series = pd.Series(["HELLO World!", "the quick brown fox"])
        result = p.transform(series)
        assert len(result) == 2
        assert isinstance(result.iloc[0], str)

    def test_transform_handles_non_string_input(self):
        p = TextPreprocessor()
        series = pd.Series([123, None, 45.6])
        result = p.transform(series)
        assert len(result) == 3
        assert all(isinstance(v, str) for v in result)


class TestSentimentAnalyzer:
    def test_positive_sentiment(self):
        a = SentimentAnalyzer()
        result = a.analyze(pd.Series(["This is great and wonderful!"]))
        assert result["sentiment_score"].iloc[0] > 0

    def test_negative_sentiment(self):
        a = SentimentAnalyzer()
        result = a.analyze(pd.Series(["This is terrible and awful."]))
        assert result["sentiment_score"].iloc[0] < 0

    def test_neutral_sentiment(self):
        a = SentimentAnalyzer()
        result = a.analyze(pd.Series(["The car is blue."]))
        assert result["sentiment_label"].iloc[0] == "Neutral"


class TestTopicModeler:
    def test_fit_returns_self(self):
        m = TopicModeler(n_topics=2)
        texts = pd.Series([
            "machine learning model data analysis",
            "deep learning model data analysis",
            "machine learning model data science",
            "deep neural network image data",
            "reinforcement learning robot control",
        ])
        result = m.fit(texts)
        assert result is m
        assert m._fitted

    def test_transform_returns_dataframe_with_expected_columns(self):
        m = TopicModeler(n_topics=2)
        texts = pd.Series([
            "machine learning model data analysis",
            "deep learning model data analysis",
            "machine learning model data science",
            "deep neural network image data",
            "reinforcement learning robot control",
        ])
        m.fit(texts)
        result = m.transform(texts)
        assert isinstance(result, pd.DataFrame)
        assert "dominant_topic" in result.columns
        assert "topic_probability" in result.columns
        assert len(result) == 5

    def test_get_topic_info_returns_list_of_dicts(self):
        m = TopicModeler(n_topics=2)
        texts = pd.Series([
            "machine learning model data analysis",
            "deep learning model data analysis",
            "machine learning model data science",
            "deep neural network image data",
            "reinforcement learning robot control",
        ])
        m.fit(texts)
        info = m.get_topic_info(n_words=5)
        assert isinstance(info, list)
        assert len(info) == 2
        for topic in info:
            assert "topic_id" in topic
            assert "top_words" in topic
            assert "prevalence" in topic
            assert isinstance(topic["top_words"], list)

    def test_plot_topics_returns_figure(self):
        m = TopicModeler(n_topics=2)
        texts = pd.Series([
            "machine learning model data analysis",
            "deep learning model data analysis",
            "machine learning model data science",
            "deep neural network image data",
            "reinforcement learning robot control",
        ])
        m.fit(texts)
        fig = m.plot_topics()
        assert isinstance(fig, plt.Figure)

    def test_transform_before_fit_raises_runtimeerror(self):
        m = TopicModeler()
        with pytest.raises(RuntimeError, match="must be fitted before transform"):
            m.transform(pd.Series(["some text"]))

    def test_get_topic_info_before_fit_raises_runtimeerror(self):
        m = TopicModeler()
        with pytest.raises(RuntimeError, match="must be fitted first"):
            m.get_topic_info()

    def test_plot_topics_before_fit_returns_none(self):
        m = TopicModeler()
        result = m.plot_topics()
        assert result is None

    def test_plot_topics_with_single_topic_returns_figure(self):
        m = TopicModeler(n_topics=1)
        texts = pd.Series([
            "machine learning model data analysis",
            "deep learning model data analysis",
            "machine learning model data science",
            "deep neural network image data",
            "reinforcement learning robot control",
        ])
        m.fit(texts)
        fig = m.plot_topics()
        assert isinstance(fig, plt.Figure)


class TestTextClassifier:
    def test_fit_stores_metrics_dict(self):
        c = TextClassifier()
        texts = pd.Series([
            "this product is amazing and fantastic",
            "terrible experience worst purchase ever",
            "very happy with the quality and service",
            "poor customer support and slow delivery",
            "absolutely love it highly recommended",
            "disappointed with the build quality",
            "great value for money and fast shipping",
            "not worth the price at all",
            "excellent product would buy again",
            "completely broken upon arrival",
        ])
        labels = pd.Series([
            "pos", "neg", "pos", "neg", "pos",
            "neg", "pos", "neg", "pos", "neg",
        ])
        result = c.fit(texts, labels)
        assert result is c
        assert isinstance(c.metrics, dict)
        assert "accuracy" in c.metrics
        assert "precision" in c.metrics
        assert "recall" in c.metrics
        assert "f1" in c.metrics
        assert c._fitted

    def test_predict_returns_array(self):
        c = TextClassifier()
        texts = pd.Series([
            "this product is amazing and fantastic",
            "terrible experience worst purchase ever",
            "very happy with the quality and service",
            "poor customer support and slow delivery",
            "absolutely love it highly recommended",
            "disappointed with the build quality",
            "great value for money and fast shipping",
            "not worth the price at all",
            "excellent product would buy again",
            "completely broken upon arrival",
        ])
        labels = pd.Series([
            "pos", "neg", "pos", "neg", "pos",
            "neg", "pos", "neg", "pos", "neg",
        ])
        c.fit(texts, labels)
        result = c.predict(pd.Series(["this is wonderful", "this is terrible"]))
        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    def test_evaluate_returns_dict_with_accuracy_and_confusion_matrix(self):
        c = TextClassifier()
        texts = pd.Series([
            "this product is amazing and fantastic",
            "terrible experience worst purchase ever",
            "very happy with the quality and service",
            "poor customer support and slow delivery",
            "absolutely love it highly recommended",
            "disappointed with the build quality",
            "great value for money and fast shipping",
            "not worth the price at all",
            "excellent product would buy again",
            "completely broken upon arrival",
        ])
        labels = pd.Series([
            "pos", "neg", "pos", "neg", "pos",
            "neg", "pos", "neg", "pos", "neg",
        ])
        c.fit(texts, labels)
        result = c.evaluate(texts.iloc[:4], labels.iloc[:4])
        assert isinstance(result, dict)
        assert "accuracy" in result
        assert "confusion_matrix" in result
        assert isinstance(result["confusion_matrix"], np.ndarray)

    def test_plot_confusion_matrix_returns_figure(self):
        c = TextClassifier()
        cm = np.array([[3, 1], [0, 4]])
        fig = c.plot_confusion_matrix(cm)
        assert isinstance(fig, plt.Figure)

    def test_plot_confusion_matrix_with_int_array_from_fit(self):
        c = TextClassifier()
        texts = pd.Series([
            "good product", "bad product", "great service", "poor quality",
            "excellent", "terrible", "love it", "hate it", "fantastic", "awful",
        ])
        labels = pd.Series(
            ["pos", "neg", "pos", "neg", "pos", "neg", "pos", "neg", "pos", "neg"]
        )
        c.fit(texts, labels)
        cm = np.array([[3, 2], [1, 4]], dtype=int)
        fig = c.plot_confusion_matrix(cm)
        assert isinstance(fig, plt.Figure)

    def test_predict_before_fit_raises_runtimeerror(self):
        c = TextClassifier()
        with pytest.raises(RuntimeError, match="must be fitted before predict"):
            c.predict(pd.Series(["some text"]))


class TestKeywordExtractor:
    def test_extract_returns_dict_of_lists(self):
        k = KeywordExtractor()
        texts = pd.Series([
            "machine learning and natural language processing",
            "deep learning for computer vision",
        ])
        result = k.extract(texts)
        assert isinstance(result, dict)
        assert 0 in result
        assert 1 in result
        assert isinstance(result[0], list)

    def test_handles_empty_strings(self):
        k = KeywordExtractor()
        texts = pd.Series(["", "   ", "real text here"])
        result = k.extract(texts)
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_handles_various_text_inputs(self):
        k = KeywordExtractor()
        texts = pd.Series([
            "short",
            "A sentence with some important keywords for testing extraction.",
            "",
        ])
        result = k.extract(texts)
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_handles_text_with_only_stop_words(self):
        k = KeywordExtractor()
        texts = pd.Series(["a an the of it is at by to"])
        result = k.extract(texts)
        assert isinstance(result, dict)
        assert 0 in result
        assert result[0] == []


class TestTextEmbedder:
    def test_embed_returns_array_of_correct_shape(self):
        e = TextEmbedder()
        texts = pd.Series(["hello world", "test sentence"])
        result = e.embed(texts)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 50)

    def test_similarity_matrix_returns_correct_shape(self):
        e = TextEmbedder()
        texts_a = pd.Series(["hello", "world"])
        texts_b = pd.Series(["foo", "bar", "baz"])
        result = e.similarity_matrix(texts_a, texts_b)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)

    def test_find_similar_returns_list_of_tuples(self):
        e = TextEmbedder()
        texts = pd.Series(["apple orange", "banana grape", "hello world", "some text", "more words"])
        result = e.find_similar("query fruit", texts, top_k=3)
        assert isinstance(result, list)
        assert len(result) == 3
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], int)
            assert isinstance(item[1], float)


class TestNLPVisualizer:
    def test_wordcloud_returns_none_when_not_installed(self):
        v = NLPVisualizer()
        texts = pd.Series(["some sample text for word cloud"])
        result = v.wordcloud(texts)
        assert result is None

    def test_sentiment_distribution_returns_figure(self):
        v = NLPVisualizer()
        df = pd.DataFrame({
            "sentiment_label": ["Positive", "Negative", "Positive", "Neutral", "Positive"],
        })
        fig = v.sentiment_distribution(df)
        assert isinstance(fig, plt.Figure)
