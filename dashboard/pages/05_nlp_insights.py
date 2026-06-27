import streamlit as st
import pandas as pd

from wf_analysis.nlp.preprocessor import TextPreprocessor
from wf_analysis.nlp.sentiment import SentimentAnalyzer
from wf_analysis.nlp.topic_model import TopicModeler
from wf_analysis.nlp.keywords import KeywordExtractor


def show(df=None):
    st.header("NLP Insights")

    if df is None or "TerminationDescription" not in df.columns:
        st.info("No termination description data found.")
        return

    non_empty = df["TerminationDescription"].dropna()
    non_empty = non_empty[non_empty.str.strip() != ""]
    st.metric("Non-empty Descriptions", len(non_empty))

    if len(non_empty) < 3:
        st.info("Insufficient descriptions for NLP analysis.")
        return

    tab1, tab2, tab3 = st.tabs(["Sentiment", "Topics", "Keywords"])

    with tab1:
        st.subheader("Sentiment Analysis")
        preprocessor = TextPreprocessor()
        sentiment = SentimentAnalyzer()

        cleaned = preprocessor.transform(non_empty)
        sent_df = sentiment.analyze(cleaned)
        st.dataframe(
            sent_df.describe().T, use_container_width=True
        )
        dist = sent_df["sentiment_label"].value_counts()
        st.bar_chart(dist)

    with tab2:
        st.subheader("Topic Modeling")
        topic_modeler = TopicModeler(n_topics=min(5, len(non_empty) // 2))
        topic_modeler.fit(non_empty)
        info = topic_modeler.get_topic_info()
        for t in info:
            st.write(f"**Topic {t['topic_id']}**: {', '.join(t['top_words'][:8])}")

    with tab3:
        st.subheader("Key Phrases")
        extractor = KeywordExtractor()
        keywords = extractor.extract(non_empty.head(20))
        for doc_id, phrases in keywords.items():
            st.write(f"Doc {doc_id}: {', '.join(p for p, s in phrases[:5])}")
