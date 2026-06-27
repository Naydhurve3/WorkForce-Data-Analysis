import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from wf_analysis.analysis.base import BaseAnalysis, AnalysisResult


class CareerPathAnalysis(BaseAnalysis):
    def run(self, df: pd.DataFrame) -> AnalysisResult:
        result = AnalysisResult(summary="Career Path Analysis")

        if "JobFunctionDescription" in df.columns:
            dist = df["JobFunctionDescription"].value_counts().to_dict()
            result.metrics["job_function_dist"] = dist

        if "Title" in df.columns:
            title_dist = df["Title"].value_counts().head(20).to_dict()
            result.metrics["top_titles"] = title_dist

        if "JobFamily" in df.columns:
            jf_dist = df["JobFamily"].value_counts().to_dict()
            result.metrics["job_family_dist"] = jf_dist

        if "Title" in df.columns:
            titles = df["Title"].dropna().unique()
            if len(titles) > 1:
                vectorizer = TfidfVectorizer(stop_words="english")
                tfidf = vectorizer.fit_transform(titles)
                sim_matrix = cosine_similarity(tfidf)
                n = min(len(titles), 5)
                top_similar = {}
                for i, title in enumerate(titles[:n]):
                    sims = list(enumerate(sim_matrix[i]))
                    sims.sort(key=lambda x: x[1], reverse=True)
                    top_similar[title] = [
                        {"title": titles[j], "similarity": float(s)}
                        for j, s in sims[1:4]
                    ]
                result.metrics["title_similarity"] = top_similar

        if "TenureYears" in df.columns and "JobFamily" in df.columns:
            tenure = df.groupby("JobFamily")["TenureYears"].mean().sort_values(ascending=False)
            result.metrics["avg_tenure_by_jobfamily"] = tenure.to_dict()

        result.summary = (
            f"{len(result.metrics.get('job_function_dist', {}))} job functions, "
            f"{len(result.metrics.get('top_titles', {}))} top titles."
        )
        return result
