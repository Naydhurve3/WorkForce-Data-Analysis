import pandas as pd
import networkx as nx

from wf_analysis.analysis.base import BaseAnalysis, AnalysisResult


class NetworkAnalysis(BaseAnalysis):
    def run(self, df: pd.DataFrame) -> AnalysisResult:
        result = AnalysisResult(summary="Organizational Network Analysis")

        if "Supervisor" not in df.columns or "EmpID" not in df.columns:
            return result

        G = nx.DiGraph()
        for _, row in df.iterrows():
            emp = str(row["EmpID"])
            sup = str(row["Supervisor"])
            G.add_node(emp)
            G.add_node(sup)
            G.add_edge(sup, emp)

        result.metrics["total_nodes"] = G.number_of_nodes()
        result.metrics["total_edges"] = G.number_of_edges()
        result.metrics["total_supervisors"] = df["Supervisor"].nunique()

        soc = df.groupby("Supervisor")["EmpID"].nunique()
        result.metrics["span_of_control"] = {
            "mean": float(soc.mean()),
            "median": float(soc.median()),
            "std": float(soc.std()),
            "min": int(soc.min()),
            "max": int(soc.max()),
        }

        try:
            betweenness = nx.betweenness_centrality(G)
            top_influencers = sorted(
                betweenness.items(), key=lambda x: x[1], reverse=True
            )[:10]
            result.metrics["top_influencers"] = [
                {"node": n, "betweenness": float(v)}
                for n, v in top_influencers
            ]
        except Exception:
            pass

        result.summary = (
            f"Org graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges. "
            f"Avg span of control: {soc.mean():.1f}."
        )
        return result
