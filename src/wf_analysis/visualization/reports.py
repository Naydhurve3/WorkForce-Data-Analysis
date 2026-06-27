from pathlib import Path

from wf_analysis.analysis.base import AnalysisResult
from wf_analysis.visualization.theme import Theme


class ReportGenerator:
    @staticmethod
    def generate_html(
        analyses: dict[str, AnalysisResult],
        figures: list | None = None,
        output_path: str = "reports/analysis_report.html",
    ) -> str:
        Theme.set_style()
        sections = []
        for name, result in analyses.items():
            sections.append(f"""
            <section>
                <h2>{name}</h2>
                <p>{result.summary}</p>
                <h3>Metrics</h3>
                <pre>{result.metrics}</pre>
            </section>
            """)

        html = f"""<!DOCTYPE html>
<html>
<head><title>WorkForce Analysis Report</title>
<style>
    body {{ font-family: Arial, sans-serif; margin: 40px; }}
    h1 {{ color: #2E86AB; }}
    section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }}
</style>
</head>
<body>
    <h1>WorkForce Data Analysis Report</h1>
    {''.join(sections)}
</body>
</html>"""

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html)
        return html
