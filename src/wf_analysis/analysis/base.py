from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import matplotlib.pyplot as plt


@dataclass
class AnalysisResult:
    summary: str = ""
    metrics: dict = field(default_factory=dict)
    plots: list[plt.Figure] = field(default_factory=list)


class BaseAnalysis(ABC):
    @abstractmethod
    def run(self, df) -> AnalysisResult:
        ...

    def plot(self, result: AnalysisResult) -> plt.Figure | None:
        return result.plots[0] if result.plots else None

    def summarize(self, result: AnalysisResult) -> str:
        return result.summary
