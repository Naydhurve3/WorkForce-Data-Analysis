from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnalysisResult:
    name: str = ""
    summary: str = ""
    description: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    figures: list[str] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
    recommendations: list[dict] = field(default_factory=list)
