"""DAG-based pipeline orchestrator for multi-stage data processing."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import pandas as pd

from wf_analysis.config import PipelineConfig


class PipelineError(Exception):
    """Raised when pipeline execution fails."""


class PipelineStage:
    def __init__(
        self,
        name: str,
        func: Callable,
        dependencies: list[str] | None = None,
        output_key: str | None = None,
    ):
        self.name = name
        self.func = func
        self.dependencies = dependencies or []
        self.output_key = output_key or name


class Pipeline:
    def __init__(self):
        self._stages: OrderedDict[str, PipelineStage] = OrderedDict()

    def add_stage(
        self,
        name: str,
        func: Callable,
        dependencies: list[str] | None = None,
        output_key: str | None = None,
    ) -> "Pipeline":
        if name in self._stages:
            raise ValueError(f"Stage '{name}' already exists")
        self._stages[name] = PipelineStage(name, func, dependencies, output_key)
        return self

    def _validate_dag(self) -> None:
        for name, stage in self._stages.items():
            for dep in stage.dependencies:
                if dep not in self._stages:
                    raise ValueError(
                        f"Stage '{name}' depends on '{dep}' which does not exist"
                    )

    def _topological_sort(self) -> list[str]:
        self._validate_dag()
        visited: set[str] = set()
        sorted_order: list[str] = []

        def dfs(node: str) -> None:
            if node in visited:
                return
            visited.add(node)
            stage = self._stages[node]
            for dep in stage.dependencies:
                dfs(dep)
            sorted_order.append(node)

        for name in self._stages:
            dfs(name)
        return sorted_order

    def run(
        self,
        config: PipelineConfig,
        df: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:
        order = self._topological_sort()
        results: dict[str, pd.DataFrame] = {}

        for name in order:
            stage = self._stages[name]
            dep_results = {dep: results[dep] for dep in stage.dependencies}

            if not dep_results:
                result = stage.func(df, config)
            else:
                result = stage.func(**dep_results, config=config)

            output_key = stage.output_key
            results[output_key] = result

        return results
