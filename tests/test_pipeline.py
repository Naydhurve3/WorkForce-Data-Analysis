import pytest

from wf_analysis.pipeline import Pipeline
from wf_analysis.config import PipelineConfig


class TestPipeline:
    def test_add_stage(self):
        p = Pipeline()
        p.add_stage("test", func=lambda df, config: df)
        assert "test" in p._stages

    def test_run(self, sample_df):
        p = Pipeline()
        p.add_stage("add_col", func=lambda df, config: df.assign(test_col=1))
        config = PipelineConfig()
        results = p.run(config, df=sample_df)
        assert "add_col" in results
        assert results["add_col"]["test_col"].iloc[0] == 1
