import pytest

from wf_analysis.config import PipelineConfig, DataConfig


class TestPipelineConfig:
    def test_default_config_creates(self):
        config = PipelineConfig()
        assert config.data.raw_path.endswith("employee_data.csv")

    def test_from_yaml_invalid_path(self):
        with pytest.raises(FileNotFoundError):
            PipelineConfig.from_yaml("nonexistent.yaml")


class TestDataConfig:
    def test_pii_columns_default(self):
        config = DataConfig(raw_path="a", processed_dir="b", schema_path="c")
        assert "FirstName" in config.pii_columns
