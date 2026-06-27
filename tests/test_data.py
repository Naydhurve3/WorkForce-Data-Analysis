from wf_analysis.data.loader import DataLoader
from wf_analysis.data.cleaner import DataCleaner
from wf_analysis.data.validator import DataValidator
from wf_analysis.data.exporter import DataExporter


class TestDataLoader:
    def test_load_csv(self, sample_df, tmp_path):
        path = tmp_path / "test.csv"
        sample_df.to_csv(path, index=False)
        loaded = DataLoader.load(str(path), validate=False)
        assert loaded.shape == sample_df.shape

    def test_loading_handles_boolean_cols(self, sample_df, tmp_path):
        path = tmp_path / "test_bool.csv"
        sample_df.to_csv(path, index=False)
        loaded = DataLoader.load(str(path), validate=False)
        assert "EmpID" in loaded.columns


class TestDataCleaner:
    def test_remove_pii(self, sample_df):
        cleaned = DataCleaner.remove_pii(sample_df, columns=["EmpID"])
        assert "EmpID" not in cleaned.columns

    def test_standardize_dates(self, sample_df):
        cleaned = DataCleaner.standardize_dates(sample_df, ["StartDate", "ExitDate"])
        assert cleaned["StartDate"].dtype == "datetime64[ns]"


class TestDataValidator:
    def test_generate_report_returns_report(self, sample_df):
        report = DataValidator.generate_report(sample_df)
        assert report.shape == sample_df.shape

    def test_validate_schema_no_schema_path(self, sample_df):
        report = DataValidator.validate_schema(sample_df, schema_path=None)
        assert report.passed is True


class TestDataExporter:
    def test_to_csv(self, sample_df, tmp_path):
        path = tmp_path / "out.csv"
        DataExporter.to_csv(sample_df, path)
        assert path.exists()

    def test_to_parquet(self, sample_df, tmp_path):
        path = tmp_path / "out.parquet"
        DataExporter.to_parquet(sample_df, path)
        assert path.exists()
