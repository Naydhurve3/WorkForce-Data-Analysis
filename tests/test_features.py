import pandas as pd

from wf_analysis.features.demographic import DemographicTransformer
from wf_analysis.features.temporal import TemporalTransformer
from wf_analysis.features.categorical import CategoricalTransformer
from wf_analysis.features.embeddings import EmbeddingTransformer


class TestDemographicTransformer:
    def test_age_group_created(self, sample_df):
        df = sample_df.assign(DOB=pd.to_datetime([
            "1969-07-10", "1994-03-15", "1991-06-10",
            "1998-04-04", "1965-12-01"
        ]))
        t = DemographicTransformer()
        result = t.fit(df).transform(df)
        assert "AgeGroup" in result.columns
        assert "Generation" in result.columns
        assert "TenureYears" in result.columns


class TestTemporalTransformer:
    def test_tenure_computed(self, sample_df):
        t = TemporalTransformer()
        result = t.transform(sample_df)
        assert "TenureDays" in result.columns
        assert "TenureYears" in result.columns


class TestCategoricalTransformer:
    def test_job_family_mapped(self, sample_df):
        t = CategoricalTransformer()
        result = t.fit(sample_df).transform(sample_df)
        assert "JobFamily" in result.columns

    def test_region_mapped(self, sample_df):
        t = CategoricalTransformer()
        result = t.fit(sample_df).transform(sample_df)
        assert "Region" in result.columns

    def test_director_mapped_to_executive(self):
        t = CategoricalTransformer()
        assert t.map_job_family("Director of Sales") == "Executive & Leadership"

    def test_manager_mapped_to_production(self):
        t = CategoricalTransformer()
        assert t.map_job_family("Production Manager") == "Production"

    def test_region_ca_maps_to_west(self):
        t = CategoricalTransformer()
        assert t.map_region("CA") == "West"

    def test_region_tx_maps_to_south(self):
        t = CategoricalTransformer()
        assert t.map_region("TX") == "South"


class TestEmbeddingTransformer:
    def test_transform_creates_embedding_columns(self, sample_df):
        t = EmbeddingTransformer()
        result = t.fit(sample_df).transform(sample_df)
        assert "Title_Emb_0" in result.columns
        assert "FuncDesc_Emb_0" in result.columns
        assert "Division_Emb_0" in result.columns

    def test_with_missing_columns(self):
        df = pd.DataFrame({"EmpID": [1, 2]})
        t = EmbeddingTransformer()
        result = t.fit(df).transform(df)
        assert "Title_Emb_0" not in result.columns
