"""Shared pytest fixtures."""

import pandas as pd
import pytest

from wf_analysis.config import PipelineConfig


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "EmpID": [3427, 3428, 3429, 3430, 3431],
        "StartDate": pd.to_datetime(["2019-09-20", "2023-02-11", "2018-12-10", "2021-06-21", "2019-06-29"]),
        "ExitDate": pd.to_datetime([None, None, None, None, None]),
        "Title": ["Production Technician I", "Production Technician I", "Area Sales Manager", "Area Sales Manager", "Area Sales Manager"],
        "Supervisor": ["Peter Oneill", "Renee Mccormick", "Crystal Walker", "Rebekah Wright", "Jason Kim"],
        "BusinessUnit": ["CCDR", "EW", "PL", "CCDR", "TNS"],
        "EmployeeStatus": ["Active", "Active", "Active", "Active", "Active"],
        "EmployeeType": ["Contract", "Contract", "Full-Time", "Contract", "Contract"],
        "PayZone": ["Zone C", "Zone A", "Zone B", "Zone A", "Zone A"],
        "EmployeeClassificationType": ["Temporary", "Part-Time", "Part-Time", "Full-Time", "Temporary"],
        "TerminationType": ["Unk", "Unk", "Unk", "Unk", "Unk"],
        "TerminationDescription": ["", "", "", "", ""],
        "DepartmentType": ["Production", "Production", "Sales", "Sales", "Sales"],
        "Division": ["Finance & Accounting", "Aerial", "General - Sga", "Finance & Accounting", "General - Con"],
        "DOB": pd.to_datetime(["1969-07-10", None, "1991-06-10", "1998-04-04", None]),
        "State": ["MA", "MA", "MA", "ND", "FL"],
        "JobFunctionDescription": ["Accounting", "Labor", "Assistant", "Clerk", "Laborer"],
        "GenderCode": ["Female", "Male", "Male", "Male", "Female"],
        "LocationCode": [34904, 6593, 2330, 58782, 33174],
        "RaceDesc": ["White", "Hispanic", "Hispanic", "Other", "Other"],
        "MaritalDesc": ["Widowed", "Widowed", "Widowed", "Single", "Married"],
        "Performance Score": ["Fully Meets", "Fully Meets", "Fully Meets", "Fully Meets", "Fully Meets"],
        "Current Employee Rating": [4, 3, 4, 2, 3],
    })


@pytest.fixture
def config() -> PipelineConfig:
    return PipelineConfig()
