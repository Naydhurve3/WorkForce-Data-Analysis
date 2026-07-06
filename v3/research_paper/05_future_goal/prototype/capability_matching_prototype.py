"""
v3.0 Prototype: Capability-to-Role Matching Engine
===================================================
A working prototype demonstrating how v2.0's employee profiles can be
used to match employees to roles based on capabilities — reducing
reliance on interview-only evaluation.

This prototype:
1. Loads employee profiles from v2.0 engineered features
2. Defines role archetypes with capability requirements
3. Scores employee-to-role fit using cosine similarity
4. Generates "what-if" placement recommendations
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = ROOT / "data" / "analysis"


class CapabilityProfiler:
    """Maps employee features to capability dimensions."""

    DIMENSIONS = {
        "technical": ["PerfScore", "PromotionRate", "Current Employee Rating"],
        "experience": ["TenureYears", "SeniorityLevel", "TenureVsAvg"],
        "leadership": ["SpanOfControl", "IsManager", "IsExecutive", "OrgLevel"],
        "diversity": ["DeptDiversityScore", "DeptGenderRatio"],
        "mobility": ["PromotionLag", "PromotionReadiness", "CareerStage"],
        "tenure_stability": ["IsLongTenure", "EngagementFlag"],
    }

    def __init__(self):
        self._feature_stats = {}

    def fit(self, df):
        """Store normalization stats from the dataset."""
        all_features = []
        for dim, feats in self.DIMENSIONS.items():
            all_features.extend(feats)
        for feat in all_features:
            if feat in df.columns:
                col = df[feat]
                self._feature_stats[feat] = {
                    "mean": col.mean(),
                    "std": col.std() if col.std() > 0 else 1.0,
                    "min": col.min(),
                    "max": col.max(),
                }
        return self

    def profile(self, employee_row):
        """Compute capability profile vector for a single employee."""
        profile = {}
        for dim, feats in self.DIMENSIONS.items():
            scores = []
            for feat in feats:
                raw = employee_row.get(feat)
                if raw is None or pd.isna(raw):
                    continue
                stats = self._feature_stats.get(feat)
                if stats is None:
                    continue
                # z-score normalization
                z = (raw - stats["mean"]) / stats["std"]
                # clip to [-3, 3] and rescale to [0, 1]
                normalized = max(0, min(1, (z + 3) / 6))
                scores.append(normalized)
            profile[dim] = np.mean(scores) if scores else 0.0
        return profile


class RoleArchetypeModel:
    """Defines role archetypes with capability requirements."""

    ARCHETYPES = {
        "Executive_Leader": {
            "technical": 0.5,
            "experience": 0.8,
            "leadership": 0.9,
            "diversity": 0.6,
            "mobility": 0.3,
            "tenure_stability": 0.8,
        },
        "Engineering_IC": {
            "technical": 0.9,
            "experience": 0.5,
            "leadership": 0.2,
            "diversity": 0.4,
            "mobility": 0.5,
            "tenure_stability": 0.4,
        },
        "Sales_Manager": {
            "technical": 0.4,
            "experience": 0.7,
            "leadership": 0.7,
            "diversity": 0.5,
            "mobility": 0.6,
            "tenure_stability": 0.5,
        },
        "HR_Specialist": {
            "technical": 0.4,
            "experience": 0.5,
            "leadership": 0.3,
            "diversity": 0.9,
            "mobility": 0.5,
            "tenure_stability": 0.6,
        },
        "Support_IC": {
            "technical": 0.5,
            "experience": 0.4,
            "leadership": 0.1,
            "diversity": 0.5,
            "mobility": 0.3,
            "tenure_stability": 0.6,
        },
    }

    def get_requirements(self, archetype_name):
        return self.ARCHETYPES.get(archetype_name, {})

    def list_archetypes(self):
        return list(self.ARCHETYPES.keys())


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(list(a.values()))
    b_arr = np.array(list(b.values()))
    dot = np.dot(a_arr, b_arr)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    return dot / norm if norm > 0 else 0.0


def match_employees_to_roles(employee_profiles, role_model, top_k=3):
    """Score each employee against each role archetype."""
    results = []
    for emp_id, profile in employee_profiles.items():
        for archetype in role_model.list_archetypes():
            requirements = role_model.get_requirements(archetype)
            score = cosine_similarity(profile, requirements)
            results.append({
                "employee_id": emp_id,
                "archetype": archetype,
                "match_score": round(score, 3),
                "current_match": "",
            })
    return pd.DataFrame(results).sort_values(
        ["employee_id", "match_score"], ascending=[True, False]
    )


def generate_recommendations(match_df, top_k=3):
    """Generate top-K role recommendations per employee."""
    recs = []
    for emp_id, group in match_df.groupby("employee_id"):
        top = group.head(top_k)
        for _, row in top.iterrows():
            recs.append({
                "employee_id": emp_id,
                "recommended_role": row["archetype"],
                "confidence": row["match_score"],
                "rank": _,
            })
    return pd.DataFrame(recs)


def main():
    print("=" * 60)
    print("v3.0 Capability-to-Role Matching Prototype")
    print("=" * 60)

    # Load v2.0 employee data (use the easiest available dataset)
    paths = [
        DATA_DIR / "02_attrition" / "dataset.parquet",
        DATA_DIR / "04_performance" / "dataset.parquet",
    ]
    df = None
    for p in paths:
        if p.exists():
            df = pd.read_parquet(p)
            print(f"\nLoaded: {p.name} ({len(df)} employees)")
            break

    if df is None:
        print("No v2.0 analysis datasets found. Using sample mode.")
        np.random.seed(42)
        sample_data = {
            "EmpID": range(1, 21),
            "PerfScore": np.random.uniform(1, 5, 20),
            "PromotionRate": np.random.uniform(0, 1, 20),
            "Current Employee Rating": np.random.randint(1, 6, 20),
            "TenureYears": np.random.uniform(0, 15, 20),
            "SeniorityLevel": np.random.randint(1, 6, 20),
            "TenureVsAvg": np.random.uniform(-2, 2, 20),
            "SpanOfControl": np.random.randint(0, 15, 20),
            "IsManager": np.random.randint(0, 2, 20),
            "IsExecutive": np.random.randint(0, 2, 20),
            "OrgLevel": np.random.randint(1, 6, 20),
            "DeptDiversityScore": np.random.uniform(0, 1, 20),
            "DeptGenderRatio": np.random.uniform(0, 1, 20),
            "PromotionLag": np.random.uniform(0, 5, 20),
            "PromotionReadiness": np.random.uniform(0, 1, 20),
            "CareerStage": np.random.uniform(1, 5, 20),
            "IsLongTenure": np.random.randint(0, 2, 20),
            "EngagementFlag": np.random.randint(0, 2, 20),
        }
        df = pd.DataFrame(sample_data)

    # Build profiler and role model
    profiler = CapabilityProfiler()
    profiler.fit(df)
    role_model = RoleArchetypeModel()

    # Profile all employees
    employee_profiles = {}
    for _, row in df.iterrows():
        emp_id = row.get("EmpID", _)
        employee_profiles[emp_id] = profiler.profile(row)

    # Match
    match_df = match_employees_to_roles(employee_profiles, role_model)
    recommendations = generate_recommendations(match_df)

    # Show results
    print(f"\nProfiled {len(employee_profiles)} employees")
    print(f"Role archetypes: {role_model.list_archetypes()}")
    print(f"\nTop Recommendations (first 10):")
    print("-" * 60)
    for _, row in recommendations.head(10).iterrows():
        print(f"  Employee {row['employee_id']} -> {row['recommended_role']} "
              f"(confidence: {row['confidence']:.2f})")

    # Show a detailed example
    emp_id = list(employee_profiles.keys())[0]
    print(f"\nDetailed Profile for Employee {emp_id}:")
    print("-" * 60)
    for dim, val in employee_profiles[emp_id].items():
        print(f"  {dim}: {val:.3f}")

    emp_matches = match_df[match_df["employee_id"] == emp_id].head(3)
    print(f"\nBest Role Matches for Employee {emp_id}:")
    for _, row in emp_matches.iterrows():
        print(f"  {row['archetype']}: {row['match_score']:.3f}")

    print(f"\n{'=' * 60}")
    print("Prototype complete. Ready for extension with:")
    print("  - Actual v2.0 feature data (24+ features)")
    print("  - Role requirements from real job descriptions")
    print("  - Interview bias detection module")
    print("  - Cultural context adjustment module")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
