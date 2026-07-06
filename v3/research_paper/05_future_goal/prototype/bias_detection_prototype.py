"""
v3.0 Prototype: Interview Bias Detection
=========================================
Demonstrates how to detect bias in hiring by comparing interview scores
against actual performance, controlling for demographic factors.

This prototype uses simulated data to show the methodology. In production,
it would use real interview scores + performance data + demographics.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


def simulate_hiring_data(n=1000, seed=42):
    """Simulate hiring data with known bias patterns."""
    rng = np.random.default_rng(seed)

    data = pd.DataFrame({
        "emp_id": range(n),
        "gender": rng.choice(["M", "F"], n),
        "race": rng.choice(["White", "Asian", "Black", "Hispanic"], n,
                           p=[0.6, 0.2, 0.1, 0.1]),
        "department": rng.choice(["Engineering", "Sales", "HR", "IT"], n),
        "interview_score": rng.uniform(1, 10, n),
        "actual_performance": np.zeros(n),
    })

    # True capability: 60% from interview + 40% hidden factors
    true_capability = 0.6 * data["interview_score"] / 10 + 0.4 * rng.uniform(0, 1, n)

    # Introduce gender bias: male candidates get +0.5 bonus to interview
    bias_mask = data["gender"] == "M"
    data.loc[bias_mask, "interview_score"] += 0.5 * rng.uniform(0.5, 1.5,
                                                                 bias_mask.sum())
    data["interview_score"] = data["interview_score"].clip(1, 10)

    # Actual performance reflects TRUE capability (unbiased)
    data["actual_performance"] = (true_capability > 0.5).astype(int)

    # Decision based on biased interview
    data["hired"] = (data["interview_score"] > 5.5).astype(int)

    return data


def detect_bias(data):
    """Compare interview-based predictions vs. actual performance by group."""
    print("=" * 60)
    print("BIAS DETECTION REPORT")
    print("=" * 60)

    # Method 1: Compare hiring rate vs. actual performance rate by group
    print("\n1. HIRING RATE vs PERFORMANCE RATE BY GENDER")
    print("-" * 40)
    for gender in ["M", "F"]:
        group = data[data["gender"] == gender]
        hire_rate = group["hired"].mean()
        perf_rate = group["actual_performance"].mean()
        bias_gap = hire_rate - perf_rate
        print(f"  {gender}: Hire={hire_rate:.1%}  Perf={perf_rate:.1%}  "
              f"Bias Gap={bias_gap:+.1%}")

    # Method 2: Model-based bias detection
    print("\n2. MODEL-BASED ANALYSIS")
    print("-" * 40)
    X = pd.get_dummies(data[["interview_score", "gender", "race"]],
                       drop_first=True)
    y = data["actual_performance"]

    model = LogisticRegression(class_weight="balanced")
    model.fit(X, y)
    y_pred = model.predict(X)

    # Accuracy by group
    for col in ["gender", "race"]:
        print(f"\n  Accuracy by {col}:")
        for val in data[col].unique():
            mask = data[col] == val
            acc = accuracy_score(data.loc[mask, "actual_performance"],
                                 y_pred[mask])
            print(f"    {val}: {acc:.1%}")

    # Method 3: Demographic parity check
    print("\n3. DEMOGRAPHIC PARITY CHECK")
    print("-" * 40)
    print("  (Expected: equal hire rates across groups)")
    for col in ["gender", "race"]:
        rates = data.groupby(col)["hired"].mean()
        min_rate, max_rate = rates.min(), rates.max()
        ratio = min_rate / max_rate if max_rate > 0 else 0
        print(f"  {col}: {rates.to_dict()}")
        print(f"  Disparate impact ratio: {ratio:.2f} "
              f"{'[BIAS DETECTED]' if ratio < 0.8 else '[OK]'}")

    return model


def main():
    print("Generating simulated hiring data...")
    data = simulate_hiring_data(n=2000)

    print(f"Dataset: {len(data)} candidates")
    print(f"Hired: {data['hired'].sum()} ({data['hired'].mean():.1%})")
    print(f"High performers: {data['actual_performance'].sum()} "
          f"({data['actual_performance'].mean():.1%})")

    model = detect_bias(data)

    print(f"\n{'=' * 60}")
    print("BIAS DETECTION SUMMARY")
    print("=" * 60)
    print("""
    Methodology demonstrates:
    1. Comparing hire rates vs. actual performance rates by group
    2. Model accuracy parity across demographic groups
    3. Disparate impact ratio (80% rule from US EEOC)

    For real deployment, this would use:
    - Actual interview scores from structured interviews
    - Actual performance evaluations (6-12 months post-hire)
    - Real demographic data
    - Multiple bias metrics (demographic parity, equal opportunity, etc.)
    - Cultural context adjustment module
    """)


if __name__ == "__main__":
    main()
