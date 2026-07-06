"""Phase A.2: Compute majority/random/frequency baselines per outcome."""
import json, numpy as np, pandas as pd
from pathlib import Path

DATA = Path("data") / "interaction"
OUT = DATA / "baselines.json"
RAW = Path("data") / "raw" / "employee_data.csv"

raw = pd.read_csv(RAW)

# Is terminated
term = raw['EmployeeStatus'].str.lower().str.contains('terminat').astype(int)
n = len(term)
p_minor = term.mean()
baselines = {
    "is_terminated": {
        "majority": round(max(p_minor, 1 - p_minor), 4),
        "random": 0.5,
        "frequency": round(p_minor**2 + (1-p_minor)**2, 4),
        "minority_rate": round(p_minor, 4),
        "n": int(n),
        "n_minority": int(term.sum()),
        "n_majority": int(n - term.sum()),
    }
}

# PayZone (3-class)
pz = raw['PayZone'].fillna('Unknown')
pz_counts = pz.value_counts()
pz_props = pz_counts / n
baselines["PayZone_encoded"] = {
    "majority": round(pz_counts.max() / n, 4),
    "random": round(1 / 3, 4),
    "frequency": round(sum(p**2 for p in pz_props), 4),
    "n": int(n),
    "class_counts": pz_counts.to_dict(),
}

# PerfScore
ps = raw['Current Employee Rating'].fillna(0).astype(int)
ps_counts = ps.value_counts()
ps_props = ps_counts / n
baselines["PerfScore"] = {
    "majority": round(ps_counts.max() / n, 4),
    "random": round(1 / len(ps_counts), 4),
    "frequency": round(sum(p**2 for p in ps_props), 4),
    "n": int(n),
    "class_counts": ps_counts.sort_index().to_dict(),
}

# SeniorityLevel (from feature space)
from wf_analysis.data.loader import DataLoader
from wf_analysis.interaction.config import InteractionConfig
from wf_analysis.interaction.features import FeatureEngineer
cfg = InteractionConfig()
rr = DataLoader.load(cfg.raw_path, validate=False)
eng = FeatureEngineer(cfg)
fdf = eng.compute_all(rr)
sl = fdf['SeniorityLevel']
sl_counts = sl.value_counts()
sl_props = sl_counts / n
baselines["SeniorityLevel"] = {
    "majority": round(sl_counts.max() / n, 4),
    "random": round(1 / len(sl_counts), 4),
    "frequency": round(sum(p**2 for p in sl_props), 4),
    "n": int(n),
    "class_counts": sl_counts.sort_index().to_dict(),
}

# is_minority_dept (from model results confusion matrix)
with open(DATA / "model_results.json") as f:
    mr = json.load(f)
cm = list(mr.get("is_minority_dept", {}).values())[0].get("cm", [[0,0],[0,0]])
cm_arr = np.array(cm)
y_counts = cm_arr.sum(axis=1)
p_minor_dept = y_counts[1] / y_counts.sum()
baselines["is_minority_dept"] = {
    "majority": round(max(p_minor_dept, 1 - p_minor_dept), 4),
    "random": 0.5,
    "frequency": round(p_minor_dept**2 + (1-p_minor_dept)**2, 4),
    "minority_rate": round(p_minor_dept, 4),
    "n": int(y_counts.sum()),
}

with open(OUT, "w") as f:
    json.dump(baselines, f, indent=2)
print(f"Baselines saved to {OUT}")
for k, v in baselines.items():
    print(f"  {k:25s} majority={v['majority']:.3f} random={v['random']:.3f} freq={v['frequency']:.3f}")
