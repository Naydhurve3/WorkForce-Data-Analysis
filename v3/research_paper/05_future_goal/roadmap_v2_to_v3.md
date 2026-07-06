# v2.0 → v3.0 Migration Roadmap

## Current State (v2.0 — Complete)
- ✅ 7-phase interaction mining pipeline
- ✅ 1,755 pairwise interaction tests
- ✅ 13 models across 5 outcomes
- ✅ 60 publication-ready figures
- ✅ HTML report
- ✅ 58 passing tests
- ✅ 10 analysis notebooks

## Intermediate State (v2.1 — Paper-Ready)
- ✅ Cleaned up failing tests
- ✅ LICENSE + CITATION.cff added
- ✅ Research data extracted to CSVs
- ✅ 11 curated figures for paper
- ⬜ Paper written and posted to arXiv
- ⬜ Dataset made citable (Zenodo or similar)

## Future State (v3.0 — Capability Matching)

### Phase A: Role Archetype Model (Week 1-2)
- [ ] Cluster all job titles from v2.0 features
- [ ] Define capability dimensions (technical, social, domain)
- [ ] Assign dimension weights per archetype
- [ ] Validate against known performance outcomes

### Phase B: Capability Profiler (Week 2-3)
- [ ] Map existing 53 features to capability dimensions
- [ ] Build profile vector for each employee
- [ ] Handle missing data via imputation
- [ ] Normalize profiles to [0,1] per dimension

### Phase C: Matching Engine (Week 3-4)
- [ ] Cosine similarity between profile and requirements
- [ ] ML-based outcome prediction for role changes
- [ ] "What-if" interface: simulate role changes
- [ ] Confidence intervals from interaction data

### Phase D: Bias Detection (Week 4-5)
- [ ] Collect/simulate interview score data
- [ ] Compare interview vs. actual performance
- [ ] Train fairness audit model
- [ ] Generate audit reports per demographic group

### Phase E: Cultural Context Module (Week 5-6)
- [ ] Implement Hofstede cultural dimensions
- [ ] Build configurable parameter sliders
- [ ] Map capabilities to cultural valuation functions
- [ ] Japan mode vs. Western mode comparison

### Phase F: Integration & Dashboard (Week 6-7)
- [ ] Integrate all modules into a unified pipeline
- [ ] Build recommendation dashboard
- [ ] Add explainability for each recommendation
- [ ] Final testing and documentation

## Resources Needed
- Python 3.10+ (already available)
- scikit-learn (already available)
- Additional: Hofstede cultural data, job description data
- Optional: Interview score data (real or synthetic)
