# v3.0 Capability-to-Role Matching Engine — Design Blueprint

## Overview

The core insight behind v3.0 is that **traditional hiring interviews are biased
toward extroversion and cultural norms** (e.g., Western preference for
self-promotion, Japanese preference for modesty), leading to systematic
misjudgment of employee capabilities.

v3.0 builds on v2.0's workforce data and interaction mining framework to
create a **data-driven capability assessment and role-matching system** that
evaluates employees holistically — reducing reliance on interview-only
evaluation.

## Architecture

```
v2.0 Outputs                              v3.0 Modules
=============                              ============

Employee Profiles         ──►  Capability Profiler
(53 features from v2.0)        │
                                ├─ Technical dimension: performance,
                                │   tenure, rating, promotion history
                                ├─ Social dimension: diversity exposure,
                                │   span of control, mobility
                                └─ Domain dimension: department, division,
                                    job family, region

Job Title Clusters        ──►  Role Archetype Model
(from v2.0 features)            │
                                ├─ Cluster all job roles into archetypes
                                ├─ Define capability requirements per
                                │   archetype (min/max/optimal ranges)
                                └─ Weight dimensions by importance

Interaction Insights      ──►  What-If Simulator
(from v2.0 mining)              │
                                ├─ "If we move employee X to role Y,
                                │   what is the predicted outcome?"
                                ├─ Performance, retention, satisfaction
                                └─ Confidence interval based on
                                    historical interaction data

Bias Detection            ──►  Fairness Auditor
(new module)                    │
                                ├─ Compare interview scores vs. actual
                                │   performance by demographic group
                                ├─ Detect cultural preference patterns
                                ├─ Flag extroversion/introversion bias
                                └─ Recommend blind spots in hiring

Cultural Context          ──►  Cultural Parameterization
(new module)                    │
                                ├─ Adjustable cultural parameters:
                                │   - Extroversion preference weight
                                │   - Directness communication style
                                │   - Hierarchy respect level
                                │   - Group vs. individual orientation
                                ├─ Japan mode, Western mode, custom
                                └─ "How would this profile be valued
                                    under different cultural norms?"
```

## Key Research Questions

1. **Capability matching**: "Can we predict role success from existing employee
   data, reducing reliance on interview-only assessment?"

2. **Bias detection**: "Can we quantify interview bias by comparing interview
   scores to actual performance, controlling for demographics?"

3. **Cultural fairness**: "How do cultural norms (introversion vs. extroversion
   preference) affect hiring fairness, and can we model this?"

4. **Interaction-aware simulation**: "What is the predicted outcome if we move
   employee X to role Y, leveraging interaction patterns discovered in v2.0?"

## Implementation Phases

### Phase A: Role Archetype Model (Week 1-2)
- Cluster all 32 job titles using k-means on feature vectors
- Define capability dimensions (technical, social, domain)
- Assign weight importance per archetype
- Validate against known performance outcomes

### Phase B: Capability Profiler (Week 2-3)
- Map existing 53 features to capability dimensions
- Build a profile vector per employee
- Handle missing data via v2.0 imputation methods
- Normalize profiles to [0, 1] per dimension

### Phase C: Matching Engine (Week 3-4)
- Cosine similarity between employee profile and role requirements
- Optional: ML-based outcome prediction for placement scenarios
- "What-if" interface: adjust role assignment, see predicted outcomes
- Confidence intervals from historical interaction data

### Phase D: Bias Detection (Week 4-5)
- Collect interview score data (simulated or real)
- Compare to actual performance outcomes
- Train bias detection model (demographic parity test)
- Generate fairness audit reports

### Phase E: Cultural Context Module (Week 5-6)
- Research cultural dimension frameworks (Hofstede, Trompenaars)
- Define cultural parameter sliders
- Map capabilities to cultural valuation functions
- Build comparison views: "Japan lens" vs "Western lens"
