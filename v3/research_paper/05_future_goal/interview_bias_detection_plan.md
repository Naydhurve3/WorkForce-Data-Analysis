# Interview Bias Detection — Module Design

## Problem

Traditional hiring interviews are subject to systematic biases:
- **Extroversion bias**: Outgoing, talkative candidates are rated higher
  regardless of actual capability
- **Cultural bias**: Western interview culture rewards self-promotion; East
  Asian culture (e.g., Japan) rewards modesty and group-orientation
- **Confirmation bias**: Interviewers seek evidence confirming first impressions
- **Similarity bias**: Interviewers favor candidates similar to themselves

## Detection Approach

### Data Requirements
- Interview scores (structured, per-criterion)
- Actual performance data (post-hire, 6-12 months)
- Demographic data (gender, race, age, cultural background)
- Interviewer metadata (demographics, department, role)

### Methodology

```
1. Train a baseline model predicting performance from interview scores only
2. Add demographic features → observe if bias exists
3. Train a fairness-aware model → compare predictions
4. Generate audit report with per-group metrics
```

### Fairness Metrics
- **Demographic Parity**: P(predicted positive | group A) = P(predicted positive | group B)
- **Equal Opportunity**: TPR for group A = TPR for group B
- **Equalized Odds**: TPR + FPR equal across groups
- **Disparate Impact**: Ratio of favorable outcomes between groups

### Detection Pipeline

```
Interview Scores ──┐
                   ├──► Bias Detector ──► Audit Report
Demographics ──────┘
                           │
                           ▼
                   ┌───────────────┐
                   │ Bias Found?   │
                   ├───────┬───────┤
                   │ YES   │ NO    │
                   ▼       ▼
            Adjust weights  Deploy
            or retrain      as-is
```

## Prototype Scope

For the initial prototype:
- Use simulated interview scores based on existing employee data
- Compare interview rankings vs. actual performance rankings
- Detect bias across gender, race, and department boundaries
- Generate a fairness audit report with visualization
