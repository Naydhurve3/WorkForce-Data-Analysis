# Cultural Context Modeling for Bias-Aware HR Analytics

## The Core Insight

> "In Japan, most people are introverted, so their hiring process is completely
> different. They don't want to only choose extroverts. In Western/American
> culture, they mostly prefer people who can openly express themselves and
> convince others — even if they don't actually know much about the work."

This is not just a cultural observation — it's a **measurable bias in HR
decision-making** that affects company performance.

## Cultural Dimensions Framework (Hofstede)

| Dimension | Japan | USA | Impact on Hiring |
|-----------|-------|-----|------------------|
| Individualism | Low (46) | High (91) | USA: values self-promotion; Japan: values group harmony |
| Uncertainty Avoidance | High (92) | Moderate (46) | Japan: prefers structured interviews; USA: flexible assessment |
| Long-term Orientation | High (88) | Moderate (26) | Japan: values loyalty & tenure; USA: values immediate skills |
| Indulgence | Low (42) | High (68) | USA: values optimism & confidence; Japan: values modesty |

## How This Affects Capability Assessment

### Western (Extrovert-Preferred) Context
- **Valued traits**: Verbal fluency, confidence, assertiveness, eye contact,
  firm handshake, personal achievements
- **Biased against**: Reserved candidates, thoughtful pauses, modesty about
  achievements, indirect communication
- **Missed capabilities**: Deep technical skills, team collaboration,
  long-term reliability, crisis composure

### Japanese (Introvert-Compatible) Context
- **Valued traits**: Listening skills, group orientation, humility, accuracy,
  reliability, tenure potential
- **Biased against**: Outspoken candidates, direct disagreement, immediate
  self-promotion, changing jobs frequently
- **Missed capabilities**: Innovation leadership, cross-functional
  communication, change management

## The Cultural Parameterization Module

### Design

```python
class CulturalContext:
    def __init__(self, name="Western"):
        self.params = self.get_default_params(name)

    def get_default_params(self, name):
        profiles = {
            "Western": {
                "extroversion_weight": 0.8,    # How much extroversion matters
                "self_promotion_bonus": 0.3,    # Bonus for articulating achievements
                "modesty_penalty": -0.2,        # Penalty for understating ability
                "directness_preference": 0.7,   # Prefer direct communication
                "group_orientation": 0.3,       # Value individual achievements more
                "tenure_value": 0.2,            # Less weight on long tenure
                "education_weight": 0.3,        # Moderate weight on credentials
            },
            "Japan": {
                "extroversion_weight": 0.3,     # Introversion is normal
                "self_promotion_bonus": -0.1,   # Self-promotion seen negatively
                "modesty_bonus": 0.3,           # Modesty is a positive signal
                "directness_preference": 0.3,   # Indirect communication valued
                "group_orientation": 0.8,       # Group harmony valued highly
                "tenure_value": 0.7,            # Long tenure is very valuable
                "education_weight": 0.6,        # Higher weight on credentials
            },
        }
        return profiles.get(name, profiles["Western"])

    def adjust_capability_score(self, raw_scores, employee_profile):
        """Adjust capability assessment based on cultural context."""
        adjusted = {}
        for dimension, score in raw_scores.items():
            if dimension == "communication_style":
                extroversion = employee_profile.get("extroversion_score", 0.5)
                weight = self.params["extroversion_weight"]
                # Extroversion-penalizing cultures will value balanced scores
                adjusted[dimension] = score * (1 - weight * (extroversion - 0.5))
            # ... more dimension adjustments
        return adjusted
```

### Usage in the Matching Engine

1. Start with raw employee capability profile (v2.0 features)
2. Apply cultural context adjustment
3. Match against role archetype requirements
4. Generate "Cultural Fit Score" alongside "Capability Score"
5. Decision-maker sees: "This employee scores 85/100 on capability but
   60/100 under Japan cultural lens — the difference reveals cultural bias"

### Research Questions for This Module

1. "Can we quantify the magnitude of cultural bias in hiring by comparing
   capability assessments under different cultural lenses?"
2. "Do certain roles show more cultural bias than others?"
3. "What is the cost of cultural bias — how many mis-hires could be avoided?"
4. "Can we recommend a 'culture-blind' assessment score that removes bias?"
