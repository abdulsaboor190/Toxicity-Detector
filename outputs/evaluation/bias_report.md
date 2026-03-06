## BIAS AUDIT REPORT -- Chat Toxicity Detector
### Model: distilbert-base-uncased fine-tuned on Jigsaw Toxic Comments
### Date: 2026-03-05

---

#### Overall Performance
- Overall AUC: 0.8854
- Mean Subgroup AUC: 0.8123

#### Subgroup AUC Results
- Best performing group: other_race_or_ethnicity (AUC = 1.0000)
- Worst performing group: bisexual (AUC = 0.6000)
- Groups with significant bias (Subgroup AUC < 0.835): bisexual, muslim, transgender, black, white, homosexual_gay_or_lesbian, heterosexual, latino, buddhist, jewish, psychiatric_or_mental_illness

#### Key Findings
1. **Over-flagging**: The most over-flagged identity group is `bisexual` (FPR = 0.2000 vs overall FPR = 0.0463).
2. **Under-flagging**: The least protected group is `muslim` (BNSP AUC = 0.7860).
3. **Dominant bias type**: BPSN (over-flagging neutral mentions) is the dominant bias type. Mean BPSN AUC = 0.8511, Mean BNSP AUC = 0.8640.

#### Known Limitations
- This model was trained on Wikipedia comments and may not generalize to other domains (social media, gaming chat, etc.)
- Annotation reflects the perspectives of the annotators (predominantly English-speaking, US-based)
- Subgroup metrics are only computed for groups with > 100 mentions in the bias dataset
- Identity columns are based on annotator agreement (threshold >= 0.5) and may miss subtle mentions
- The bias dataset itself may contain annotation biases

#### Recommendations
- Apply threshold calibration per identity subgroup to equalize FPR across groups
- Consider debiasing techniques such as counterfactual data augmentation
- Regularly re-evaluate on updated bias datasets as language patterns evolve
- For deployment: implement human-in-the-loop review for comments mentioning identity terms
- Given significant bias detected in 11 groups, additional debiasing is recommended before production deployment.
