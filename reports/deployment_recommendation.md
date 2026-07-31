# Deployment Recommendation Report
## Loan Risk Assessment System

---

## 1. Executive Summary

This report compares three classification models trained to predict loan
approval outcomes: **Logistic Regression**, **Decision Tree**, and
**Random Forest**. Based on a comprehensive evaluation across Accuracy,
Precision, Recall, F1 Score, and ROC-AUC, we recommend deploying the
**Logistic Regression** model in the banking production environment.

---

## 2. Model Comparison Table

| Model               |   Accuracy |   Precision |   Recall |   F1 Score |   ROC AUC |
|:--------------------|-----------:|------------:|---------:|-----------:|----------:|
| Logistic Regression |     0.7154 |      0.8132 |   0.8043 |     0.8087 |    0.6739 |
| Random Forest       |     0.7317 |      0.8041 |   0.8478 |     0.8254 |    0.668  |
| Decision Tree       |     0.7073 |      0.8111 |   0.7935 |     0.8022 |    0.6648 |

---

## 3. Metric-by-Metric Analysis

### Precision
Precision measures the proportion of predicted approvals that are truly
creditworthy. In a banking context, low precision means the bank
extends credit to applicants who are likely to default, directly
increasing **credit risk and non-performing assets (NPA)**.

### Recall
Recall measures the proportion of truly creditworthy applicants that the
model correctly approves. Low recall means the bank **rejects good
customers**, resulting in lost business opportunity and poor customer
experience.

### ROC-AUC
ROC-AUC summarizes the model's ability to discriminate between approved
and rejected applicants across all classification thresholds, making it
the most robust metric for comparing models independent of a specific
decision threshold — which is critical since banks often adjust
thresholds based on risk appetite and economic conditions.

---

## 4. Recommended Model: Logistic Regression

**Key metrics for Logistic Regression:**

| Metric | Score |
|---|---|
| Accuracy | 0.7154 |
| Precision | 0.8132 |
| Recall | 0.8043 |
| F1 Score | 0.8087 |
| ROC AUC | 0.6739 |

### Justification

1. **Highest discriminative power**: Logistic Regression achieved the highest
   ROC-AUC score among all candidates, indicating superior ability to
   rank applicants by true creditworthiness across all decision thresholds.
2. **Balanced precision-recall trade-off**: The model does not
   over-optimize for one metric at the expense of the other, which is
   essential in banking where both false approvals (credit risk) and
   false rejections (lost revenue) carry real financial cost.
3. **Interpretability & auditability**: Tree-based models (Decision
   Tree / Random Forest) provide feature importance rankings that support
   regulatory requirements for explainable credit decisions, while
   Logistic Regression provides transparent, coefficient-based
   interpretability valued in credit scoring.
4. **Robustness**: Ensemble methods such as Random Forest reduce
   overfitting risk relative to a single Decision Tree by averaging
   predictions across many de-correlated trees, generally yielding more
   stable performance on unseen applicant populations.

---

## 5. Risk Considerations for Production Deployment

- **Model monitoring**: Continuously monitor for data drift in applicant
  demographics and macroeconomic conditions (interest rates, unemployment)
  that could degrade model performance over time.
- **Fairness & compliance**: Periodically audit approval rates across
  protected attributes (e.g. gender) to ensure compliance with fair
  lending regulations.
- **Threshold calibration**: The deployed decision threshold should be
  calibrated to the bank's specific risk appetite rather than using the
  default 0.5 cutoff, and should be reviewed periodically by the credit
  risk committee.
- **Human-in-the-loop**: Borderline predictions (probabilities near the
  decision threshold) should be routed to human underwriters for manual
  review rather than fully automated decisioning.

---

## 6. Conclusion

The **Logistic Regression** model is recommended for deployment in the banking
loan approval workflow based on its superior ROC-AUC and balanced
performance across all evaluation metrics. Deployment should be
accompanied by ongoing monitoring, periodic retraining, and a
human-in-the-loop review process for borderline cases.

---
*Report generated automatically by the Loan Risk Assessment System pipeline.*
