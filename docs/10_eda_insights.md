# Phase 5 – Exploratory Data Analysis (EDA) Insights

## Overview
This document summarizes key insights discovered during exploratory data analysis on the customer-level dataset generated after feature engineering. The goal of this phase was to understand churn behavior, identify predictive signals, and guide model development decisions.

---

## 1. Churn Distribution
- The churn rate observed during EDA is moderately imbalanced.
- Churned customers form a significant minority, making classification feasible.
- Class imbalance was noted and later handled using stratified splits.

**Business implication:**  
Retention strategies should focus on a well-defined subset rather than the entire customer base.

---

## 2. Recency Analysis (Strongest Signal)
- Churned customers have **significantly higher recency** than active customers.
- Statistical testing confirms this difference is highly significant (p < 0.001).

**Insight:**  
Customers who have not purchased recently are far more likely to churn.

**Business action:**  
Trigger retention campaigns when recency exceeds a defined threshold (e.g., 60–90 days).

---

## 3. Frequency & Monetary Patterns
- Active customers purchase more frequently and spend more overall.
- Churned customers show:
  - Lower purchase counts
  - Lower lifetime value

**Insight:**  
High-frequency and high-value customers are naturally more loyal.

---

## 4. Recent Activity Windows
- Purchases in the last 30, 60, and 90 days show strong separation between churned and active users.
- Customers with **zero recent activity** have a high probability of churn.

**Insight:**  
Short-term inactivity is a reliable early warning signal.

---

## 5. Behavioral Features
- Customers with irregular purchase intervals churn more often.
- Larger and more consistent basket sizes correlate with retention.

---

## 6. Customer Segments (RFM-Based)
- "Champions" and "Loyal" segments have very low churn rates.
- "At Risk" and "Lost" segments show the highest churn concentration.

**Business implication:**  
Retention resources should prioritize “At Risk” customers before they transition to “Lost”.

---

## 7. Feature Correlation Summary
Top features correlated with churn:
1. Recency
2. Purchases_Last90Days
3. Frequency
4. TotalSpent
5. PurchaseVelocity

These findings guided feature selection in modeling phases.

---

## 8. Statistical Validation
- Multiple numerical features show statistically significant differences between churned and active customers.
- This validates that churn is **not random** and can be predicted.

---

## 9. Key Hypotheses Formed
- H1: Customers inactive for >90 days are significantly more likely to churn.
- H2: High-frequency customers rarely churn regardless of recency.
- H3: Recent purchasing behavior outweighs long-term historical behavior.

These hypotheses were later tested during modeling.

---

## 10. Limitations Identified
- Temporal splits significantly affect churn rate.
- Naïve random splits introduce data leakage.
- Seasonality may influence churn signals.

These limitations were explicitly addressed in later phases through temporal validation.

---

## Conclusion
EDA confirms that churn is predictable using customer behavior signals, especially recency and recent activity. Insights from this phase directly informed feature selection, model choice, and leakage-aware validation strategies.
