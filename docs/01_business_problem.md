# Business Problem Statement

## 1. Business Context
The e-commerce industry is highly competitive, and customer acquisition costs are significantly higher than customer retention costs. Studies show that acquiring a new customer can cost **5–25× more** than retaining an existing one.  
RetailCo Analytics faces increasing customer attrition, leading to lost revenue and reduced lifetime value.

Currently, the business lacks a data-driven mechanism to proactively identify customers who are at risk of churn. Marketing campaigns are broad and untargeted, resulting in inefficient spend and low ROI.

## 2. Problem Definition
The objective is to **predict customer churn**.

**Churn Definition:**  
A customer is considered *churned* if they have **not made any purchase in the last 90 days** following an observation cutoff date.

This allows the business to identify customers likely to stop purchasing in the near future.

## 3. Stakeholders
- **Marketing Team:** Target high-risk customers with retention campaigns  
- **Sales Team:** Focus on customers with declining engagement  
- **Product Team:** Understand purchasing behavior and drop-off patterns  
- **Executive Team:** Measure revenue impact and ROI of churn reduction strategies

## 4. Business Impact
A successful churn prediction system enables:
- **15–20% reduction in churn rate**
- Improved customer lifetime value
- Reduced marketing costs through targeted campaigns
- Better allocation of retention budgets

## 5. Success Metrics
**Primary Metric**
- ROC-AUC ≥ **0.78**

**Secondary Metrics**
- Precision ≥ **0.75**
- Recall ≥ **0.70**
- F1-score ≥ **0.72**

These metrics ensure a balance between identifying true churners and minimizing unnecessary retention costs.
