# 🚗 Insurance Customer Behavior Predictive Modeling  
*Multi-task machine learning and data protection project for Sure Tomorrow Insurance.*

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)  
![pandas](https://img.shields.io/badge/pandas-EDA-green?logo=pandas)  
![numpy](https://img.shields.io/badge/numpy-Numerical-blue?logo=numpy)  
![matplotlib](https://img.shields.io/badge/matplotlib-Visualization-orange?logo=matplotlib)  
![seaborn](https://img.shields.io/badge/seaborn-EDA-blue?logo=python)  
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)  
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)  
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Overview
**Sure Tomorrow** wants to leverage ML to better understand customers and protect sensitive data. This notebook delivers a four-part workflow:

1) **Find similar customers** — retrieve lookalikes for marketing via distance-based search.  
2) **Predict insurance benefit eligibility** — binary classification (will a customer receive a benefit).  
3) **Predict the number of insurance benefits** — regression on expected benefit counts.  
4) **Protect clients’ personal data** — apply a privacy-preserving transformation that keeps model quality intact.

This project demonstrates **EDA, similarity search, classification, regression, and practical data masking**.

---

## 📊 Data
- **Source:** TripleTen (synthetic insurance dataset)
- **Features (examples):** `age`, `gender`, `income`, `family_size` (naming consistent with notebook usage)
- **Targets:**
  - **Task 2:** Derived **binary** target for benefit eligibility (from benefit counts)
  - **Task 3:** **Count** of insurance benefits

---

## ⚙️ Methods & Tools
**Libraries:** pandas, NumPy, matplotlib, seaborn, scikit-learn

**Task 1 — Similarity Search**
- Standardization where appropriate; Euclidean & Manhattan distances
- **k-Nearest Neighbors–style lookup** for similar profiles

**Task 2 — Benefit Eligibility (Classification)**
- Model: **KNeighborsClassifier**
- Variants: **unscaled vs. scaled** features (scaling improved results)
- **Metrics:** **F1 score** (primary) and **confusion matrices** to analyze errors
- No accuracy or ROC-AUC were used for this task

**Task 3 — Benefit Count (Regression)**
- Model: **Linear Regression** (custom implementation class)
- Metrics: **RMSE** and **R²**

**Task 4 — Data Protection**
- Linear **obfuscation/masking transform** applied to features
- Verified that **model quality for Task 3 is preserved** (pre/post RMSE and R² remain comparable)

---

## 📈 Results (High Level)
- **Task 1:** Similar-customer lookup worked as intended, grouping profiles by distance in feature space.
- **Task 2:** **Scaled KNN** delivered higher **F1** and cleaner **confusion matrices** than unscaled KNN (clear reduction in misclassifications).
- **Task 3:** Linear Regression produced reasonable **RMSE/R²**, capturing key drivers of benefit counts.
- **Task 4:** The masking transform **protected sensitive data** without degrading Task 3’s model quality.

---

## 💡 Business Takeaways
- Lookalike retrieval supports **targeted outreach** and personalization.
- Using **F1** + confusion matrices focuses optimization on the real business tradeoff between missed positives and false alarms.
- Benefit-count predictions help with **capacity planning** and **risk/pricing**.
- **Privacy-preserving transforms** can be operationalized without sacrificing predictive performance.

---

## 🗂 Repo Structure
```
insurance-customer-behavior/
├── notebooks/ <- Final cleaned Jupyter notebook
├── data/ <- Source data
├── requirements.txt <- Dependencies
├── LICENSE <- MIT License
└── README.md <- This file
```

---

## 🚀 How to Run
1. Clone this repo:
    ```bash
    git clone https://github.com/Flamingo-Rocker/insurance-customer-behavior.git
    cd insurance-customer-behavior
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
3. Open the notebook in `/notebooks` to reproduce this analysis.

---

## 📦 Requirements
```
pandas==2.3.2
numpy==2.2.5
numpy-base==2.2.5            
numpydoc==1.9.0            
matplotlib==3.10.5           
matplotlib-base==3.10.5           
matplotlib-inline==0.1.6
seaborn==0.13.2
scikit-learn==1.7.1 
ipython==8.30.0
mpmath==1.3.0
sphinxcontrib-jsmath==1.0.1
```

---

## 🙏 Acknowledgment
Developed as part of the TripleTen Data Science Bootcamp, combining similarity search, classification, regression, and privacy-preserving ML in a single, production-minded workflow.
