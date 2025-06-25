# 📊 Detect & Mitigate Bias in Machine Learning Models

## 🔍 Overview

This project investigates bias detection and mitigation in a binary classification context using real-world loan eligibility data. The notebook demonstrates how even performant machine learning models can make systematically unfair decisions—and how to responsibly address them.

---

## 🎯 Problem Statement

We automate a **loan approval process** using historical customer data, including gender, income, credit history, and education, to predict loan eligibility. The challenge: ensuring the model is both **accurate** and **fair**, especially with respect to sensitive attributes like **gender**.

---

## ⚙️ Workflow

1. **📂 Data Preprocessing**
   - Handled missing values and categorical encoding
   - Normalized inputs using `MinMaxScaler` and `StandardScaler`

2. **📈 Exploratory Analysis**
   - Uncovered potential gender-based imbalances in loan approval rates
   - Visualized correlation and distribution plots to support fairness risk assumptions

3. **✅ Baseline Model: Logistic Regression**
   - Trained a binary classifier using `sklearn`
   - Key performance metrics on test set:
     - **Accuracy**: ~81.5%
     - **Precision**: ~87.5%
     - **Recall**: ~89.1%
     - **F1-score**: ~88.3%

4. **📉 Fairness Evaluation**
   - Computed **Statistical Parity Difference (SPD)** and **Disparate Impact (DI)** between male and female applicants:
     - **Disparate Impact**: `~0.68` (ideal = 1.0)
     - **Statistical Parity Difference**: `~ -0.15` (ideal = 0)
   - Insight: Model disproportionately favors male applicants in loan approvals.

5. **🧪 Stress Testing**
   - Swapped gender values and re-evaluated loan decisions for same profiles
   - Found significant variation in predictions—highlighting embedded bias

6. **🛠️ Bias Mitigation**
   - Applied:
     - **Disparate Impact Remover** (AIF360) to preprocess features
     - **Reweighing** (AIF360) to balance training instances
   - Post-mitigation improvements:
     - **Disparate Impact** increased to ~0.9
     - **SPD** improved closer to zero
     - **Only minor drop (~2%) in model accuracy**

7. **📊 Post-Mitigation Insights**
   - Tradeoff: Slight accuracy loss, but significant **fairness gains**
   - Achieved a better balance between model utility and ethical AI practices

---

## 🧠 Skills Demonstrated

- Applied ML: feature engineering, model tuning, metric evaluation
- Responsible AI: fairness definitions, bias detection, mitigation techniques
- Tooling: `scikit-learn`, `AIF360`, `Fairlearn`, `Seaborn`, `Pandas`

---
