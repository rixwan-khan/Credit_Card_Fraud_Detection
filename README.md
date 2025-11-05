# 💳 Credit Card Fraud Detection

## 📌 Project Overview
This project focuses on detecting fraudulent credit card transactions using machine learning. The dataset is highly imbalanced, with very few fraud cases compared to normal ones. The goal was to build models that can effectively identify fraudulent transactions while minimizing false alarms.

The project was implemented in **two phases**:
- **Phase 01:** Used SMOTE oversampling and probability threshold tuning to improve fraud detection.  
- **Phase 02:** Trained models using class-weight balancing to simulate real-world imbalance conditions.

---

## 🧹 Data Preprocessing

### 1. Basic Exploration
- Checked dataset structure using `.info()` and `.describe()`.
- Confirmed there were **no missing values**.
- **Removed duplicate rows** to ensure data integrity.

### 2. Class Distribution
| Class | Percentage |
|--------|-------------|
| 0 (Non-Fraud) | 99.833% |
| 1 (Fraud) | 0.167% |

The dataset is **highly imbalanced**, meaning most transactions are non-fraudulent. Such imbalance can make it difficult for models to detect fraud effectively.

### 3. Outlier Detection
- **Class 0 (Non-Fraud):** Most transactions are small, with a few reaching up to 2000 — considered normal.  
- **Class 1 (Fraud):** Contains several extremely high-value transactions, indicating that amount can be a useful fraud indicator.

### 4. IQR Analysis
- **Class 0 IQR:** 71.79 → Transactions are closely grouped.
- **Class 1 IQR:** 104.89 → Wider spread of transaction amounts, suggesting variability in fraud cases.

### 5. Correlation & Feature Insights
- **Amount vs. Class:** Very weak correlation (close to 0).  
- **Time vs. Class:** Very weak correlation, meaning fraud can occur at any time.  
- **Top correlated features (V17, V14, V12, V10, V16):** Fraud transactions often show **low amounts** with **large negative values**, forming distinct clusters in scatter plots.

---

## 📊 Key Insights from Visualization
- The dataset is **highly imbalanced**, making fraud detection challenging.  
- Fraudulent transactions are **rare and hidden** among normal ones.  
- **High transaction amounts alone** do not indicate fraud, but extreme values can be suspicious.  
- **Amount and Class alone** are not enough to detect fraud — other engineered features are essential.

---

## ⚙️ Phase 01: SMOTE Oversampling & Threshold Tuning

1. Balanced the dataset using **SMOTE** to create additional fraud samples.
2. Trained multiple models:
   - Random Forest  
   - AdaBoost  
   - CatBoost  
   - XGBoost  
   - LightGBM
3. Applied **threshold tuning** using:
   - **ROC Curve (Youden’s J statistic)** to balance sensitivity and specificity.  
   - **Precision-Recall Curve** to maximize the F1-score.

### Best Model Results (PR-based Thresholds)
| Model | Precision | Recall | F1 | ROC AUC |
|--------|------------|--------|----|----------|
| **XGBoost** | **0.962** | **0.789** | **0.867** | **0.966** |
| CatBoost | 0.937 | 0.779 | 0.851 | 0.959 |

✅ **XGBoost (Threshold = 0.958)** achieved the best performance, balancing precision and recall effectively.

---

## 🧭 Risk Tier Classification
To make the fraud detection model more practical, transactions were categorized into risk tiers based on predicted fraud probability:

| Tier | Fraud Probability | Action |
|------|-------------------|--------|
| **High Risk** | ≥ 0.7 | Reject / Flag |
| **Medium Risk** | 0.4 ≤ p < 0.7 | Manual Review |
| **Low Risk** | < 0.4 | Approve |

### Example (XGBoost Results)
| Risk Tier | True Label 0 | True Label 1 | Fraud Rate (%) |
|------------|---------------|---------------|----------------|
| Low Risk | 56636 | 19 | 0.034% |
| Medium Risk | 9 | 0 | 0.000% |
| High Risk | 6 | 76 | 92.68% |

This tiered approach helps prioritize manual reviews and reduce false alarms.

---

## ⚙️ Phase 02: Class-Weight Balancing

Models were retrained using **class weights** instead of oversampling.

| Model | Precision | Recall | F1 | ROC AUC |
|--------|------------|--------|----|----------|
| RandomForest | 0.959 | 0.737 | 0.833 | 0.945 |
| LightGBM | 0.971 | 0.695 | 0.810 | 0.965 |
| CatBoost | 1.000 | 0.253 | 0.403 | 0.932 |
| XGBoost | 0.000 | 0.000 | 0.000 | 0.921 |
| AdaBoost | 0.002 | 1.000 | 0.003 | 0.974 |

🚫 **Observation:** Class-weight balancing alone performed worse.  
XGBoost failed to detect any fraud, while RandomForest and LightGBM performed moderately well.

---

## 🏁 Final Comparison

| Phase | Technique | Best Model | F1 | ROC AUC | Remarks |
|--------|------------|------------|----|----------|----------|
| **Phase 01** | SMOTE + Threshold Tuning | **XGBoost** | **0.867** | **0.966** | Best overall performance |
| **Phase 02** | Class Weight Balancing | RandomForest | 0.833 | 0.945 | Moderate but lower recall |

✅ **Best Performing Model:**  
**XGBoost (Phase 01, PR-based threshold 0.958)**

---

## 💡 Key Takeaways
1. **SMOTE + Threshold Tuning** significantly improves fraud detection.  
2. **XGBoost** provides the best balance between recall and precision.  
3. **Risk-based tiers** make fraud detection actionable and business-friendly.  
4. **Transaction amount alone** isn’t reliable — engineered features hold stronger fraud signals.

---

## 🚀 Future Improvements
- Combine models (e.g., **XGBoost + CatBoost**) using ensemble stacking.  
- Add **time-based or behavioral features** for stronger fraud insights.  
- Use **SHAP or LIME** for explainable AI.  
- Test the model on **real-time streaming data**.  
- Regularly **retrain the model** to adapt to new fraud patterns.

---

## 🧠 Tech Stack
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn, XGBoost, CatBoost, LightGBM  
- **Environment:** Jupyter Notebook

---

## 👨‍💻 Author
**Rizwan**  
Data Scientist | Machine Learning Engineer
