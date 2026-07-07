# CodeAlpha_CreditScoringModel

## 📌 Task
**CodeAlpha Machine Learning Internship — Task 1: Credit Scoring Model**

Predict an individual's creditworthiness (Good Credit vs Bad Credit) using
classification algorithms, based on financial history features such as
income, debt, payment history, and credit utilization.

## 📂 Project Structure
```
CodeAlpha_CreditScoringModel/
│
├── generate_data.py        # Creates the synthetic credit dataset (credit_data.csv)
├── train_model.py          # Preprocessing, feature engineering, model training & evaluation
├── credit_data.csv         # Generated dataset (or replace with a real dataset)
├── outputs/
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   └── feature_importance.png
└── README.md
```

## 🧠 Approach
1. **Data**: 13 financial features per applicant (income, debt-to-income
   ratio, credit utilization, late payments, credit history length, etc.)
   Target: `creditworthy` (1 = Good Credit, 0 = Bad Credit).
2. **Feature Engineering**: added derived ratios — debt-to-savings,
   loan-to-income, income-per-credit-line, late-payment-rate.
3. **Models trained**:
   - Logistic Regression (with feature scaling)
   - Decision Tree Classifier
   - Random Forest Classifier
4. **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC,
   Confusion Matrix.

## 📊 Results (example run)

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.764    | 0.790     | 0.868  | 0.827    | 0.831   |
| Random Forest       | 0.750    | 0.763     | 0.892  | 0.823    | 0.805   |
| Decision Tree       | 0.703    | 0.757     | 0.800  | 0.778    | 0.749   |

Logistic Regression achieved the best ROC-AUC in this run, while Random
Forest gave the highest recall for the "Good Credit" class.

## ▶️ How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn

python generate_data.py     # generates credit_data.csv
python train_model.py       # trains models, prints metrics, saves plots to outputs/
```

## 📁 Using a Real Dataset
This project ships with a synthetic dataset (generated with realistic
statistical relationships) since no internet access was available while
building it. To use a real dataset instead:

1. Download **"Give Me Some Credit"** from Kaggle
   (https://www.kaggle.com/c/GiveMeSomeCredit/data) or the **UCI German
   Credit Data** (https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).
2. Save it as `credit_data.csv` in this folder, with the target column
   renamed to `creditworthy` (1 = good credit, 0 = bad/default risk).
3. Run `train_model.py` — no other code changes needed.

## 🛠️ Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## 🎓 Internship
This project was completed as part of the **CodeAlpha Machine Learning
Internship**.
