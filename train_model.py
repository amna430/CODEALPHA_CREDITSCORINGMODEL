"""
train_model.py
----------------
CodeAlpha ML Internship - Task 1: Credit Scoring Model

Predicts an individual's creditworthiness using classification algorithms:
Logistic Regression, Decision Tree, and Random Forest.

Evaluates each model using Precision, Recall, F1-Score, and ROC-AUC.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

from generate_data import generate_credit_dataset

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# 1. LOAD / GENERATE DATA
# ---------------------------------------------------------------------
def load_data():
    csv_path = "credit_data.csv"
    if os.path.exists(csv_path):
        print(f"Loading dataset from {csv_path} ...")
        df = pd.read_csv(csv_path)
    else:
        print("No credit_data.csv found — generating synthetic dataset ...")
        df = generate_credit_dataset(5000)
        df.to_csv(csv_path, index=False)
    return df


# ---------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()

    # Ratio-based engineered features often used in real credit scoring
    df["debt_to_savings"] = df["total_debt"] / (df["savings_balance"] + 1)
    df["loan_to_income"] = df["loan_amount"] / (df["annual_income"] + 1)
    df["income_per_credit_line"] = df["annual_income"] / (df["num_credit_lines"] + 1)
    df["late_payment_rate"] = df["late_payments_2yr"] / (df["credit_history_length"] + 1)

    return df


# ---------------------------------------------------------------------
# 3. TRAIN / EVALUATE MODELS
# ---------------------------------------------------------------------
def evaluate_model(name, model, X_test, y_test, results):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\n===== {name} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Bad Credit", "Good Credit"]))

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1-Score": f1, "ROC-AUC": roc_auc
    })

    cm = confusion_matrix(y_test, y_pred)
    return y_proba, cm


def main():
    df = load_data()
    df = engineer_features(df)

    X = df.drop(columns=["creditworthy"])
    y = df["creditworthy"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (important for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    }

    results = []
    roc_data = {}
    cms = {}

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_proba, cm = evaluate_model(name, model, X_test_scaled, y_test, results)
        else:
            model.fit(X_train, y_train)
            y_proba, cm = evaluate_model(name, model, X_test, y_test, results)
        roc_data[name] = y_proba
        cms[name] = cm

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    print("\n===== MODEL COMPARISON (sorted by ROC-AUC) =====")
    print(results_df.to_string(index=False))

    # -----------------------------------------------------------------
    # PLOTS
    # -----------------------------------------------------------------
    sns.set_style("whitegrid")

    # ROC Curves
    plt.figure(figsize=(7, 6))
    for name, y_proba in roc_data.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Credit Scoring Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=150)
    plt.close()

    # Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"), dpi=150)
    plt.close()

    # Model comparison bar chart
    plt.figure(figsize=(8, 5))
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    results_df.set_index("Model")[metrics].plot(kind="bar", figsize=(9, 5))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
    plt.close()

    # Feature importance (Random Forest)
    rf_model = models["Random Forest"]
    importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances.values, y=importances.index, color="steelblue")
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close()

    print(f"\nAll plots and results saved in the '{OUTPUT_DIR}/' folder.")
    return results_df


if __name__ == "__main__":
    main()
