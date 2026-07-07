"""
generate_data.py
-----------------
Generates a realistic synthetic credit scoring dataset.

In a real internship submission, you would normally use a public dataset
such as the "German Credit Data" or "Give Me Some Credit" (Kaggle) dataset.
Since this environment has no internet access, this script builds a
synthetic dataset with the SAME structure and realistic relationships,
so you can later swap in a real CSV (see the note at the bottom) without
changing any downstream code.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

def generate_credit_dataset(n_samples=5000):
    # --- Base demographic / financial features ---
    age = np.random.randint(18, 70, n_samples)
    annual_income = np.random.lognormal(mean=10.8, sigma=0.5, size=n_samples).round(2)  # skewed like real income
    employment_years = np.clip(np.random.exponential(scale=5, size=n_samples), 0, 40).round(1)

    # Existing debt as a fraction of income (debt-to-income ratio)
    debt_to_income = np.clip(np.random.normal(0.35, 0.15, n_samples), 0, 1.2).round(3)
    total_debt = (debt_to_income * annual_income).round(2)

    # Number of open credit lines / accounts
    num_credit_lines = np.random.poisson(4, n_samples)

    # Number of late payments in the last 2 years
    late_payments_2yr = np.random.poisson(1.2, n_samples)

    # Credit utilization ratio (how much of available credit is used)
    credit_utilization = np.clip(np.random.beta(2, 5, n_samples), 0, 1).round(3)

    # Length of credit history in years
    credit_history_length = np.clip(np.random.normal(8, 5, n_samples), 0, 40).round(1)

    # Number of hard inquiries in last 6 months
    recent_inquiries = np.random.poisson(1, n_samples)

    # Loan amount requested
    loan_amount = np.random.lognormal(mean=9.2, sigma=0.6, size=n_samples).round(2)

    # Home ownership status (0 = rent, 1 = mortgage, 2 = own)
    home_ownership = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.4, 0.2])

    # Savings balance
    savings_balance = np.clip(np.random.lognormal(mean=8, sigma=1.2, size=n_samples), 0, None).round(2)

    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income,
        "employment_years": employment_years,
        "total_debt": total_debt,
        "debt_to_income": debt_to_income,
        "num_credit_lines": num_credit_lines,
        "late_payments_2yr": late_payments_2yr,
        "credit_utilization": credit_utilization,
        "credit_history_length": credit_history_length,
        "recent_inquiries": recent_inquiries,
        "loan_amount": loan_amount,
        "home_ownership": home_ownership,
        "savings_balance": savings_balance,
    })

    # --- Build a "true" creditworthiness score using a realistic weighted formula ---
    # Higher score = more creditworthy (this simulates the hidden ground truth)
    risk_score = (
        -0.020 * df["debt_to_income"] * 100
        - 0.35 * df["late_payments_2yr"]
        - 1.8 * df["credit_utilization"]
        - 0.15 * df["recent_inquiries"]
        + 0.05 * df["credit_history_length"]
        + 0.00002 * df["annual_income"]
        + 0.00003 * df["savings_balance"]
        + 0.08 * df["employment_years"]
        - 0.00001 * df["loan_amount"]
        + 0.3 * df["home_ownership"]
        + np.random.normal(0, 1.0, n_samples)  # noise
    )

    # Convert risk_score into a binary target: 1 = Good/Creditworthy, 0 = Bad/Default risk
    threshold = np.percentile(risk_score, 35)  # ~35% flagged as bad credit risk
    df["creditworthy"] = (risk_score > threshold).astype(int)

    return df


if __name__ == "__main__":
    df = generate_credit_dataset(5000)
    df.to_csv("credit_data.csv", index=False)
    print(f"Dataset generated: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df["creditworthy"].value_counts(normalize=True))
    print(df.head())

# NOTE: To use a REAL dataset instead (recommended for your GitHub submission):
#   1. Download "Give Me Some Credit" from Kaggle:
#      https://www.kaggle.com/c/GiveMeSomeCredit/data
#      OR the UCI "German Credit Data":
#      https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
#   2. Save it as credit_data.csv with a target column (rename target to
#      'creditworthy', 1 = good credit, 0 = bad credit).
#   3. Everything in train_model.py will work unchanged.
