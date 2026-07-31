"""
generate_dataset.py
--------------------
Generates a realistic synthetic loan prediction dataset that mirrors the
structure and statistical properties of the well-known Loan Prediction
dataset (Loan_ID, Gender, Married, Dependents, Education, Self_Employed,
ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
Credit_History, Property_Area, Loan_Status).

This script is a one-time data-generation utility and is NOT part of the
production pipeline (src/ modules). It is used only to populate
Dataset/loan_prediction.csv for demonstration purposes.
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_SAMPLES = 614  # matches the size of the classic loan prediction dataset

rng = np.random.default_rng(RANDOM_STATE)


def generate_loan_dataset(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    """Generate a synthetic loan prediction dataset.

    Args:
        n_samples: Number of loan applications to generate.

    Returns:
        A pandas DataFrame with the loan prediction schema.
    """
    loan_id = [f"LP{str(i).zfill(6)}" for i in range(1, n_samples + 1)]

    gender = rng.choice(["Male", "Female"], size=n_samples, p=[0.81, 0.19])
    married = rng.choice(["Yes", "No"], size=n_samples, p=[0.65, 0.35])
    dependents = rng.choice(["0", "1", "2", "3+"], size=n_samples, p=[0.58, 0.17, 0.17, 0.08])
    education = rng.choice(["Graduate", "Not Graduate"], size=n_samples, p=[0.78, 0.22])
    self_employed = rng.choice(["No", "Yes"], size=n_samples, p=[0.86, 0.14])
    property_area = rng.choice(["Urban", "Semiurban", "Rural"], size=n_samples, p=[0.38, 0.38, 0.24])

    # Income distributions (right-skewed, like real income data)
    applicant_income = rng.lognormal(mean=8.55, sigma=0.55, size=n_samples).round(0)
    applicant_income = np.clip(applicant_income, 150, 81000)

    coapplicant_income = np.where(
        married == "Yes",
        rng.lognormal(mean=7.6, sigma=0.9, size=n_samples),
        0,
    )
    coapplicant_income = np.where(rng.random(n_samples) < 0.15, 0, coapplicant_income)
    coapplicant_income = np.clip(coapplicant_income, 0, 42000).round(0)

    total_income_raw = applicant_income + coapplicant_income
    loan_amount = (total_income_raw / rng.uniform(4.5, 9.0, size=n_samples)).round(0)
    loan_amount = np.clip(loan_amount, 9, 700)

    loan_amount_term = rng.choice(
        [360, 180, 120, 240, 60, 300, 84, 36, 12],
        size=n_samples,
        p=[0.73, 0.08, 0.03, 0.05, 0.02, 0.03, 0.02, 0.02, 0.02],
    )

    credit_history = rng.choice([1.0, 0.0], size=n_samples, p=[0.84, 0.16])

    # Introduce realistic missing values
    def inject_missing(arr, frac):
        arr = arr.copy().astype(object)
        n_missing = int(len(arr) * frac)
        idx = rng.choice(len(arr), size=n_missing, replace=False)
        arr[idx] = np.nan
        return arr

    gender = inject_missing(gender, 0.02)
    married = inject_missing(married, 0.005)
    dependents = inject_missing(dependents, 0.025)
    self_employed = inject_missing(self_employed, 0.05)
    loan_amount = inject_missing(loan_amount, 0.035)
    loan_amount_term = inject_missing(loan_amount_term, 0.023)
    credit_history = inject_missing(credit_history, 0.08)

    # Loan_Status generated from a latent probability model that depends on
    # credit history, income ratio and education, then adds noise -- this
    # ensures the models trained downstream have genuine, learnable signal.
    credit_history_filled = pd.Series(credit_history).fillna(1.0).astype(float).values
    loan_amount_filled = pd.Series(loan_amount).fillna(np.nanmedian(loan_amount.astype(float))).astype(float).values

    total_income = applicant_income + coapplicant_income
    loan_income_ratio = loan_amount_filled / np.maximum(total_income, 1)

    logit = (
        2.6 * credit_history_filled
        - 1.6 * loan_income_ratio
        + 0.35 * (education == "Graduate").astype(float)
        + 0.15 * (property_area == "Semiurban").astype(float)
        - 0.10 * (property_area == "Rural").astype(float)
        + rng.normal(0, 0.6, size=n_samples)
        - 1.1
    )
    prob_approved = 1 / (1 + np.exp(-logit))
    loan_status = np.where(rng.random(n_samples) < prob_approved, "Y", "N")

    df = pd.DataFrame(
        {
            "Loan_ID": loan_id,
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income.astype(int),
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_amount_term,
            "Credit_History": credit_history,
            "Property_Area": property_area,
            "Loan_Status": loan_status,
        }
    )
    return df


if __name__ == "__main__":
    dataset = generate_loan_dataset()
    output_path = "Dataset/loan_prediction.csv"
    dataset.to_csv(output_path, index=False)
    print(f"Synthetic loan prediction dataset generated: {output_path}")
    print(f"Shape: {dataset.shape}")
    print(dataset["Loan_Status"].value_counts(normalize=True))
