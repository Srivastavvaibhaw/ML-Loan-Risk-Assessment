"""
streamlit_app.py
------------------
Streamlit frontend for the Loan Risk Assessment System.

Provides:
    - An interactive loan application form that predicts approval likelihood
      using the trained Logistic Regression, Decision Tree, and Random Forest
      models (user selectable).
    - A model performance dashboard (comparison table, ROC curve, confusion
      matrices, feature importance) built from the pipeline's saved artifacts.

Run locally with:
    streamlit run streamlit_app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Risk Assessment System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
IMAGES_DIR = PROJECT_ROOT / "Images"
REPORTS_DIR = PROJECT_ROOT / "reports"

# The exact feature column order the models were trained on
FEATURE_COLUMNS = [
    "Dependents",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "total_income",
    "loan_income_ratio",
    "Gender_Male",
    "Married_Yes",
    "Education_Not Graduate",
    "Self_Employed_Yes",
    "Property_Area_Semiurban",
    "Property_Area_Urban",
]


# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load all trained models and the fitted scaler from disk.

    Returns:
        A dictionary containing the three trained models and the scaler.
        Missing artifacts are reported so the app can degrade gracefully.
    """
    artifacts = {}
    filenames = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "Random Forest": "random_forest.pkl",
        "scaler": "scaler.pkl",
    }
    missing = []
    for key, filename in filenames.items():
        path = MODELS_DIR / filename
        if path.exists():
            artifacts[key] = joblib.load(path)
        else:
            missing.append(filename)
    return artifacts, missing


@st.cache_data
def load_comparison_table():
    """Load the model comparison CSV if available."""
    path = OUTPUTS_DIR / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_deployment_report():
    """Load the deployment recommendation markdown report if available."""
    path = REPORTS_DIR / "deployment_recommendation.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def build_feature_row(inputs: dict) -> pd.DataFrame:
    """Transform raw form inputs into the exact engineered feature row
    the models expect (same order/columns as training).

    Args:
        inputs: Dictionary of raw form field values.

    Returns:
        A single-row DataFrame matching FEATURE_COLUMNS.
    """
    dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
    dependents = dependents_map[inputs["dependents"]]

    applicant_income = inputs["applicant_income"]
    coapplicant_income = inputs["coapplicant_income"]
    loan_amount = inputs["loan_amount"]
    loan_term = inputs["loan_term"]
    credit_history = inputs["credit_history"]

    total_income = applicant_income + coapplicant_income
    loan_income_ratio = loan_amount / (total_income + 1e-6)

    row = {
        "Dependents": dependents,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "total_income": total_income,
        "loan_income_ratio": loan_income_ratio,
        "Gender_Male": 1 if inputs["gender"] == "Male" else 0,
        "Married_Yes": 1 if inputs["married"] == "Yes" else 0,
        "Education_Not Graduate": 1 if inputs["education"] == "Not Graduate" else 0,
        "Self_Employed_Yes": 1 if inputs["self_employed"] == "Yes" else 0,
        "Property_Area_Semiurban": 1 if inputs["property_area"] == "Semiurban" else 0,
        "Property_Area_Urban": 1 if inputs["property_area"] == "Urban" else 0,
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
st.sidebar.title("🏦 Loan Risk Assessment")
page = st.sidebar.radio("Navigate", ["🔮 Predict Loan Approval", "📊 Model Dashboard", "📄 Deployment Report"])

artifacts, missing_files = load_models()

if missing_files:
    st.sidebar.error(
        "⚠️ Missing model files: " + ", ".join(missing_files) + "\n\nRun `python main.py` first to train and save models."
    )

# ========================================================================
# PAGE 1: Prediction Form
# ========================================================================
if page == "🔮 Predict Loan Approval":
    st.title("🏦 Loan Risk Assessment System")
    st.markdown("Fill in applicant details below to get an instant loan approval prediction.")

    model_choice = st.sidebar.selectbox(
        "Select Model",
        ["Random Forest", "Logistic Regression", "Decision Tree"],
        help="Choose which trained model to use for the prediction.",
    )

    with st.form("loan_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("👤 Applicant Info")
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])

        with col2:
            st.subheader("💰 Financial Info")
            applicant_income = st.number_input("Applicant Monthly Income (₹)", min_value=0, value=5000, step=500)
            coapplicant_income = st.number_input("Coapplicant Monthly Income (₹)", min_value=0, value=0, step=500)
            loan_amount = st.number_input("Loan Amount (in thousands ₹)", min_value=1, value=120, step=5)
            loan_term = st.selectbox(
                "Loan Term (months)", [360, 180, 120, 240, 60, 300, 84, 36, 12], index=0
            )

        with col3:
            st.subheader("🏠 Credit & Property")
            credit_history = st.selectbox(
                "Credit History",
                options=[1.0, 0.0, -1.0],
                format_func=lambda x: {1.0: "Good (meets guidelines)", 0.0: "Bad (does not meet guidelines)", -1.0: "Unknown / No history"}[x],
            )
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        submitted = st.form_submit_button("🔍 Predict Loan Approval", use_container_width=True)

    if submitted:
        if model_choice not in artifacts or "scaler" not in artifacts:
            st.error("Required model artifacts are missing. Please run `python main.py` first.")
        else:
            inputs = {
                "gender": gender,
                "married": married,
                "dependents": dependents,
                "education": education,
                "self_employed": self_employed,
                "applicant_income": applicant_income,
                "coapplicant_income": coapplicant_income,
                "loan_amount": loan_amount,
                "loan_term": loan_term,
                "credit_history": credit_history,
                "property_area": property_area,
            }
            feature_row = build_feature_row(inputs)

            model = artifacts[model_choice]
            if model_choice == "Logistic Regression":
                scaler = artifacts["scaler"]
                X_input = scaler.transform(feature_row)
            else:
                X_input = feature_row

            prediction = model.predict(X_input)[0]
            probability = model.predict_proba(X_input)[0][1]

            st.markdown("---")
            result_col1, result_col2 = st.columns([1, 2])

            with result_col1:
                if prediction == 1:
                    st.success("### ✅ Loan Approved")
                else:
                    st.error("### ❌ Loan Rejected")
                st.metric("Approval Probability", f"{probability:.1%}")
                st.caption(f"Model used: **{model_choice}**")

            with result_col2:
                st.progress(min(max(probability, 0.0), 1.0))
                if probability >= 0.7:
                    st.info("High confidence in approval — strong applicant profile.")
                elif probability >= 0.5:
                    st.warning("Borderline case — recommend manual underwriter review.")
                else:
                    st.warning("Low approval likelihood — key risk factors present (e.g. credit history, high loan-to-income ratio).")

            with st.expander("View computed features sent to the model"):
                st.dataframe(feature_row.T.rename(columns={0: "Value"}))

# ========================================================================
# PAGE 2: Model Dashboard
# ========================================================================
elif page == "📊 Model Dashboard":
    st.title("📊 Model Performance Dashboard")

    comparison_df = load_comparison_table()

    if comparison_df is not None:
        st.subheader("Model Comparison Table")
        st.dataframe(
            comparison_df.style.format(
                {"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1 Score": "{:.4f}", "ROC AUC": "{:.4f}"}
            ).highlight_max(subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"], color="lightgreen"),
            use_container_width=True,
        )
        best_model = comparison_df.sort_values("ROC AUC", ascending=False).iloc[0]
        st.success(f"🏆 Best model by ROC-AUC: **{best_model['Model']}** ({best_model['ROC AUC']:.4f})")
    else:
        st.warning("Run `python main.py` to generate `outputs/model_comparison.csv`.")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Model Comparison", "📉 ROC Curve", "🔲 Confusion Matrices", "🌟 Feature Importance"])

    def show_image(filename: str, caption: str):
        path = IMAGES_DIR / filename
        if path.exists():
            st.image(str(path), caption=caption, use_container_width=True)
        else:
            st.info(f"Image not found: {filename}. Run `python main.py` first.")

    with tab1:
        show_image("model_comparison.png", "Model Comparison Across Evaluation Metrics")

    with tab2:
        show_image("roc_curve.png", "ROC Curve Comparison — All Models")

    with tab3:
        c1, c2, c3 = st.columns(3)
        with c1:
            show_image("logistic_confusion_matrix.png", "Logistic Regression")
        with c2:
            show_image("decision_tree_confusion_matrix.png", "Decision Tree")
        with c3:
            show_image("random_forest_confusion_matrix.png", "Random Forest")

    with tab4:
        show_image("feature_importance.png", "Top Feature Importances — Random Forest")

    st.markdown("---")
    st.subheader("📊 Exploratory Data Analysis")
    eda_tab1, eda_tab2 = st.tabs(["Approval Rate Breakdowns", "Distributions & Correlation"])

    with eda_tab1:
        e1, e2, e3 = st.columns(3)
        with e1:
            show_image("approval_credit_history.png", "By Credit History")
        with e2:
            show_image("approval_property_area.png", "By Property Area")
        with e3:
            show_image("approval_education.png", "By Education")

    with eda_tab2:
        d1, d2 = st.columns(2)
        with d1:
            show_image("applicant_income_distribution.png", "Applicant Income")
            show_image("loan_amount_distribution.png", "Loan Amount")
            show_image("total_income_distribution.png", "Total Income")
        with d2:
            show_image("coapplicant_income_distribution.png", "Coapplicant Income")
            show_image("loan_income_ratio_distribution.png", "Loan-to-Income Ratio")
            show_image("correlation_heatmap.png", "Correlation Heatmap")

# ========================================================================
# PAGE 3: Deployment Report
# ========================================================================
elif page == "📄 Deployment Report":
    st.title("📄 Deployment Recommendation Report")
    report = load_deployment_report()
    if report:
        st.markdown(report)
    else:
        st.warning("Run `python main.py` to generate `reports/deployment_recommendation.md`.")

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Loan Risk Assessment System · Built with Streamlit")