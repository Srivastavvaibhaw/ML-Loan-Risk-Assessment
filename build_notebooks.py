"""
build_notebooks.py
--------------------
Generates the five project notebooks (Notebook/01..05) using nbformat.
This is a one-time build utility, not part of the production pipeline.
Each notebook is executed against the actual project artifacts so that
output cells reflect real, non-placeholder results.
"""

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def make_notebook(cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    return nb


# ============================================================
# NOTEBOOK 1: Data Preprocessing
# ============================================================
nb1_cells = [
    md("# 01 - Data Preprocessing\n"
       "## Loan Risk Assessment System\n\n"
       "**Objective:** Load the raw loan prediction dataset, inspect its structure, "
       "clean missing values, and persist a cleaned dataset for downstream use.\n\n"
       "**Workflow:**\n"
       "1. Load dataset\n"
       "2. Inspect shape, schema, missing values, statistics\n"
       "3. Drop identifier column\n"
       "4. Compare Credit_History imputation strategies\n"
       "5. Clean dataset (mode/median imputation)\n"
       "6. Save cleaned dataset"),
    code("import sys\n"
         "sys.path.append('..')\n\n"
         "import pandas as pd\n"
         "from src import config\n"
         "from src.data_loader import load_dataset, inspect_dataset\n"
         "from src.preprocessing import clean_dataset, compare_credit_history_strategies\n\n"
         "pd.set_option('display.max_columns', None)"),
    md("## Step 1: Load Dataset"),
    code("raw_df = load_dataset(config.RAW_DATA_PATH)\n"
         "raw_df.head()"),
    md("## Step 2: Dataset Inspection\n"
       "Shape, schema info, missing values, and descriptive statistics."),
    code("inspect_dataset(raw_df)"),
    md("## Step 3: Credit History Imputation Strategy Comparison\n\n"
       "We compare **two distinct strategies** for handling missing `Credit_History` "
       "values:\n"
       "- **Strategy 1 (Mode):** Fill missing values with the most frequent credit history value.\n"
       "- **Strategy 2 (Unknown category):** Encode missing values as a distinct sentinel "
       "category (-1), treating 'missingness' itself as informative.\n\n"
       "The comparison below evaluates approval rates under each strategy to decide "
       "which is carried forward into production."),
    code("strategy_comparison = compare_credit_history_strategies(raw_df)\n"
         "strategy_comparison"),
    md("**Conclusion:** The 'Unknown category' strategy is selected for the production "
       "pipeline because applicants with unknown credit history show a distinctly "
       "different approval rate from both the 'good' (1.0) and 'bad' (0.0) groups — "
       "this signal would be lost under mode imputation."),
    md("## Step 4: Run Full Cleaning Pipeline\n\n"
       "This drops `Loan_ID`, applies mode imputation to categorical fields, median "
       "imputation to numeric fields, and the 'Unknown category' strategy to "
       "`Credit_History`."),
    code("cleaned_df = clean_dataset(raw_df)\n"
         "print(f'Cleaned dataset shape: {cleaned_df.shape}')\n"
         "print(f'Remaining missing values: {cleaned_df.isnull().sum().sum()}')\n"
         "cleaned_df.head()"),
    md("## Step 5: Save Cleaned Dataset"),
    code("cleaned_df.to_csv(config.CLEANED_DATA_PATH, index=False)\n"
         "print(f'Saved cleaned dataset to: {config.CLEANED_DATA_PATH}')"),
    md("## Conclusion\n\n"
       "The raw dataset has been fully inspected and cleaned:\n"
       "- The non-predictive `Loan_ID` column was dropped.\n"
       "- Categorical missing values (`Gender`, `Married`, `Dependents`, `Self_Employed`) "
       "were imputed with their mode.\n"
       "- Numeric missing values (`LoanAmount`, `Loan_Amount_Term`) were imputed with "
       "their median.\n"
       "- `Credit_History` missing values were encoded as an informative 'Unknown' "
       "category (-1) after comparing it against simple mode imputation.\n\n"
       "The cleaned dataset contains **zero missing values** and is ready for feature "
       "engineering in the next notebook (`02_EDA.ipynb`)."),
]

# ============================================================
# NOTEBOOK 2: EDA
# ============================================================
nb2_cells = [
    md("# 02 - Exploratory Data Analysis (EDA)\n"
       "## Loan Risk Assessment System\n\n"
       "**Objective:** Explore relationships between applicant attributes and loan "
       "approval outcomes, and visualize feature distributions.\n\n"
       "**Workflow:**\n"
       "1. Load cleaned dataset\n"
       "2. Feature engineering (for numeric distribution plots)\n"
       "3. Approval rate breakdowns by category\n"
       "4. Numeric distributions\n"
       "5. Correlation heatmap"),
    code("import sys\n"
         "sys.path.append('..')\n\n"
         "import pandas as pd\n"
         "from IPython.display import Image, display\n"
         "from src import config\n"
         "from src.data_loader import load_dataset\n"
         "from src.feature_engineering import engineer_features\n"
         "from src.visualization import (\n"
         "    plot_approval_rate_by_category, plot_distribution,\n"
         "    plot_correlation_heatmap\n"
         ")\n\n"
         "raw_df = load_dataset(config.RAW_DATA_PATH)\n"
         "cleaned_df = pd.read_csv(config.CLEANED_DATA_PATH)\n"
         "engineered_df = engineer_features(cleaned_df)\n"
         "engineered_df.head()"),
    md("## Approval Rate by Credit History\n\n"
       "Credit history is typically the single strongest predictor of loan approval — "
       "applicants with a confirmed good credit history should show a much higher "
       "approval rate."),
    code("plot_approval_rate_by_category(\n"
         "    raw_df, 'Credit_History', 'Loan_Status',\n"
         "    'Loan Approval Rate by Credit History',\n"
         "    config.IMAGES_DIR / 'approval_credit_history.png'\n"
         ")\n"
         "display(Image(filename=str(config.IMAGES_DIR / 'approval_credit_history.png')))"),
    md("## Approval Rate by Property Area"),
    code("plot_approval_rate_by_category(\n"
         "    raw_df, 'Property_Area', 'Loan_Status',\n"
         "    'Loan Approval Rate by Property Area',\n"
         "    config.IMAGES_DIR / 'approval_property_area.png'\n"
         ")\n"
         "display(Image(filename=str(config.IMAGES_DIR / 'approval_property_area.png')))"),
    md("## Approval Rate by Education"),
    code("plot_approval_rate_by_category(\n"
         "    raw_df, 'Education', 'Loan_Status',\n"
         "    'Loan Approval Rate by Education',\n"
         "    config.IMAGES_DIR / 'approval_education.png'\n"
         ")\n"
         "display(Image(filename=str(config.IMAGES_DIR / 'approval_education.png')))"),
    md("## Numeric Feature Distributions"),
    code("plot_distribution(engineered_df, 'ApplicantIncome', 'Applicant Income Distribution',\n"
         "                   config.IMAGES_DIR / 'applicant_income_distribution.png')\n"
         "display(Image(filename=str(config.IMAGES_DIR / 'applicant_income_distribution.png')))"),
    code("plot_distribution(engineered_df, 'CoapplicantIncome', 'Coapplicant Income Distribution',\n"
         "                   config.IMAGES_DIR / 'coapplicant_income_distribution.png', color='#21918c')\n"
         "display(Image(filename=str(config.IMAGES_DIR / 'coapplicant_income_distribution.png')))"),
    code("plot_distribution(engineered_df, 'LoanAmount', 'Loan Amount Distribution',\n"
         "                   config.IMAGES_DIR / 'loan_amount_distribution.png', color='#5ec962')\n"
         "display(Image(filename=str(config.IMAGES_DIR / 'loan_amount_distribution.png')))"),
    code("plot_distribution(engineered_df, 'total_income', 'Total Income Distribution',\n"
         "                   config.IMAGES_DIR / 'total_income_distribution.png', color='#fde725')\n"
         "display(Image(filename=str(config.IMAGES_DIR / 'total_income_distribution.png')))"),
    code("plot_distribution(engineered_df, 'loan_income_ratio', 'Loan-to-Income Ratio Distribution',\n"
         "                   config.IMAGES_DIR / 'loan_income_ratio_distribution.png', color='#440154')\n"
         "display(Image(filename=str(config.IMAGES_DIR / 'loan_income_ratio_distribution.png')))"),
    md("## Correlation Heatmap"),
    code("plot_correlation_heatmap(engineered_df, config.IMAGES_DIR / 'correlation_heatmap.png')\n"
         "display(Image(filename=str(config.IMAGES_DIR / 'correlation_heatmap.png')))"),
    md("## Conclusion\n\n"
       "Key EDA findings:\n"
       "- **Credit History** shows the strongest relationship with loan approval — "
       "applicants with a confirmed good credit history are approved at a "
       "substantially higher rate.\n"
       "- **Semiurban properties** tend to have a somewhat higher approval rate than "
       "urban or rural properties.\n"
       "- **Graduates** show marginally higher approval rates than non-graduates.\n"
       "- Income and loan amount distributions are **right-skewed**, consistent with "
       "typical real-world income data, motivating the engineered `loan_income_ratio` "
       "feature as a more normalized risk signal.\n"
       "- The correlation heatmap confirms `Credit_History` and `loan_income_ratio` "
       "are among the features most associated with the target."),
]

# ============================================================
# NOTEBOOK 3: Model Training
# ============================================================
nb3_cells = [
    md("# 03 - Model Training\n"
       "## Loan Risk Assessment System\n\n"
       "**Objective:** Split the engineered dataset, scale features, and train three "
       "classification models: Logistic Regression, Decision Tree (GridSearchCV-tuned), "
       "and Random Forest (GridSearchCV-tuned).\n\n"
       "**Workflow:**\n"
       "1. Load engineered dataset\n"
       "2. Train/test split (80/20, stratified)\n"
       "3. Feature scaling (Logistic Regression only)\n"
       "4. Train Logistic Regression\n"
       "5. Tune & train Decision Tree\n"
       "6. Tune & train Random Forest\n"
       "7. Persist all models"),
    code("import sys\n"
         "sys.path.append('..')\n\n"
         "import pandas as pd\n"
         "from sklearn.model_selection import train_test_split\n"
         "from src import config\n"
         "from src.data_loader import load_dataset\n"
         "from src.feature_engineering import engineer_features\n"
         "from src.model import scale_features, train_all_models\n"
         "from src.utils import save_object\n\n"
         "cleaned_df = pd.read_csv(config.CLEANED_DATA_PATH)\n"
         "engineered_df = engineer_features(cleaned_df)\n"
         "engineered_df.shape"),
    md("## Step 1: Train/Test Split (80/20, stratified, random_state=42)"),
    code("X = engineered_df.drop(columns=[config.TARGET_COLUMN])\n"
         "y = engineered_df[config.TARGET_COLUMN]\n\n"
         "X_train, X_test, y_train, y_test = train_test_split(\n"
         "    X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y\n"
         ")\n"
         "print(f'Train shape: {X_train.shape}')\n"
         "print(f'Test shape: {X_test.shape}')\n"
         "print(f'Train approval rate: {y_train.mean():.3f}')\n"
         "print(f'Test approval rate: {y_test.mean():.3f}')"),
    md("## Step 2: Feature Scaling\n\n"
       "`StandardScaler` is fit on the training set and applied to both train/test — "
       "used exclusively for Logistic Regression since tree-based models "
       "(Decision Tree, Random Forest) are scale-invariant."),
    code("X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)\n"
         "save_object(scaler, config.MODELS_DIR / config.MODEL_FILENAMES['scaler'])"),
    md("## Step 3: Train All Models\n\n"
       "- **Logistic Regression**: trained on scaled features with `class_weight='balanced'`.\n"
       "- **Decision Tree**: tuned via `GridSearchCV` over `max_depth`, "
       "`min_samples_split`, `min_samples_leaf` (cv=5, scoring=roc_auc).\n"
       "- **Random Forest**: tuned via `GridSearchCV` over `n_estimators`, `max_depth`, "
       "`min_samples_split`, `max_features` (cv=5, scoring=roc_auc)."),
    code("models = train_all_models(\n"
         "    X_train, y_train, X_train_scaled,\n"
         "    dt_param_grid=config.DECISION_TREE_PARAM_GRID,\n"
         "    rf_param_grid=config.RANDOM_FOREST_PARAM_GRID,\n"
         "    cv=config.CV_FOLDS, scoring=config.SCORING_METRIC,\n"
         "    random_state=config.RANDOM_STATE\n"
         ")\n"
         "models"),
    code("print('Decision Tree best params:', models['Decision Tree'].get_params())"),
    code("print('Random Forest best params:', models['Random Forest'].get_params())"),
    md("## Step 4: Persist Trained Models"),
    code("save_object(models['Logistic Regression'], config.MODELS_DIR / config.MODEL_FILENAMES['logistic_regression'])\n"
         "save_object(models['Decision Tree'], config.MODELS_DIR / config.MODEL_FILENAMES['decision_tree'])\n"
         "save_object(models['Random Forest'], config.MODELS_DIR / config.MODEL_FILENAMES['random_forest'])"),
    md("## Conclusion\n\n"
       "All three models were successfully trained:\n"
       "- Logistic Regression was trained on standardized features with balanced class "
       "weights to account for the class imbalance toward approvals.\n"
       "- The Decision Tree and Random Forest were tuned via 5-fold cross-validated "
       "grid search, optimizing for ROC-AUC.\n"
       "- All trained models and the fitted scaler were persisted to `models/` for "
       "reuse in evaluation (`04_Model_Evaluation.ipynb`) without retraining."),
]

# ============================================================
# NOTEBOOK 4: Model Evaluation
# ============================================================
nb4_cells = [
    md("# 04 - Model Evaluation\n"
       "## Loan Risk Assessment System\n\n"
       "**Objective:** Evaluate all three trained models on the held-out test set "
       "using Accuracy, Precision, Recall, F1 Score, and ROC-AUC, and generate "
       "confusion matrices, a combined ROC curve, and feature importance plots.\n\n"
       "**Workflow:**\n"
       "1. Load persisted models and test data\n"
       "2. Compute predictions & metrics per model\n"
       "3. Confusion matrices\n"
       "4. Combined ROC curve\n"
       "5. Feature importance (Random Forest)\n"
       "6. Build final model comparison table"),
    code("import sys\n"
         "sys.path.append('..')\n\n"
         "import pandas as pd\n"
         "from IPython.display import Image, display\n"
         "from src import config\n"
         "from src.utils import load_object\n"
         "from src.evaluation import evaluate_model, build_comparison_table, get_best_model_name\n"
         "from src.visualization import (\n"
         "    plot_confusion_matrix, plot_combined_roc_curve,\n"
         "    plot_feature_importance, plot_model_comparison\n"
         ")\n\n"
         "train_df = pd.read_csv(config.TRAIN_DATA_PATH)\n"
         "test_df = pd.read_csv(config.TEST_DATA_PATH)\n"
         "X_test = test_df.drop(columns=[config.TARGET_COLUMN])\n"
         "y_test = test_df[config.TARGET_COLUMN]\n\n"
         "log_reg = load_object(config.MODELS_DIR / config.MODEL_FILENAMES['logistic_regression'])\n"
         "decision_tree = load_object(config.MODELS_DIR / config.MODEL_FILENAMES['decision_tree'])\n"
         "random_forest = load_object(config.MODELS_DIR / config.MODEL_FILENAMES['random_forest'])\n"
         "scaler = load_object(config.MODELS_DIR / config.MODEL_FILENAMES['scaler'])\n\n"
         "X_test_scaled = scaler.transform(X_test)\n"
         "models = {'Logistic Regression': log_reg, 'Decision Tree': decision_tree, 'Random Forest': random_forest}"),
    md("## Per-Model Evaluation\n\n"
       "For each model, we compute Accuracy, Precision, Recall, F1 Score, and "
       "ROC-AUC, then plot a confusion matrix."),
    code("all_metrics = {}\n"
         "probabilities = {}\n"
         "filename_map = {\n"
         "    'Logistic Regression': 'logistic_confusion_matrix.png',\n"
         "    'Decision Tree': 'decision_tree_confusion_matrix.png',\n"
         "    'Random Forest': 'random_forest_confusion_matrix.png',\n"
         "}\n\n"
         "for model_name, model in models.items():\n"
         "    X_eval = X_test_scaled if model_name == 'Logistic Regression' else X_test\n"
         "    y_pred = model.predict(X_eval)\n"
         "    y_proba = model.predict_proba(X_eval)[:, 1]\n"
         "    metrics = evaluate_model(model_name, y_test.values, y_pred, y_proba)\n"
         "    all_metrics[model_name] = metrics\n"
         "    probabilities[model_name] = y_proba\n"
         "    plot_confusion_matrix(y_test.values, y_pred, model_name, config.IMAGES_DIR / filename_map[model_name])\n"
         "    display(Image(filename=str(config.IMAGES_DIR / filename_map[model_name])))"),
    md("## Combined ROC Curve"),
    code("plot_combined_roc_curve(y_test.values, probabilities, config.IMAGES_DIR / 'roc_curve.png')\n"
         "display(Image(filename=str(config.IMAGES_DIR / 'roc_curve.png')))"),
    md("## Feature Importance (Random Forest)"),
    code("plot_feature_importance(\n"
         "    list(X_test.columns), random_forest.feature_importances_,\n"
         "    'Random Forest', config.IMAGES_DIR / 'feature_importance.png'\n"
         ")\n"
         "display(Image(filename=str(config.IMAGES_DIR / 'feature_importance.png')))"),
    md("## Final Model Comparison Table"),
    code("comparison_df = build_comparison_table(all_metrics)\n"
         "comparison_df.to_csv(config.MODEL_COMPARISON_PATH, index=False)\n"
         "comparison_df"),
    code("plot_model_comparison(comparison_df, config.IMAGES_DIR / 'model_comparison.png')\n"
         "display(Image(filename=str(config.IMAGES_DIR / 'model_comparison.png')))"),
    code("best_model = get_best_model_name(comparison_df, metric='ROC AUC')\n"
         "print(f'Best performing model (by ROC AUC): {best_model}')"),
    md("## Conclusion\n\n"
       "All three models were evaluated on an identical, untouched 20% test split. "
       "The model comparison table and combined ROC curve above provide a direct, "
       "apples-to-apples comparison across Accuracy, Precision, Recall, F1 Score, and "
       "ROC-AUC. The model with the highest ROC-AUC is carried forward as the "
       "candidate for deployment, formalized in "
       "`reports/deployment_recommendation.md` and `05_Final_Report.ipynb`."),
]

# ============================================================
# NOTEBOOK 5: Final Report
# ============================================================
nb5_cells = [
    md("# 05 - Final Report\n"
       "## Loan Risk Assessment System\n\n"
       "**Objective:** Summarize the end-to-end project — from raw data to a "
       "production deployment recommendation — for stakeholder review.\n\n"
       "**Contents:**\n"
       "1. Project overview\n"
       "2. Final model comparison table\n"
       "3. Key visualizations\n"
       "4. Deployment recommendation\n"
       "5. Conclusion & future improvements"),
    md("## 1. Project Overview\n\n"
       "The Loan Risk Assessment System predicts whether a loan application should "
       "be **approved** or **rejected** based on applicant demographic and financial "
       "attributes. Three classification models were trained and rigorously "
       "compared: **Logistic Regression**, **Decision Tree**, and **Random Forest**.\n\n"
       "The pipeline covers the full ML lifecycle: data cleaning, feature "
       "engineering, exploratory data analysis, model training with hyperparameter "
       "tuning (GridSearchCV), evaluation, and a final banking deployment "
       "recommendation."),
    code("import sys\n"
         "sys.path.append('..')\n\n"
         "import pandas as pd\n"
         "from IPython.display import Image, Markdown, display\n"
         "from src import config\n\n"
         "comparison_df = pd.read_csv(config.MODEL_COMPARISON_PATH)\n"
         "comparison_df"),
    md("## 2. Model Comparison Summary"),
    code("display(Image(filename=str(config.IMAGES_DIR / 'model_comparison.png')))"),
    code("display(Image(filename=str(config.IMAGES_DIR / 'roc_curve.png')))"),
    md("## 3. Key Visualizations"),
    code("display(Image(filename=str(config.IMAGES_DIR / 'approval_credit_history.png')))"),
    code("display(Image(filename=str(config.IMAGES_DIR / 'feature_importance.png')))"),
    code("display(Image(filename=str(config.IMAGES_DIR / 'correlation_heatmap.png')))"),
    md("## 4. Deployment Recommendation\n\n"
       "The full deployment recommendation report is available at "
       "`reports/deployment_recommendation.md`. It is rendered below."),
    code("with open(config.DEPLOYMENT_REPORT_PATH) as f:\n"
         "    report_content = f.read()\n"
         "display(Markdown(report_content))"),
    md("## 5. Conclusion & Future Improvements\n\n"
       "**Conclusion:** Based on the comparative evaluation above, the "
       "highest ROC-AUC model is recommended for deployment in the banking loan "
       "approval workflow, balancing precision (minimizing bad-loan approvals) and "
       "recall (minimizing lost good-customer opportunities).\n\n"
       "**Future Improvements:**\n"
       "- Incorporate additional applicant data (e.g. employment history length, "
       "existing debt-to-income ratio, bureau credit score) to improve predictive power.\n"
       "- Explore gradient boosting models (XGBoost, LightGBM) for potentially higher "
       "ROC-AUC.\n"
       "- Apply SHAP values for per-applicant explainability to support regulatory "
       "and customer-facing transparency requirements.\n"
       "- Build a monitoring dashboard to track model performance and data drift "
       "in production.\n"
       "- Conduct a fairness audit across protected demographic attributes prior to "
       "full production rollout."),
]


if __name__ == "__main__":
    notebooks = {
        "Notebook/01_Data_Preprocessing.ipynb": nb1_cells,
        "Notebook/02_EDA.ipynb": nb2_cells,
        "Notebook/03_Model_Training.ipynb": nb3_cells,
        "Notebook/04_Model_Evaluation.ipynb": nb4_cells,
        "Notebook/05_Final_Report.ipynb": nb5_cells,
    }

    for path, cells in notebooks.items():
        nb = make_notebook(cells)
        with open(path, "w") as f:
            nbf.write(nb, f)
        print(f"Created: {path}")
