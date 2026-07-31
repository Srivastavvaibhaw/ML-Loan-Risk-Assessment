"""
main.py
-------
Entry point for the Loan Risk Assessment System.

Executes the full pipeline end-to-end:
    1. Load dataset
    2. Clean data
    3. Feature engineering
    4. Exploratory data analysis (EDA)
    5. Train/test split & scaling
    6. Train models (Logistic Regression, Decision Tree, Random Forest)
    7. Evaluate models
    8. Save models
    9. Generate reports (model comparison table + deployment recommendation)

Usage:
    python main.py
"""

import sys
import time

from src.train import run_full_pipeline
from src.utils import get_logger, print_section

logger = get_logger(__name__)


def main() -> int:
    """Run the complete Loan Risk Assessment ML pipeline.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    start_time = time.time()
    print_section("Loan Risk Assessment System - Pipeline Started", width=80)

    try:
        results = run_full_pipeline()
    except Exception as exc:
        logger.error(f"Pipeline failed with error: {exc}")
        raise

    elapsed = time.time() - start_time

    print_section("Pipeline Completed Successfully", width=80)
    print(f"Total execution time: {elapsed:.2f} seconds")
    print("\nGenerated artifacts:")
    print("  - outputs/cleaned_dataset.csv")
    print("  - outputs/train.csv")
    print("  - outputs/test.csv")
    print("  - outputs/model_comparison.csv")
    print("  - models/logistic_regression.pkl")
    print("  - models/decision_tree.pkl")
    print("  - models/random_forest.pkl")
    print("  - models/scaler.pkl")
    print("  - Images/*.png (14 EDA + evaluation figures)")
    print("  - reports/deployment_recommendation.md")

    comparison_df = results["comparison_df"]
    best_model = comparison_df.iloc[0]["Model"]
    print(f"\nRecommended model for deployment: {best_model}")
    print(f"(ROC AUC: {comparison_df.iloc[0]['ROC AUC']:.4f})")

    print("\n✅ Loan Risk Assessment System pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
