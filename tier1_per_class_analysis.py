"""
Tier 1 — Per-Class Analysis (Out-of-Fold Evaluation)

Uses cross_val_predict to generate out-of-fold predictions
and computes per-class classification reports for all models.
"""

import warnings
import pandas as pd

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report

from evaluation_pipeline import load_and_prepare, define_models

warnings.filterwarnings("ignore", module="sklearn")


def evaluate_per_class(models, X, y, cv=5, random_state=42):
    """
    Run cross_val_predict for each model and generate classification reports.
    """

    skf = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state
    )

    reports = {}

    for name, model in models.items():

        print(f"Running OOF predictions for: {name}")

        y_pred = cross_val_predict(
            model,
            X,
            y,
            cv=skf,
            n_jobs=-1
        )

        reports[name] = classification_report(
            y,
            y_pred,
            output_dict=True
        )

    return reports


def summarize_minority_class(reports):
    """
    Extract churn class (1) metrics and rank models.
    """

    rows = []

    for model_name, report in reports.items():

        rows.append({
            "model": model_name,
            "precision_churn": report["1"]["precision"],
            "recall_churn": report["1"]["recall"],
            "f1_churn": report["1"]["f1-score"],
            "support": report["1"]["support"]
        })

    df = pd.DataFrame(rows)

    return df.sort_values(by="f1_churn", ascending=False)


def print_full_reports(reports):
    """
    Optional: print full classification reports per model.
    """

    for model_name, report in reports.items():

        print("\n" + "=" * 60)
        print(f"MODEL: {model_name}")
        print("=" * 60)

        df = pd.DataFrame(report).transpose()
        print(df)


def main():
    # Load data
    X, y = load_and_prepare()

    print(f"Data loaded: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Churn rate: {y.mean():.2%}")

    # Load models
    models = define_models()

    # Run per-class evaluation
    reports = evaluate_per_class(models, X, y)

    # Summary table (what you need for PR)
    summary = summarize_minority_class(reports)

    print("\n=== Minority Class (Churn=1) Comparison ===")
    print(summary.to_string(index=False))

    print_full_reports(reports)


if __name__ == "__main__":
    main()