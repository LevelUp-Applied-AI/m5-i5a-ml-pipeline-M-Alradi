"""
Tier 2 — Pipeline Factory with Feature Engineering

This module:
1. Builds reusable sklearn pipelines via a factory function
2. Adds polynomial + interaction features
3. Compares models with and without feature engineering
"""

import warnings
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import cross_validate, StratifiedKFold

from evaluation_pipeline import load_and_prepare, define_models

warnings.filterwarnings("ignore", module="sklearn")


# -------------------------------------------------------
# Feature Engineering Transformer
# -------------------------------------------------------

def build_preprocessor(numeric_features, categorical_features, use_feature_engineering=False):
    """
    Builds preprocessing pipeline with optional feature engineering.
    """

    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    )

    if use_feature_engineering:
        numeric_pipeline = Pipeline(steps=[
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("scaler", numeric_transformer)
        ])
    else:
        numeric_pipeline = numeric_transformer

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


# -------------------------------------------------------
# Pipeline Factory
# -------------------------------------------------------

def build_pipeline(model, numeric_features, categorical_features, use_feature_engineering=False):
    """
    Factory function that builds full ML pipeline.
    """

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        use_feature_engineering=use_feature_engineering
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline


# -------------------------------------------------------
# Evaluation function
# -------------------------------------------------------

def evaluate_models(models, X, y, numeric_features, categorical_features, use_feature_engineering=False):
    """
    Run CV evaluation for a set of models.
    """

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []

    for name, model in models.items():

        pipeline = build_pipeline(
            model,
            numeric_features,
            categorical_features,
            use_feature_engineering=use_feature_engineering
        )

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=skf,
            scoring=["accuracy", "f1"],
            n_jobs=-1
        )

        results.append({
            "model": name,
            "accuracy_mean": scores["test_accuracy"].mean(),
            "f1_mean": scores["test_f1"].mean()
        })

    return pd.DataFrame(results)


# -------------------------------------------------------
# Main comparison runner
# -------------------------------------------------------

def main():

    # Load data
    X, y = load_and_prepare()

    numeric_features = [
        "tenure", "monthly_charges", "total_charges",
        "num_support_calls", "senior_citizen",
        "has_partner", "has_dependents"
    ]

    categorical_features = [
        "gender", "contract_type", "internet_service",
        "payment_method"
    ]

    # Base models (from your existing pipeline)
    base_models = define_models()

    # Remove pipelines (we rebuild them via factory)
    # Extract only model objects
    clean_models = {
        name: pipe.named_steps["model"]
        for name, pipe in base_models.items()
    }

    print("\n================ BASELINE (NO FEATURE ENGINEERING) ================")

    baseline_results = evaluate_models(
        clean_models,
        X,
        y,
        numeric_features,
        categorical_features,
        use_feature_engineering=False
    )

    print(baseline_results)

    print("\n================ WITH FEATURE ENGINEERING =========================")

    fe_results = evaluate_models(
        clean_models,
        X,
        y,
        numeric_features,
        categorical_features,
        use_feature_engineering=True
    )

    print(fe_results)

    print("\n================ COMPARISON SUMMARY ===============================")

    comparison = baseline_results.merge(
        fe_results,
        on="model",
        suffixes=("_baseline", "_feat_eng")
    )

    print(comparison)


if __name__ == "__main__":
    main()