"""
Module 5 Week A — Integration: ML Evaluation Pipeline

Build a structured evaluation pipeline that compares 5 model
configurations using cross-validation with ColumnTransformer + Pipeline.
"""

import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore", module="sklearn")
    

NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents"]

CATEGORICAL_FEATURES = ["gender", "contract_type", "internet_service",
                        "payment_method"]


def load_and_prepare(filepath="data/telecom_churn.csv"):
    """Load data and separate features from target.

    Returns:
        Tuple of (X, y) where X is a DataFrame of features
        and y is a Series of the target (churned).
    """
    # Load CSV, drop customer_id, separate features and target
    df = pd.read_csv(filepath)

    # Drop ID column
    df = df.drop(columns=["customer_id"])

    # Separate target
    y = df["churned"]
    X = df.drop(columns=["churned"])

    return X, y


def build_preprocessor():
    """Build a ColumnTransformer for numeric and categorical features.

    Returns:
        ColumnTransformer that scales numeric features and
        one-hot encodes categorical features.
    """
    # Create a ColumnTransformer with StandardScaler for numeric
    #       and OneHotEncoder for categorical columns
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def define_models():
    """Define the 5 model configurations to compare.

    Two dummy baselines are included to teach two different lessons:
    most_frequent demonstrates the accuracy inflation problem on imbalanced
    data; stratified shows what random guessing in proportion to class
    frequencies looks like, so F1 carries meaningful signal when comparing.

    Returns:
        Dictionary mapping model name to (preprocessor, model) Pipeline.
    """
    # Create 5 Pipelines, each using the preprocessor + a model:
    #   1. "LogReg_default" — LogisticRegression with default C
    #   2. "LogReg_L1" — LogisticRegression with C=0.1, penalty='l1', solver='saga'
    #   3. "RidgeClassifier" — RidgeClassifier
    #   4. "Dummy_most_frequent" — DummyClassifier(strategy='most_frequent')
    #   5. "Dummy_stratified" — DummyClassifier(strategy='stratified', random_state=42)
    
    preprocessor = build_preprocessor()

    models = {
        "LogReg_default": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                C=1.0,
                random_state=42,
                max_iter=1000,
                class_weight="balanced"
            ))
        ]),

        "LogReg_L1": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                C=0.1,
                penalty="l1",
                solver="saga",
                random_state=42,
                max_iter=1000,
                class_weight="balanced"
            ))
        ]),

        "RidgeClassifier": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RidgeClassifier(
                alpha=1.0,
                random_state=42,
                class_weight="balanced"
            ))
        ]),

        "Dummy_most_frequent": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DummyClassifier(
                strategy="most_frequent"
            ))
        ]),

        "Dummy_stratified": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DummyClassifier(
                strategy="stratified",
                random_state=42
            ))
        ]),
    }

    return models


def evaluate_models(models, X, y, cv=5, random_state=42):
    """Run cross-validation on all models and return results.

    Args:
        models: Dictionary of {name: Pipeline}.
        X: Feature DataFrame.
        y: Target Series.
        cv: Number of folds.
        random_state: Random seed.

    Returns:
        DataFrame with columns: model, accuracy_mean, accuracy_std,
        precision_mean, recall_mean, f1_mean.
    """
    # Loop over models, run cross_validate with scoring metrics,
    #       collect results into a DataFrame

    skf = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state
    )

    results_list = []

    for name, pipeline in models.items():

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=skf,
            scoring=["accuracy", "precision", "recall", "f1"],
            n_jobs=-1
        )

        results_list.append({
            "model": name,
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
            "precision_mean": scores["test_precision"].mean(),
            "recall_mean": scores["test_recall"].mean(),
            "f1_mean": scores["test_f1"].mean(),
        })

    results_df = pd.DataFrame(results_list)

    return results_df


def final_evaluation(pipeline, X_train, X_test, y_train, y_test):
    """Train a pipeline on full training data and evaluate on the held-out test set.

    Use this on the best model from Task 4 as a final sanity check — the
    test-set metrics should be close to the CV estimates if the model
    generalizes. If they diverge substantially, the CV estimates were
    optimistic and you should investigate.

    Args:
        pipeline: An unfitted sklearn Pipeline (one entry from define_models).
        X_train, X_test: Feature DataFrames (train and held-out test).
        y_train, y_test: Target Series (train and held-out test).

    Returns:
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'.
    """
    # Fit the pipeline on (X_train, y_train), predict on X_test,
    #       compute and return the 4 metrics as a dictionary
    pipeline.fit(X_train, y_train)

    # Predict on test set
    y_pred = pipeline.predict(X_test)

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    return metrics


def recommend_model(results_df):
    """Print a recommendation based on the results.

    Args:
        results_df: DataFrame from evaluate_models.
    """
    print("\n=== Model Comparison Table (CV results) ===")
    print(results_df.to_string(index=False))
    print("\n=== Recommendation ===")
    print("Write your recommendation in the PR description.")


if __name__ == "__main__":
    data = load_and_prepare()
    if data is not None:
        X, y = data
        print(f"Data: {X.shape[0]} rows, {X.shape[1]} features")
        print(f"Churn rate: {y.mean():.2%}")

        # Create 80/20 train/test split. The test set is held out for the
        # final evaluation in Task 5 — do not use it during cross-validation.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

        models = define_models()
        if models:
            # Task 4: cross-validation on training data only
            results = evaluate_models(models, X_train, y_train)
            if results is not None:
                recommend_model(results)

                # Task 5: final evaluation on the held-out test set.
                # Remove dummy models
                real_models = results[~results["model"].str.contains("Dummy")]

                # Select model with highest Mean F1
                best_row = real_models.loc[real_models["f1_mean"].idxmax()]
                best_model_name = best_row["model"]

                print(f"\nBest model based on CV F1: {best_model_name}")

                best_pipeline = models[best_model_name]

                # Run final evaluation
                test_metrics = final_evaluation(
                    best_pipeline,
                    X_train,
                    X_test,
                    y_train,
                    y_test
                )

                print("\n=== Final Test Set Evaluation ===")
                for metric, value in test_metrics.items():
                    print(f"{metric.capitalize()}: {value:.4f}")

                print("\n=== CV vs Test Comparison (F1) ===")
                print(f"CV Mean F1: {best_row['f1_mean']:.4f}")
                print(f"Test F1: {test_metrics['f1']:.4f}")


"""
Recommendation:

Based on cross-validation F1 score, I recommend the RidgeClassifier as the best-performing configuration.
It achieved the highest mean CV F1 (0.341) among the real models and clearly outperformed both dummy 
baselines on F1. Although the most-frequent dummy achieved much higher accuracy (0.84), 
it had an F1 of 0.00 because it completely ignored churners — demonstrating that accuracy alone
is misleading on imbalanced churn data. The recommended Ridge model trades lower 
accuracy (0.61 CV, 0.65 test) for substantially higher recall (~0.62 CV, 0.65 test), 
which is desirable because missing a churning customer is more costly than a false alarm. 
Compared to the stratified dummy (F1 ≈ 0.17), the Ridge model’s F1 (~0.34 CV, 0.38 test) 
is roughly double random guessing proportional to class frequencies, indicating meaningful 
signal but still moderate predictive power. Importantly, the test-set F1 (0.381) slightly 
exceeds the CV estimate (0.341), confirming that the model generalizes as expected and that 
the CV procedure was reliable. While this performance may be acceptable for a first baseline, 
it likely represents the ceiling of what linear models can extract from these features, 
suggesting that more flexible models (e.g., tree-based methods) may further improve churn detection.
"""