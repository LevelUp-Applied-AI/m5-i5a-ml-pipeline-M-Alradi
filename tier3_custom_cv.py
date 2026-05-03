"""
Tier 3 — Custom Stratified K-Fold Cross Validation (from scratch)

No sklearn cross-validation utilities allowed.
Only numpy + pandas.
"""

import numpy as np
import pandas as pd


def stratified_k_fold_indices(y, k, random_state=42):
    """
    Create stratified folds preserving class distribution.

    Returns:
        List of (train_idx, test_idx) tuples
    """

    rng = np.random.RandomState(random_state)

    y = np.array(y)
    classes = np.unique(y)

    # Store indices per class
    class_indices = {
        c: np.where(y == c)[0]
        for c in classes
    }

    # Shuffle indices per class
    for c in classes:
        rng.shuffle(class_indices[c])

    # Create empty folds
    folds = [[] for _ in range(k)]

    # Distribute each class proportionally
    for c in classes:
        idx = class_indices[c]
        split = np.array_split(idx, k)

        for i in range(k):
            folds[i].extend(split[i])

    # Convert to train/test splits
    results = []

    all_idx = np.arange(len(y))

    for i in range(k):
        test_idx = np.array(sorted(folds[i]))
        train_idx = np.array(sorted(np.setdiff1d(all_idx, test_idx)))

        results.append((train_idx, test_idx))

    return results


def cross_val_score_custom(model, X, y, k=5, random_state=42, metric=None):
    """
    Custom stratified cross-validation engine.

    Args:
        model: sklearn-like model with fit/predict
        X: pandas DataFrame or numpy array
        y: target array
        k: number of folds
        metric: function(y_true, y_pred)

    Returns:
        np.array of k scores
    """

    if metric is None:
        from sklearn.metrics import accuracy_score
        metric = accuracy_score

    X = np.array(X)
    y = np.array(y)

    # Edge case: class size < k
    unique, counts = np.unique(y, return_counts=True)
    if np.any(counts < k):
        raise ValueError("One or more classes have fewer samples than k.")

    folds = stratified_k_fold_indices(y, k, random_state)

    scores = []

    for train_idx, test_idx in folds:

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        score = metric(y_test, preds)
        scores.append(score)

    return np.array(scores)


def compare_with_sklearn(model, X, y, k=5, random_state=42):
    """
    Validate custom CV against sklearn cross_val_score.
    """

    from sklearn.model_selection import StratifiedKFold, cross_val_score

    skf = StratifiedKFold(
        n_splits=k,
        shuffle=True,
        random_state=random_state
    )

    sk_scores = cross_val_score(
        model,
        X,
        y,
        cv=skf,
        scoring="accuracy"
    )

    custom_scores = cross_val_score_custom(
        model,
        X,
        y,
        k=k,
        random_state=random_state
    )

    return {
        "sklearn_mean": sk_scores.mean(),
        "custom_mean": custom_scores.mean(),
        "difference": abs(sk_scores.mean() - custom_scores.mean())
    }