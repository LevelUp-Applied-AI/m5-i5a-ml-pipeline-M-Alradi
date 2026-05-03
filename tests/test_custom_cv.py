import numpy as np
from tier3_custom_cv import stratified_k_fold_indices, cross_val_score_custom


# ---------------------------
# Dummy model for testing
# ---------------------------

class DummyModel:
    def fit(self, X, y):
        self.majority = np.bincount(y.astype(int)).argmax()

    def predict(self, X):
        return np.full(len(X), self.majority)


# ---------------------------
# Test 1: Correct number of folds
# ---------------------------

def test_correct_number_of_folds():
    y = np.array([0]*50 + [1]*50)

    folds = stratified_k_fold_indices(y, k=5)

    assert len(folds) == 5


# ---------------------------
# Test 2: No overlap between test sets
# ---------------------------

def test_no_overlap_between_folds():
    y = np.array([0]*50 + [1]*50)

    folds = stratified_k_fold_indices(y, k=5)

    all_test = []

    for _, test_idx in folds:
        all_test.extend(test_idx)

    # No duplicates allowed
    assert len(all_test) == len(set(all_test))


# ---------------------------
# Test 3: Class ratio preservation (approx)
# ---------------------------

def test_class_ratio_preserved():
    y = np.array([0]*90 + [1]*10)

    folds = stratified_k_fold_indices(y, k=5)

    for _, test_idx in folds:
        y_test = y[test_idx]

        ratio = y_test.mean()  # proportion of class 1

        # original ratio = 0.1
        assert abs(ratio - 0.1) < 0.05


# ---------------------------
# Test 4: CV returns correct length
# ---------------------------

def test_cv_score_length():
    X = np.random.randn(100, 3)
    y = np.array([0]*50 + [1]*50)

    model = DummyModel()

    scores = cross_val_score_custom(model, X, y, k=5)

    assert len(scores) == 5