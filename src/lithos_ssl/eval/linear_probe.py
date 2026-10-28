from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def linear_probe_accuracy(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    max_iter: int = 2000,
) -> float:
    """Fit a logistic-regression probe on `train_embeddings`, report accuracy on `test_embeddings`."""
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter),
    )
    probe.fit(train_embeddings, train_labels)
    predictions = probe.predict(test_embeddings)
    return float(accuracy_score(test_labels, predictions))
