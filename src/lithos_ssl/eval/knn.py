from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def knn_accuracy(
    reference_embeddings: np.ndarray,
    reference_labels: np.ndarray,
    query_embeddings: np.ndarray,
    query_labels: np.ndarray,
    k: int = 20,
) -> float:
    """Fraction of `query_embeddings` whose kNN-predicted label matches the true one.

    Uses cosine distance (the standard choice for comparing embeddings from models
    trained with a contrastive/similarity-based loss, where only the *direction* of an
    embedding is meaningful, not its magnitude).
    """
    k = min(k, len(reference_embeddings))
    classifier = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    classifier.fit(reference_embeddings, reference_labels)
    predictions = classifier.predict(query_embeddings)
    return float(accuracy_score(query_labels, predictions))
