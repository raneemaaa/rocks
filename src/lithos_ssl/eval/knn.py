"""k-nearest-neighbour evaluation of a frozen embedding space.

This is the standard "does this representation even separate the classes" check used
throughout the self-supervised learning literature (it's how the DINO paper reports its
headline numbers): embed a labelled "reference" set, embed a labelled "query" set, and
for each query, vote on its label using the k closest reference embeddings by cosine
similarity. No training/gradient descent involved -- which is exactly why it's cheap
enough to run *every few epochs during pretraining* (see eval/callbacks.py) to track
whether the representation is actually improving, not just whether the pretext loss is
going down (a lower SSL loss doesn't always mean better downstream features).
"""

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
