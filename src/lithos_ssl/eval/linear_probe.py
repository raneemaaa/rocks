"""Linear-probe evaluation of a frozen embedding space.

The other standard SSL evaluation protocol, alongside kNN (eval/knn.py): freeze the
backbone entirely and fit the *simplest possible* classifier -- a linear decision
boundary -- on top of its embeddings. If a linear classifier can separate the mineral
classes well, the backbone has organised its embedding space in a genuinely useful way
(a strong representation makes even "dumb" downstream classifiers work well); if it
can't, the information needed might still be in there somewhere, but not laid out
usefully. This is slower than kNN (it's an actual training loop, scikit-learn's
L-BFGS solver here) so it's used only for the final, end-of-run evaluation
(scripts/evaluate.py), not every few epochs during pretraining.

We standardize (zero mean, unit variance) the embeddings before fitting, since raw
backbone features from different architectures can live on very different scales and
logistic regression is sensitive to that.
"""

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
