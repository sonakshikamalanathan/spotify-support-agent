"""Metric helpers with bootstrap confidence intervals (the golden set is small, so every
headline number is reported with a 95% CI)."""
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score, fbeta_score

from config import SEED


def bootstrap_ci(metric_fn, y_true, y_pred, n_boot=1000, seed=SEED):
    """Return (point_estimate, low, high) for metric_fn using a percentile bootstrap."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    point = metric_fn(y_true, y_pred)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(point), float(low), float(high)


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def escalation_metrics(y_true, y_pred):
    """Positive class = escalate. Recall matters most: a missed escalation is the costly error."""
    y_true, y_pred = np.asarray(y_true, dtype=bool), np.asarray(y_pred, dtype=bool)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "escalation_rate": float(y_pred.mean()),
        "missed_escalations": int((y_true & ~y_pred).sum()),
        "unneeded_escalations": int((~y_true & y_pred).sum()),
    }


def recall_fn(y_true, y_pred):
    return recall_score(np.asarray(y_true, dtype=bool), np.asarray(y_pred, dtype=bool), zero_division=0)


def weighted_kappa(a, b):
    """Linear-weighted Cohen's kappa for ordinal 1-5 ratings."""
    return float(cohen_kappa_score(a, b, weights="linear"))


def kappa(a, b):
    return float(cohen_kappa_score(a, b))


def fmt_ci(point, low, high, pct=False):
    if pct:
        return f"{point:.1%} [{low:.1%}, {high:.1%}]"
    return f"{point:.3f} [{low:.3f}, {high:.3f}]"
