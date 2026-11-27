"""Small deterministic logistic models and tie-aware binary metrics."""
from __future__ import annotations
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


class LogisticModel:
    def __init__(self, regularization=.03, steps=1400, learning_rate=.07):
        self.regularization, self.steps, self.learning_rate = regularization, steps, learning_rate

    def fit(self, x, y):
        self.mean = np.asarray(x).mean(0)
        self.scale = np.asarray(x).std(0)
        self.scale[self.scale < .05] = 1
        z = (x - self.mean) / self.scale
        self.weights = np.zeros(z.shape[1])
        self.bias = 0.0
        # Full-batch Adam, deterministic for a fixed matrix and numerical library.
        m, v = np.zeros(z.shape[1] + 1), np.zeros(z.shape[1] + 1)
        for step in range(1, self.steps + 1):
            err = sigmoid(z @ self.weights + self.bias) - y
            grad = np.r_[z.T @ err / len(y) + self.regularization * self.weights, err.mean()]
            m, v = .9 * m + .1 * grad, .999 * v + .001 * grad * grad
            update = self.learning_rate * (m / (1 - .9 ** step)) / (np.sqrt(v / (1 - .999 ** step)) + 1e-8)
            self.weights -= update[:-1]
            self.bias -= update[-1]
        return self

    def predict_proba(self, x):
        return sigmoid(((np.asarray(x) - self.mean) / self.scale) @ self.weights + self.bias)

    def to_dict(self):
        return {"weights": self.weights.tolist(), "bias": float(self.bias), "mean": self.mean.tolist(), "scale": self.scale.tolist(), "regularization": self.regularization, "steps": self.steps}

    @classmethod
    def from_dict(cls, value):
        model = cls()
        for key in ("weights", "mean", "scale"):
            setattr(model, key, np.asarray(value[key], dtype=float))
        model.bias = float(value["bias"])
        return model


def metrics(y, probability):
    y, p = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    predicted = p >= .5
    tp, tn = int(((y == 1) & predicted).sum()), int(((y == 0) & ~predicted).sum())
    fp, fn = int(((y == 0) & predicted).sum()), int(((y == 1) & ~predicted).sum())
    def divide(a, b):
        return float(a / b) if b else 0.0
    precision, recall = divide(tp, tp + fp), divide(tp, tp + fn)
    real_precision, real_recall = divide(tn, tn + fn), divide(tn, tn + fp)
    # Group tied scores for threshold curves. AP is the step integral PR-AUC.
    thresholds = np.r_[np.inf, np.unique(p)[::-1]]
    tpr, fpr, precision_curve = [0.0], [0.0], [1.0]
    for threshold in thresholds[1:]:
        positive = p >= threshold
        hits, false = int(((y == 1) & positive).sum()), int(((y == 0) & positive).sum())
        tpr.append(divide(hits, int((y == 1).sum())))
        fpr.append(divide(false, int((y == 0).sum())))
        precision_curve.append(divide(hits, hits + false))
    roc_auc = float(np.sum(np.diff(fpr) * (np.asarray(tpr[:-1]) + np.asarray(tpr[1:])) / 2)) if len(np.unique(y)) == 2 else None
    pr_auc = float(np.sum(np.diff(tpr) * np.asarray(precision_curve)[1:])) if np.any(y == 1) else None
    return {"accuracy": divide(tp + tn, len(y)), "precision": precision, "recall": recall, "f1": divide(2 * precision * recall, precision + recall), "roc_auc": roc_auc, "pr_auc": pr_auc,
            "confusion_matrix": [[tn, fp], [fn, tp]], "support": int(len(y)),
            "per_class": {"REAL": {"precision": real_precision, "recall": real_recall, "f1": divide(2 * real_precision * real_recall, real_precision + real_recall), "support": tn + fp}, "FAKE": {"precision": precision, "recall": recall, "f1": divide(2 * precision * recall, precision + recall), "support": tp + fn}},
            "curves": {"false_positive_rate": fpr, "true_positive_rate": tpr, "precision": precision_curve, "recall": tpr}}
