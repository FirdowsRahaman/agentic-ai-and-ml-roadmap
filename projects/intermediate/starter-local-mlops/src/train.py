"""Trains a leak-free churn pipeline, evaluates PR-AUC, and saves safe non-pickle .npz weights."""
from __future__ import annotations

from pathlib import Path
import numpy as np

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "churn_model.npz"


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def compute_pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    order = np.argsort(-y_scores)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(np.sum(y_true == 1)), 1)
    # Stepwise area under Precision-Recall curve
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


class LeakFreeChurnPipeline:
    """Fits StandardScaler strictly on X_train + L2-regularized Logistic Classifier (zero pickle!)."""

    def __init__(self, lr: float = 0.15, epochs: int = 800, l2: float = 1e-3):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LeakFreeChurnPipeline":
        self.mean_ = X_train.mean(axis=0)
        self.std_ = np.where(X_train.std(axis=0) > 1e-8, X_train.std(axis=0), 1.0)
        X_scaled = (X_train - self.mean_) / self.std_

        n, d = X_scaled.shape
        self.weights_ = np.zeros(d, dtype=float)
        self.bias_ = 0.0
        for _ in range(self.epochs):
            probs = _sigmoid(X_scaled @ self.weights_ + self.bias_)
            diff = probs - y_train
            dw = (X_scaled.T @ diff) / n + self.l2 * self.weights_
            db = float(np.mean(diff))
            self.weights_ -= self.lr * dw
            self.bias_ -= self.lr * db
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None and self.weights_ is not None
        X_scaled = (X - self.mean_) / self.std_
        return _sigmoid(X_scaled @ self.weights_ + self.bias_)

    def save_npz(self, path: Path, baseline_spend: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self.mean_,
            std=self.std_,
            weights=self.weights_,
            bias=np.array([self.bias_]),
            baseline_spend=baseline_spend,
        )

    @classmethod
    def load_npz(cls, path: Path) -> tuple["LeakFreeChurnPipeline", np.ndarray]:
        data = np.load(path, allow_pickle=False)
        pipe = cls()
        pipe.mean_ = data["mean"]
        pipe.std_ = data["std"]
        pipe.weights_ = data["weights"]
        pipe.bias_ = float(data["bias"][0])
        return pipe, data["baseline_spend"]


def train_and_save() -> float:
    rng = np.random.default_rng(42)
    n = 1000
    tenure = rng.integers(1, 72, size=n).astype(float)
    monthly_spend = rng.normal(loc=65.0, scale=20.0, size=n).clip(5.0, 200.0)
    support_calls = rng.poisson(lam=1.5, size=n).astype(float)

    X = np.column_stack([tenure, monthly_spend, support_calls])
    logits = -0.05 * tenure + 0.02 * monthly_spend + 0.70 * support_calls - 1.1
    y = (rng.random(n) < _sigmoid(logits)).astype(int)

    # Leak-free chronological/indexed 80/20 split
    split = int(0.8 * n)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    pipeline = LeakFreeChurnPipeline().fit(X_train, y_train)
    pr_auc = compute_pr_auc(y_test, pipeline.predict_proba(X_test))
    pipeline.save_npz(MODEL_PATH, baseline_spend=X_train[:, 1])
    return pr_auc


if __name__ == "__main__":
    score = train_and_save()
    print(f"Saved safe zero-pickle .npz model to {MODEL_PATH} | Validation PR-AUC: {score:.4f}")
