"""ML Visualizer - Evaluation metrics."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, silhouette_samples
)
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    report: str
    per_class: dict


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float


@dataclass
class ClusteringMetrics:
    inertia: float
    silhouette: Optional[float]
    silhouette_samples: Optional[np.ndarray]
    n_clusters: int
    n_samples: int


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list[str]] = None,
) -> ClassificationMetrics:
    """Compute comprehensive classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

    if target_names is not None:
        per_class_report = classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
        )
        per_class = {name: per_class_report[name] for name in target_names if name in per_class_report}
    else:
        per_class = {}

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=cm,
        report=report,
        per_class=per_class,
    )


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Compute regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return RegressionMetrics(mae=mae, rmse=rmse, r2=r2)


def compute_clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    inertia: float,
) -> ClusteringMetrics:
    """Compute clustering metrics."""
    n_clusters = len(np.unique(labels))
    n_samples = len(labels)

    silhouette = None
    silhouette_samples_vals = None

    if n_clusters > 1 and n_clusters < n_samples:
        try:
            silhouette = silhouette_score(X, labels)
            silhouette_samples_vals = silhouette_samples(X, labels)
        except Exception:
            pass

    return ClusteringMetrics(
        inertia=inertia,
        silhouette=silhouette,
        silhouette_samples=silhouette_samples_vals,
        n_clusters=n_clusters,
        n_samples=n_samples,
    )


def plot_confusion_matrix(cm: np.ndarray, target_names: Optional[list[str]] = None, figsize=(6, 5)) -> plt.Figure:
    """Create confusion matrix visualization."""
    fig, ax = plt.subplots(figsize=figsize)

    if target_names is not None:
        display_labels = target_names
    else:
        display_labels = [str(i) for i in range(cm.shape[0])]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, figsize=(6, 5)) -> plt.Figure:
    """Create actual vs predicted scatter plot for regression."""
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, figsize=(6, 5)) -> plt.Figure:
    """Create residual plot for regression."""
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidth=0.5)
    ax1.axhline(y=0, color="r", linestyle="--", lw=2)
    ax1.set_xlabel("Predicted Values")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Predicted")
    ax1.grid(True, alpha=0.3)

    ax2.hist(residuals, bins=20, edgecolor="k", alpha=0.7)
    ax2.axvline(x=0, color="r", linestyle="--", lw=2)
    ax2.set_xlabel("Residual Value")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Residual Distribution")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_regression_line(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_name: str,
    target_name: str,
    figsize=(6, 5),
) -> plt.Figure:
    """Plot regression line for single feature regression."""
    fig, ax = plt.subplots(figsize=figsize)

    sort_idx = np.argsort(X)
    ax.scatter(X, y_true, alpha=0.6, label="Actual", edgecolors="k", linewidth=0.5)
    ax.plot(X[sort_idx], y_pred[sort_idx], "r-", lw=2, label="Predicted")
    ax.set_xlabel(feature_name)
    ax.set_ylabel(target_name)
    ax.set_title(f"Regression: {target_name} vs {feature_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_feature_importance(importance: pd.Series, top_n: int = 15, figsize=(8, 6)) -> plt.Figure:
    """Create feature importance bar plot."""
    fig, ax = plt.subplots(figsize=figsize)

    plot_data = importance.head(top_n).sort_values(ascending=True)
    bars = ax.barh(range(len(plot_data)), plot_data.values, edgecolor="k", linewidth=0.5)
    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(plot_data.index)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    return fig


def plot_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: Optional[np.ndarray] = None,
    feature_names: tuple[str, str] = ("Feature 1", "Feature 2"),
    figsize=(7, 6),
) -> plt.Figure:
    """Create cluster visualization for 2D data."""
    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        ax.scatter(X[mask, 0], X[mask, 1], c=[color], label=f"Cluster {label}",
                  alpha=0.7, edgecolors="k", linewidth=0.5, s=50)

    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="x", s=200,
                  linewidth=3, label="Centroids", zorder=5)

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title("K-Means Clustering")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    silhouette_samples_vals: np.ndarray,
    figsize=(8, 6),
) -> plt.Figure:
    """Create silhouette plot for clustering evaluation."""
    fig, ax = plt.subplots(figsize=figsize)

    n_clusters = len(np.unique(labels))
    y_lower = 10

    for i in range(n_clusters):
        cluster_silhouette = silhouette_samples_vals[labels == i]
        cluster_silhouette.sort()
        size = cluster_silhouette.shape[0]
        y_upper = y_lower + size

        color = plt.cm.Set1(i / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette,
                        facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size, str(i))
        y_lower = y_upper + 10

    avg_silhouette = np.mean(silhouette_samples_vals)
    ax.axvline(x=avg_silhouette, color="red", linestyle="--", label=f"Average: {avg_silhouette:.3f}")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title("Silhouette Analysis")
    ax.legend()
    ax.set_xlim([-0.1, 1])
    plt.tight_layout()
    return fig


def plot_decision_boundary(
    xx: np.ndarray,
    yy: np.ndarray,
    Z: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: tuple[str, str],
    target_names: Optional[list[str]] = None,
    figsize=(8, 6),
) -> plt.Figure:
    """Create decision boundary visualization."""
    fig, ax = plt.subplots(figsize=figsize)

    cmap_light = plt.cm.Pastel1
    cmap_bold = plt.cm.Set1

    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

    unique_classes = np.unique(y)
    for i, cls in enumerate(unique_classes):
        mask = y == cls
        label = target_names[cls] if target_names and cls < len(target_names) else f"Class {cls}"
        ax.scatter(X[mask, 0], X[mask, 1], c=[cmap_bold(i)], label=label,
                  alpha=0.8, edgecolors="k", linewidth=0.5, s=50)

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title("Decision Boundary")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig