"""ML Visualizer - Visualization utilities for Streamlit."""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Tuple
import io
import base64

from .metrics import (
    plot_confusion_matrix,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_regression_line,
    plot_feature_importance,
    plot_clusters,
    plot_silhouette,
    plot_decision_boundary,
)


def fig_to_image(fig: plt.Figure) -> bytes:
    """Convert matplotlib figure to bytes for Streamlit display."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.read()


def display_figure(fig: plt.Figure, caption: str = "") -> None:
    """Display a matplotlib figure in Streamlit."""
    img_bytes = fig_to_image(fig)
    st.image(img_bytes, caption=caption, use_container_width=True)


def plot_dataset_overview(X: pd.DataFrame, y: Optional[pd.Series], target_name: Optional[str] = None) -> plt.Figure:
    """Create an overview visualization of the dataset."""
    n_features = len(X.columns)

    if n_features <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
    else:
        n_cols = min(4, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
        if n_rows > 1:
            axes = axes.flatten()
        else:
            axes = [axes] if n_cols == 1 else axes

    for i, col in enumerate(X.columns):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            break

        if y is not None and target_name:
            unique_classes = y.unique()
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
            for cls, color in zip(unique_classes, colors):
                mask = y == cls
                label = str(cls)
                ax.hist(X.loc[mask, col], bins=20, alpha=0.5, label=label, color=color, edgecolor="k", linewidth=0.5)
            ax.legend(fontsize=8)
        else:
            ax.hist(X[col], bins=30, alpha=0.7, edgecolor="k", linewidth=0.5, color="steelblue")

        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=14)
    plt.tight_layout()
    return fig


def plot_correlation_matrix(X: pd.DataFrame, figsize=(10, 8)) -> plt.Figure:
    """Create correlation matrix heatmap."""
    fig, ax = plt.subplots(figsize=figsize)

    corr = X.corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                   fontsize=7, color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")

    ax.set_title("Feature Correlation Matrix")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    return fig


def create_comparison_table(results: dict, metric: str = "accuracy") -> pd.DataFrame:
    """Create a comparison DataFrame from multiple model results."""
    rows = []
    for model_name, result in results.items():
        if hasattr(result, 'metrics') and hasattr(result.metrics, metric):
            rows.append({
                "Model": model_name,
                metric.capitalize(): getattr(result.metrics, metric),
            })
        elif isinstance(result, dict) and metric in result:
            rows.append({
                "Model": model_name,
                metric.capitalize(): result[metric],
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(metric.capitalize(), ascending=False)
    return df


def plot_model_comparison_bar(
    results: dict,
    metric: str = "accuracy",
    figsize=(10, 6),
) -> plt.Figure:
    """Create bar chart comparing models on a metric."""
    fig, ax = plt.subplots(figsize=figsize)

    model_names = []
    values = []

    for model_name, result in results.items():
        if hasattr(result, 'metrics') and hasattr(result.metrics, metric):
            model_names.append(model_name)
            values.append(getattr(result.metrics, metric))
        elif isinstance(result, dict) and metric in result:
            model_names.append(model_name)
            values.append(result[metric])

    if not model_names:
        ax.text(0.5, 0.5, "No data to compare", ha="center", va="center", transform=ax.transAxes)
        return fig

    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    bars = ax.barh(range(len(model_names)), values, color=colors, edgecolor="k", linewidth=0.5)

    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.3f}",
               va="center", fontsize=10)

    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xlabel(metric.capitalize())
    ax.set_title(f"Model Comparison: {metric.capitalize()}")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def plot_metric_radar(results: dict, metrics: list[str], figsize=(8, 8)) -> plt.Figure:
    """Create radar chart comparing models across multiple metrics."""
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    model_names = list(results.keys())
    n_metrics = len(metrics)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    for i, (model_name, result) in enumerate(results.items()):
        values = []
        valid = True
        for metric in metrics:
            if hasattr(result, 'metrics') and hasattr(result.metrics, metric):
                values.append(getattr(result.metrics, metric))
            elif isinstance(result, dict) and metric in result:
                values.append(result[metric])
            else:
                valid = False
                break

        if not valid:
            continue

        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=model_name, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison Radar", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    return fig