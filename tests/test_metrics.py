"""Tests for evaluation metrics."""

import pytest
import numpy as np
import pandas as pd

from ml_visualizer.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_clustering_metrics,
    ClassificationMetrics,
    RegressionMetrics,
    ClusteringMetrics,
)


class TestClassificationMetrics:
    def test_binary_classification(self):
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 1, 0, 0])

        metrics = compute_classification_metrics(y_true, y_pred, ["negative", "positive"])

        assert isinstance(metrics, ClassificationMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1 <= 1
        assert metrics.confusion_matrix.shape == (2, 2)
        assert "negative" in metrics.per_class
        assert "positive" in metrics.per_class

    def test_multiclass_classification(self):
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 2, 1, 0, 1])

        metrics = compute_classification_metrics(y_true, y_pred, ["class_a", "class_b", "class_c"])

        assert metrics.confusion_matrix.shape == (3, 3)
        assert metrics.accuracy == 0.75  # 6/8 correct

    def test_perfect_classification(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        metrics = compute_classification_metrics(y_true, y_pred)

        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0

    def test_all_wrong_classification(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])

        metrics = compute_classification_metrics(y_true, y_pred)

        assert metrics.accuracy == 0.0

    def test_zero_division_handling(self):
        # All predictions same class, but true has both
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])

        metrics = compute_classification_metrics(y_true, y_pred)

        # Should not crash, precision/recall for class 1 should be 0
        assert metrics.precision >= 0
        assert metrics.recall >= 0


class TestRegressionMetrics:
    def test_regression_metrics(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        metrics = compute_regression_metrics(y_true, y_pred)

        assert isinstance(metrics, RegressionMetrics)
        assert metrics.mae >= 0
        assert metrics.rmse >= 0
        assert metrics.rmse >= metrics.mae  # RMSE >= MAE
        assert metrics.r2 <= 1.0

    def test_perfect_regression(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])

        metrics = compute_regression_metrics(y_true, y_pred)

        assert metrics.mae == 0.0
        assert metrics.rmse == 0.0
        assert metrics.r2 == 1.0

    def test_negative_r2(self):
        # Model worse than horizontal line
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([10.0, 10.0, 10.0, 10.0])

        metrics = compute_regression_metrics(y_true, y_pred)

        assert metrics.r2 < 0


class TestClusteringMetrics:
    def test_valid_clustering(self):
        X = np.array([[0, 0], [0, 1], [5, 5], [5, 6]])
        labels = np.array([0, 0, 1, 1])
        inertia = 1.0

        metrics = compute_clustering_metrics(X, labels, inertia)

        assert isinstance(metrics, ClusteringMetrics)
        assert metrics.inertia == 1.0
        assert metrics.n_clusters == 2
        assert metrics.n_samples == 4
        assert metrics.silhouette is not None
        assert -1 <= metrics.silhouette <= 1

    def test_single_cluster(self):
        X = np.array([[0, 0], [0, 1], [1, 0]])
        labels = np.array([0, 0, 0])
        inertia = 0.5

        metrics = compute_clustering_metrics(X, labels, inertia)

        assert metrics.n_clusters == 1
        assert metrics.silhouette is None

    def test_each_point_own_cluster(self):
        X = np.array([[0, 0], [1, 1], [2, 2]])
        labels = np.array([0, 1, 2])
        inertia = 0.0

        metrics = compute_clustering_metrics(X, labels, inertia)

        assert metrics.n_clusters == 3
        assert metrics.silhouette is None

    def test_silhouette_samples_shape(self):
        X = np.array([[0, 0], [0, 1], [5, 5], [5, 6]])
        labels = np.array([0, 0, 1, 1])
        inertia = 1.0

        metrics = compute_clustering_metrics(X, labels, inertia)

        assert metrics.silhouette_samples is not None
        assert len(metrics.silhouette_samples) == 4