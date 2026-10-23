"""Tests for dataset loading and metadata."""

import pytest
import pandas as pd
import numpy as np

from ml_visualizer.datasets import (
    DatasetLoader,
    DatasetInfo,
    TaskType,
    get_dataset_loader,
)


class TestDatasetLoader:
    def test_list_datasets(self):
        loader = get_dataset_loader()
        datasets = loader.list_datasets()
        assert len(datasets) == 5
        assert "iris" in datasets
        assert "breast_cancer" in datasets
        assert "diabetes" in datasets
        assert "synthetic_regression" in datasets
        assert "synthetic_blobs" in datasets

    def test_get_info(self):
        loader = get_dataset_loader()
        info = loader.get_info("iris")
        assert isinstance(info, DatasetInfo)
        assert info.name == "Iris"
        assert info.task_type == TaskType.CLASSIFICATION
        assert info.n_samples == 150
        assert info.n_features == 4
        assert info.target_names == ["setosa", "versicolor", "virginica"]

    def test_get_info_unknown(self):
        loader = get_dataset_loader()
        with pytest.raises(ValueError):
            loader.get_info("unknown_dataset")

    def test_load_iris(self):
        loader = get_dataset_loader()
        X, y, info = loader.load("iris")
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert info.name == "Iris"
        assert X.shape == (150, 4)
        assert y.shape == (150,)
        assert list(X.columns) == info.feature_names
        assert y.name == info.target_name

    def test_load_breast_cancer(self):
        loader = get_dataset_loader()
        X, y, info = loader.load("breast_cancer")
        assert X.shape == (569, 30)
        assert y.shape == (569,)
        assert info.task_type == TaskType.CLASSIFICATION

    def test_load_diabetes(self):
        loader = get_dataset_loader()
        X, y, info = loader.load("diabetes")
        assert X.shape == (442, 10)
        assert y.shape == (442,)
        assert info.task_type == TaskType.REGRESSION

    def test_load_synthetic_regression(self):
        loader = get_dataset_loader()
        X, y, info = loader.load("synthetic_regression")
        assert X.shape == (200, 5)
        assert y.shape == (200,)
        assert info.task_type == TaskType.REGRESSION

    def test_load_synthetic_blobs(self):
        loader = get_dataset_loader()
        X, y, info = loader.load("synthetic_blobs")
        assert X.shape == (300, 2)
        assert y is None
        assert info.task_type == TaskType.CLUSTERING

    def test_get_preview(self):
        loader = get_dataset_loader()
        preview = loader.get_preview("iris", n_rows=5)
        assert isinstance(preview, pd.DataFrame)
        assert len(preview) == 5
        assert "species" in preview.columns

    def test_reproducibility(self):
        loader1 = get_dataset_loader(random_state=42)
        loader2 = get_dataset_loader(random_state=42)
        X1, y1, _ = loader1.load("synthetic_regression")
        X2, y2, _ = loader2.load("synthetic_regression")
        assert np.allclose(X1.values, X2.values)
        assert np.allclose(y1.values, y2.values)

    def test_different_seeds(self):
        loader1 = get_dataset_loader(random_state=42)
        loader2 = get_dataset_loader(random_state=123)
        X1, _, _ = loader1.load("synthetic_regression")
        X2, _, _ = loader2.load("synthetic_regression")
        assert not np.allclose(X1.values, X2.values)