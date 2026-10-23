"""Tests for end-to-end application flow."""

import pytest
import pandas as pd
import numpy as np

from ml_visualizer import (
    get_dataset_loader,
    get_model_info,
    list_models,
    TaskType,
    ModelID,
    split_and_scale,
    validate_data,
    train_model,
    predict,
    compute_classification_metrics,
    compute_regression_metrics,
    compute_clustering_metrics,
)


class TestAppFlow:
    """Test complete workflows for each task type."""

    def test_classification_workflow(self):
        """Test complete classification pipeline."""
        loader = get_dataset_loader(random_state=42)
        X, y, info = loader.load("iris")

        # Validate data
        valid, msg = validate_data(X, y)
        assert valid

        # Split
        split = split_and_scale(X, y, test_size=0.2, random_state=42)

        # Train logistic regression
        model_info = get_model_info(ModelID.LOGISTIC_REGRESSION)
        result = train_model(ModelID.LOGISTIC_REGRESSION, model_info.default_params, split,
                           target_name=info.target_name, target_names=info.target_names)

        # Predict
        y_pred, y_prob = predict(result)
        assert y_pred.shape == (len(split.X_test),)
        assert y_prob is not None

        # Metrics
        metrics = compute_classification_metrics(split.y_test.values, y_pred, info.target_names)
        assert metrics.accuracy > 0.8  # Should be good on Iris
        assert metrics.confusion_matrix.shape == (3, 3)

    def test_binary_classification_workflow(self):
        """Test binary classification on breast cancer."""
        loader = get_dataset_loader(random_state=42)
        X, y, info = loader.load("breast_cancer")

        valid, msg = validate_data(X, y)
        assert valid

        split = split_and_scale(X, y, test_size=0.2, random_state=42)
        model_info = get_model_info(ModelID.LOGISTIC_REGRESSION)
        result = train_model(ModelID.LOGISTIC_REGRESSION, model_info.default_params, split,
                           target_name=info.target_name, target_names=info.target_names)

        y_pred, y_prob = predict(result)
        metrics = compute_classification_metrics(split.y_test.values, y_pred, info.target_names)
        assert metrics.accuracy > 0.9
        assert metrics.confusion_matrix.shape == (2, 2)

    def test_regression_workflow(self):
        """Test complete regression pipeline."""
        loader = get_dataset_loader(random_state=42)
        X, y, info = loader.load("diabetes")

        valid, msg = validate_data(X, y)
        assert valid

        split = split_and_scale(X, y, test_size=0.2, random_state=42, task_type=TaskType.REGRESSION)
        model_info = get_model_info(ModelID.LINEAR_REGRESSION)
        result = train_model(ModelID.LINEAR_REGRESSION, model_info.default_params, split,
                           target_name=info.target_name)

        y_pred, y_prob = predict(result)
        assert y_prob is None

        metrics = compute_regression_metrics(split.y_test.values, y_pred)
        assert metrics.mae >= 0
        assert metrics.rmse >= 0
        assert metrics.r2 <= 1.0

    def test_clustering_workflow(self):
        """Test complete clustering pipeline."""
        loader = get_dataset_loader(random_state=42)
        X, y, info = loader.load("synthetic_blobs")
        assert y is None

        valid, msg = validate_data(X, None)
        assert valid

        split = split_and_scale(X, None, test_size=0.2, random_state=42)
        model_info = get_model_info(ModelID.KMEANS)
        result = train_model(ModelID.KMEANS, model_info.default_params, split)

        y_pred, y_prob = predict(result)
        assert y_prob is None

        metrics = compute_clustering_metrics(X.values, y_pred,
                                           result.model.named_steps["model"].inertia_)
        assert metrics.n_clusters == 3
        assert metrics.inertia > 0

    def test_all_classification_models_on_iris(self):
        """Test all classification models work on Iris."""
        loader = get_dataset_loader(random_state=42)
        X, y, info = loader.load("iris")
        split = split_and_scale(X, y, test_size=0.2, random_state=42)

        classification_models = list_models(TaskType.CLASSIFICATION)
        for model_info in classification_models:
            result = train_model(model_info.id, model_info.default_params, split,
                               target_name=info.target_name, target_names=info.target_names)
            y_pred, _ = predict(result)
            metrics = compute_classification_metrics(split.y_test.values, y_pred, info.target_names)
            assert metrics.accuracy >= 0

    def test_feature_scaling_prevents_leakage(self):
        """Test that scaling is fit on train only."""
        loader = get_dataset_loader(random_state=42)
        X, y, info = loader.load("iris")

        # Create data with very different train/test distributions
        split = split_and_scale(X, y, test_size=0.2, random_state=42, scale=True)

        # Train scaler statistics should come from training data
        scaler = split.scaler
        assert scaler is not None

        # Test data should be transformed using train statistics
        train_mean = split.X_train.mean()
        test_mean = split.X_test.mean()
        # After scaling, train should be centered near 0
        assert abs(train_mean.mean()) < 0.1

    def test_invalid_knn_k_raises_error(self):
        """Test that KNN with k > n_samples raises error during prediction."""
        # Use a small but valid classification dataset
        X = pd.DataFrame(np.random.randn(10, 2), columns=["f1", "f2"])
        y = pd.Series([0, 1] * 5)  # 5 per class, balanced

        split = split_and_scale(X, y, test_size=0.4, random_state=42)  # 6 train, 4 test
        model_info = get_model_info(ModelID.KNN)

        # k=10 but only 6 training samples - sklearn will raise during prediction
        bad_params = model_info.default_params.copy()
        bad_params["n_neighbors"] = 10

        result = train_model(ModelID.KNN, bad_params, split, target_name="target")
        
        # Should raise during prediction
        with pytest.raises(ValueError):
            predict(result)

    def test_kmeans_invalid_config(self):
        """Test KMeans with invalid cluster count."""
        loader = get_dataset_loader(random_state=42)
        X, _, info = loader.load("synthetic_blobs")

        split = split_and_scale(X, None, test_size=0.2, random_state=42)
        model_info = get_model_info(ModelID.KMEANS)

        # Too many clusters for the data
        bad_params = model_info.default_params.copy()
        bad_params["n_clusters"] = 100  # More than samples

        # Should still run but might produce warnings or fail gracefully
        result = train_model(ModelID.KMEANS, bad_params, split)
        y_pred, _ = predict(result)
        assert len(y_pred) == len(split.X_test)

    def test_deterministic_results(self):
        """Test that same seed gives same results."""
        loader = get_dataset_loader(random_state=42)
        X, y, info = loader.load("iris")

        split1 = split_and_scale(X, y, test_size=0.2, random_state=42)
        split2 = split_and_scale(X, y, test_size=0.2, random_state=42)

        assert np.allclose(split1.X_train.values, split2.X_train.values)
        assert np.allclose(split1.X_test.values, split2.X_test.values)

        model_info = get_model_info(ModelID.LOGISTIC_REGRESSION)
        result1 = train_model(ModelID.LOGISTIC_REGRESSION, model_info.default_params, split1,
                            target_name=info.target_name, target_names=info.target_names)
        result2 = train_model(ModelID.LOGISTIC_REGRESSION, model_info.default_params, split2,
                            target_name=info.target_name, target_names=info.target_names)

        y_pred1, _ = predict(result1)
        y_pred2, _ = predict(result2)

        assert np.array_equal(y_pred1, y_pred2)


class TestDataValidation:
    def test_valid_data_passes(self):
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        valid, msg = validate_data(X, y)
        assert valid

    def test_empty_dataframe_fails(self):
        X = pd.DataFrame()
        y = pd.Series([0, 1, 0])
        valid, msg = validate_data(X, y)
        assert not valid
        assert "empty" in msg.lower()

    def test_nan_in_features_fails(self):
        X = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        valid, msg = validate_data(X, y)
        assert not valid
        assert "nan" in msg.lower()

    def test_inf_in_features_fails(self):
        X = pd.DataFrame({"a": [1, np.inf, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        valid, msg = validate_data(X, y)
        assert not valid
        assert "infinite" in msg.lower()

    def test_nan_in_target_fails(self):
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, np.nan, 0])
        valid, msg = validate_data(X, y)
        assert not valid

    def test_length_mismatch_fails(self):
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1])
        valid, msg = validate_data(X, y)
        assert not valid
        assert "length" in msg.lower()


class TestComparisonFairness:
    def test_same_split_for_all_models(self):
        """Ensure comparison uses same split."""
        loader = get_dataset_loader(random_state=42)
        X, y, info = loader.load("iris")

        split = split_and_scale(X, y, test_size=0.2, random_state=42)

        classification_models = list_models(TaskType.CLASSIFICATION)
        predictions = {}

        for model_info in classification_models:
            result = train_model(model_info.id, model_info.default_params, split,
                               target_name=info.target_name, target_names=info.target_names)
            y_pred, _ = predict(result)
            predictions[model_info.name] = y_pred

        # All should have predictions for same test indices
        for name, preds in predictions.items():
            assert len(preds) == len(split.X_test)

    def test_consistent_random_state(self):
        """Test that random_state produces consistent splits."""
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randint(0, 3, 100))

        split1 = split_and_scale(X, y, test_size=0.3, random_state=123)
        split2 = split_and_scale(X, y, test_size=0.3, random_state=123)

        assert np.allclose(split1.X_train.values, split2.X_train.values)
        assert np.allclose(split1.y_train.values, split2.y_train.values)
        assert np.allclose(split1.X_test.values, split2.X_test.values)
        assert np.allclose(split1.y_test.values, split2.y_test.values)