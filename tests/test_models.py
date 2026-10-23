"""Tests for model registry, training, and predictions."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, make_blobs

from ml_visualizer.model_registry import (
    ModelID,
    ModelInfo,
    HyperParameter,
    get_model_info,
    list_models,
    create_model,
    create_pipeline,
    MODEL_REGISTRY,
    TaskType,
)
from ml_visualizer.training import train_model, predict, get_feature_importance
from ml_visualizer.preprocessing import split_and_scale
from ml_visualizer.datasets import get_dataset_loader


class TestModelRegistry:
    def test_all_models_registered(self):
        expected_models = [
            ModelID.LOGISTIC_REGRESSION,
            ModelID.DECISION_TREE,
            ModelID.RANDOM_FOREST,
            ModelID.KNN,
            ModelID.SVM,
            ModelID.LINEAR_REGRESSION,
            ModelID.KMEANS,
        ]
        for model_id in expected_models:
            assert model_id in MODEL_REGISTRY

    def test_get_model_info(self):
        info = get_model_info(ModelID.LOGISTIC_REGRESSION)
        assert isinstance(info, ModelInfo)
        assert info.id == ModelID.LOGISTIC_REGRESSION
        assert info.task_type == TaskType.CLASSIFICATION
        assert len(info.hyperparameters) > 0

    def test_list_models_by_task(self):
        classification_models = list_models(TaskType.CLASSIFICATION)
        assert len(classification_models) == 5

        regression_models = list_models(TaskType.REGRESSION)
        assert len(regression_models) == 1

        clustering_models = list_models(TaskType.CLUSTERING)
        assert len(clustering_models) == 1

    def test_create_model(self):
        for model_id in ModelID:
            info = get_model_info(model_id)
            model = create_model(model_id, info.default_params)
            assert model is not None

    def test_create_pipeline(self):
        for model_id in ModelID:
            info = get_model_info(model_id)
            pipeline = create_pipeline(model_id, info.default_params, scale=True)
            assert pipeline is not None


class TestModelTraining:
    @pytest.fixture
    def classification_data(self):
        X, y = make_classification(n_samples=100, n_features=5, n_classes=3,
                                  n_informative=3, n_redundant=1, random_state=42)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]), pd.Series(y, name="target")

    @pytest.fixture
    def regression_data(self):
        X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]), pd.Series(y, name="target")

    @pytest.fixture
    def clustering_data(self):
        X, _ = make_blobs(n_samples=100, n_features=3, centers=3, random_state=42)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])

    def test_logistic_regression_train(self, classification_data):
        X, y = classification_data
        split = split_and_scale(X, y, test_size=0.2, random_state=42)
        info = get_model_info(ModelID.LOGISTIC_REGRESSION)
        result = train_model(ModelID.LOGISTIC_REGRESSION, info.default_params, split,
                           target_name="target", target_names=["0", "1", "2"])
        assert result.model is not None
        assert result.model_id == ModelID.LOGISTIC_REGRESSION

        y_pred, y_prob = predict(result)
        assert y_pred.shape == (len(split.X_test),)
        assert y_prob is not None
        assert y_prob.shape[1] == 3

    def test_decision_tree_train(self, classification_data):
        X, y = classification_data
        split = split_and_scale(X, y, test_size=0.2, random_state=42)
        info = get_model_info(ModelID.DECISION_TREE)
        result = train_model(ModelID.DECISION_TREE, info.default_params, split,
                           target_name="target")
        y_pred, y_prob = predict(result)
        assert y_pred.shape == (len(split.X_test),)
        assert y_prob is not None

    def test_random_forest_train(self, classification_data):
        X, y = classification_data
        split = split_and_scale(X, y, test_size=0.2, random_state=42)
        info = get_model_info(ModelID.RANDOM_FOREST)
        result = train_model(ModelID.RANDOM_FOREST, info.default_params, split,
                           target_name="target")
        y_pred, y_prob = predict(result)
        assert y_pred.shape == (len(split.X_test),)

    def test_knn_train(self, classification_data):
        X, y = classification_data
        split = split_and_scale(X, y, test_size=0.2, random_state=42)
        info = get_model_info(ModelID.KNN)
        result = train_model(ModelID.KNN, info.default_params, split,
                           target_name="target")
        y_pred, y_prob = predict(result)
        assert y_pred.shape == (len(split.X_test),)

    def test_svm_train(self, classification_data):
        X, y = classification_data
        split = split_and_scale(X, y, test_size=0.2, random_state=42)
        info = get_model_info(ModelID.SVM)
        result = train_model(ModelID.SVM, info.default_params, split,
                           target_name="target")
        y_pred, y_prob = predict(result)
        assert y_pred.shape == (len(split.X_test),)

    def test_linear_regression_train(self, regression_data):
        X, y = regression_data
        split = split_and_scale(X, y, test_size=0.2, random_state=42, task_type=TaskType.REGRESSION)
        info = get_model_info(ModelID.LINEAR_REGRESSION)
        result = train_model(ModelID.LINEAR_REGRESSION, info.default_params, split,
                           target_name="target")
        y_pred, y_prob = predict(result)
        assert y_pred.shape == (len(split.X_test),)
        assert y_prob is None

    def test_kmeans_train(self, clustering_data):
        X = clustering_data
        split = split_and_scale(X, None, test_size=0.2, random_state=42)
        info = get_model_info(ModelID.KMEANS)
        result = train_model(ModelID.KMEANS, info.default_params, split)
        y_pred, y_prob = predict(result)
        assert y_pred.shape == (len(split.X_test),)
        assert y_prob is None

    def test_feature_importance_classification(self, classification_data):
        X, y = classification_data
        split = split_and_scale(X, y, test_size=0.2, random_state=42)
        info = get_model_info(ModelID.LOGISTIC_REGRESSION)
        result = train_model(ModelID.LOGISTIC_REGRESSION, info.default_params, split,
                           target_name="target")
        importance = get_feature_importance(result)
        assert importance is not None
        assert len(importance) == 5

    def test_feature_importance_tree(self, classification_data):
        X, y = classification_data
        split = split_and_scale(X, y, test_size=0.2, random_state=42)
        info = get_model_info(ModelID.DECISION_TREE)
        result = train_model(ModelID.DECISION_TREE, info.default_params, split,
                           target_name="target")
        importance = get_feature_importance(result)
        assert importance is not None
        assert len(importance) == 5

    def test_feature_importance_none_for_knn(self, classification_data):
        X, y = classification_data
        split = split_and_scale(X, y, test_size=0.2, random_state=42)
        info = get_model_info(ModelID.KNN)
        result = train_model(ModelID.KNN, info.default_params, split,
                           target_name="target")
        importance = get_feature_importance(result)
        assert importance is None

    def test_feature_importance_linear_regression(self, regression_data):
        X, y = regression_data
        split = split_and_scale(X, y, test_size=0.2, random_state=42, task_type=TaskType.REGRESSION)
        info = get_model_info(ModelID.LINEAR_REGRESSION)
        result = train_model(ModelID.LINEAR_REGRESSION, info.default_params, split,
                           target_name="target")
        importance = get_feature_importance(result)
        assert importance is not None
        assert len(importance) == 5

    def test_kmeans_no_feature_importance(self, clustering_data):
        X = clustering_data
        split = split_and_scale(X, None, test_size=0.2, random_state=42)
        info = get_model_info(ModelID.KMEANS)
        result = train_model(ModelID.KMEANS, info.default_params, split)
        importance = get_feature_importance(result)
        assert importance is None

    def test_svm_hyperparameters_passed_correctly(self, classification_data):
        """Test that SVM hyperparameters (C, kernel, gamma) are passed to underlying SVC."""
        X, y = classification_data
        split = split_and_scale(X, y, test_size=0.2, random_state=42)

        # Test with custom hyperparameters
        custom_params = {"C": 10.0, "kernel": "linear", "gamma": "scale"}
        result = train_model(ModelID.SVM, custom_params, split, target_name="target")

        # Check the underlying SVC estimator through CalibratedClassifierCV
        calibrated = result.model.named_steps["model"]
        svc = calibrated.estimator

        assert svc.C == 10.0
        assert svc.kernel == "linear"
        assert svc.gamma == "scale"

        # Test with RBF kernel and numeric gamma
        custom_params_rbf = {"C": 0.5, "kernel": "rbf", "gamma": 0.1}
        result_rbf = train_model(ModelID.SVM, custom_params_rbf, split, target_name="target")
        calibrated_rbf = result_rbf.model.named_steps["model"]
        svc_rbf = calibrated_rbf.estimator

        assert svc_rbf.C == 0.5
        assert svc_rbf.kernel == "rbf"
        assert svc_rbf.gamma == 0.1

    def test_svm_probability_calibration_works(self, classification_data):
        """Test that SVM probability calibration works without deprecated probability=True."""
        X, y = classification_data
        split = split_and_scale(X, y, test_size=0.2, random_state=42)

        result = train_model(ModelID.SVM, get_model_info(ModelID.SVM).default_params, split,
                           target_name="target")

        y_pred, y_prob = predict(result)

        # Should have probability estimates from calibration
        assert y_prob is not None
        assert y_prob.shape == (len(split.X_test), 3)
        # Probabilities should sum to 1
        np.testing.assert_allclose(y_prob.sum(axis=1), 1.0, rtol=1e-5)
        # All probabilities should be in [0, 1]
        assert (y_prob >= 0).all() and (y_prob <= 1).all()

    def test_svm_no_deprecated_probability_param(self, classification_data):
        """Test that SVC is not created with deprecated probability=True."""
        X, y = classification_data
        split = split_and_scale(X, y, test_size=0.2, random_state=42)

        result = train_model(ModelID.SVM, get_model_info(ModelID.SVM).default_params, split,
                           target_name="target")

        calibrated = result.model.named_steps["model"]
        svc = calibrated.estimator

        # The probability parameter should not be True (default is False)
        # In sklearn 1.9+, it shows as 'deprecated' in get_params but the value is False
        params = svc.get_params()
        assert params.get("probability") != True  # Should not be True