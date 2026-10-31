"""Tests for API endpoints - synchronous version."""
import pytest
from fastapi.testclient import TestClient
from backend.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "algorithm-visualizer-ml-api"


def test_list_datasets():
    response = client.get("/api/datasets")
    assert response.status_code == 200
    data = response.json()
    assert "datasets" in data
    assert len(data["datasets"]) == 5
    dataset_ids = {d["id"] for d in data["datasets"]}
    assert dataset_ids == {"iris", "breast_cancer", "diabetes", "synthetic_regression", "synthetic_blobs"}


def test_get_dataset_detail():
    response = client.get("/api/datasets/iris")
    assert response.status_code == 200
    data = response.json()
    assert data["info"]["id"] == "iris"
    assert data["info"]["task_type"] == "classification"
    assert data["info"]["n_samples"] == 150
    assert data["info"]["n_features"] == 4
    assert "preview" in data
    assert "statistics" in data
    assert "target_distribution" in data


def test_get_dataset_not_found():
    response = client.get("/api/datasets/nonexistent")
    assert response.status_code == 404


def test_list_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) == 7


def test_list_models_by_task():
    response = client.get("/api/models?task_type=classification")
    assert response.status_code == 200
    data = response.json()
    assert len(data["models"]) == 5
    for model in data["models"]:
        assert model["task_type"] == "classification"

    response = client.get("/api/models?task_type=regression")
    assert response.status_code == 200
    data = response.json()
    assert len(data["models"]) == 1
    assert data["models"][0]["task_type"] == "regression"

    response = client.get("/api/models?task_type=clustering")
    assert response.status_code == 200
    data = response.json()
    assert len(data["models"]) == 1
    assert data["models"][0]["task_type"] == "clustering"


def test_get_model_detail():
    response = client.get("/api/models/logistic_regression")
    assert response.status_code == 200
    data = response.json()
    assert data["model"]["id"] == "logistic_regression"
    assert data["model"]["name"] == "Logistic Regression"
    assert len(data["model"]["hyperparameters"]) == 3


def test_get_model_not_found():
    response = client.get("/api/models/nonexistent")
    assert response.status_code == 404