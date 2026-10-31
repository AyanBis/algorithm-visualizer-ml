"""ML Visualizer - Dataset loading and metadata."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes
from sklearn.datasets import make_blobs, make_regression


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"


@dataclass
class DatasetInfo:
    name: str
    task_type: TaskType
    description: str
    feature_names: list[str]
    target_name: Optional[str]
    n_samples: int
    n_features: int
    target_names: Optional[list[str]] = None


class DatasetLoader:
    """Load and manage built-in datasets for ML visualization."""

    _DATASETS = {
        "iris": DatasetInfo(
            name="Iris",
            task_type=TaskType.CLASSIFICATION,
            description="Classic dataset for flower species classification. 3 classes, 4 features.",
            feature_names=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
            target_name="species",
            n_samples=150,
            n_features=4,
            target_names=["setosa", "versicolor", "virginica"],
        ),
        "breast_cancer": DatasetInfo(
            name="Breast Cancer Wisconsin",
            task_type=TaskType.CLASSIFICATION,
            description="Breast cancer diagnosis dataset. Binary classification (malignant/benign).",
            feature_names=[
                "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
                "mean compactness", "mean concavity", "mean concave points", "mean symmetry",
                "mean fractal dimension", "radius error", "texture error", "perimeter error",
                "area error", "smoothness error", "compactness error", "concavity error",
                "concave points error", "symmetry error", "fractal dimension error",
                "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
                "worst compactness", "worst concavity", "worst concave points", "worst symmetry",
                "worst fractal dimension",
            ],
            target_name="diagnosis",
            n_samples=569,
            n_features=30,
            target_names=["malignant", "benign"],
        ),
        "diabetes": DatasetInfo(
            name="Diabetes",
            task_type=TaskType.REGRESSION,
            description="Diabetes progression dataset. Predict disease progression after one year.",
            feature_names=[
                "age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"
            ],
            target_name="progression",
            n_samples=442,
            n_features=10,
        ),
        "synthetic_regression": DatasetInfo(
            name="Synthetic Regression",
            task_type=TaskType.REGRESSION,
            description="Generated regression dataset with controllable noise and features.",
            feature_names=[f"feature_{i}" for i in range(5)],
            target_name="target",
            n_samples=200,
            n_features=5,
        ),
        "synthetic_blobs": DatasetInfo(
            name="Synthetic Blobs",
            task_type=TaskType.CLUSTERING,
            description="Generated blob clusters for clustering demonstration.",
            feature_names=["feature_1", "feature_2"],
            target_name=None,
            n_samples=300,
            n_features=2,
        ),
    }

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._cache = {}

    def list_datasets(self) -> list[str]:
        return list(self._DATASETS.keys())

    def get_info(self, name: str) -> DatasetInfo:
        if name not in self._DATASETS:
            raise ValueError(f"Unknown dataset: {name}. Available: {self.list_datasets()}")
        return self._DATASETS[name]

    def load(self, name: str) -> tuple[pd.DataFrame, Optional[pd.Series], DatasetInfo]:
        """Load dataset and return (features, target, info). Target is None for clustering."""
        if name in self._cache:
            return self._cache[name]

        info = self.get_info(name)

        if name == "iris":
            data = load_iris()
            X = pd.DataFrame(data.data, columns=info.feature_names)
            y = pd.Series(data.target, name=info.target_name)
        elif name == "breast_cancer":
            data = load_breast_cancer()
            X = pd.DataFrame(data.data, columns=info.feature_names)
            y = pd.Series(data.target, name=info.target_name)
        elif name == "diabetes":
            data = load_diabetes()
            X = pd.DataFrame(data.data, columns=info.feature_names)
            y = pd.Series(data.target, name=info.target_name)
        elif name == "synthetic_regression":
            X_data, y_data = make_regression(
                n_samples=info.n_samples,
                n_features=info.n_features,
                noise=10.0,
                random_state=self.random_state,
            )
            X = pd.DataFrame(X_data, columns=info.feature_names)
            y = pd.Series(y_data, name=info.target_name)
        elif name == "synthetic_blobs":
            X_data, _ = make_blobs(
                n_samples=info.n_samples,
                centers=4,
                n_features=info.n_features,
                random_state=self.random_state,
            )
            X = pd.DataFrame(X_data, columns=info.feature_names)
            y = None
        else:
            raise ValueError(f"Unknown dataset: {name}")

        result = (X, y, info)
        self._cache[name] = result
        return result

    def get_preview(self, name: str, n_rows: int = 10) -> pd.DataFrame:
        X, y, info = self.load(name)
        if y is not None:
            preview = X.head(n_rows).copy()
            if info.target_names is not None:
                # Map numeric labels to human-readable class names
                preview[info.target_name] = [info.target_names[int(v)] for v in y.head(n_rows).values]
            else:
                preview[info.target_name] = y.head(n_rows).values
        else:
            preview = X.head(n_rows).copy()
        return preview


def get_dataset_loader(random_state: int = 42) -> DatasetLoader:
    return DatasetLoader(random_state)