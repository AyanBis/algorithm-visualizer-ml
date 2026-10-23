"""ML Visualizer - Model registry with definitions and hyperparameters."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .datasets import TaskType


class ModelID(Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    KNN = "knn"
    SVM = "svm"
    LINEAR_REGRESSION = "linear_regression"
    KMEANS = "kmeans"


@dataclass
class HyperParameter:
    name: str
    display_name: str
    param_type: str  # "int", "float", "select"
    default: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    options: Optional[list[Any]] = None
    help_text: str = ""


@dataclass
class ModelInfo:
    id: ModelID
    name: str
    task_type: TaskType
    description: str
    how_it_works: str
    when_to_use: str
    limitations: str
    hyperparameters: list[HyperParameter] = field(default_factory=list)
    supports_probability: bool = False
    supports_decision_boundary: bool = False
    supports_feature_importance: bool = False
    default_params: dict = field(default_factory=dict)


def _get_logistic_regression_info() -> ModelInfo:
    return ModelInfo(
        id=ModelID.LOGISTIC_REGRESSION,
        name="Logistic Regression",
        task_type=TaskType.CLASSIFICATION,
        description="A linear model for classification that estimates probabilities using the logistic (sigmoid) function.",
        how_it_works="Logistic Regression learns a linear decision boundary by modeling the log-odds of the positive class as a linear combination of features. The sigmoid function converts this to a probability between 0 and 1.",
        when_to_use="Use when you need interpretable probabilities, the relationship is roughly linear, or as a baseline classifier.",
        limitations="Assumes linear decision boundary. May underfit complex patterns. Sensitive to outliers and multicollinearity.",
        hyperparameters=[
            HyperParameter("C", "Regularization Strength (C)", "float", 1.0, 0.01, 100.0, 0.01,
                          help_text="Inverse of regularization strength. Smaller values = stronger regularization."),
            HyperParameter("max_iter", "Max Iterations", "int", 200, 50, 1000, 50,
                          help_text="Maximum iterations for solver convergence."),
            HyperParameter("solver", "Solver", "select", "lbfgs", options=["lbfgs", "liblinear", "saga"],
                          help_text="Optimization algorithm. 'liblinear' for small datasets, 'saga' for L1 penalty."),
        ],
        supports_probability=True,
        supports_decision_boundary=True,
        supports_feature_importance=True,
        default_params={"C": 1.0, "max_iter": 200, "solver": "lbfgs", "random_state": 42},
    )


def _get_decision_tree_info() -> ModelInfo:
    return ModelInfo(
        id=ModelID.DECISION_TREE,
        name="Decision Tree Classifier",
        task_type=TaskType.CLASSIFICATION,
        description="A tree-based model that splits data recursively based on feature thresholds.",
        how_it_works="The tree recursively partitions the feature space by selecting the best feature and threshold to split on (using Gini impurity or entropy). Each leaf node predicts the majority class.",
        when_to_use="Use for interpretable models, non-linear boundaries, mixed feature types, or when you need feature importance.",
        limitations="Prone to overfitting (especially deep trees). Unstable - small data changes can create very different trees. Biased toward features with more levels.",
        hyperparameters=[
            HyperParameter("max_depth", "Max Depth", "int", 5, 1, 20, 1,
                          help_text="Maximum depth of the tree. Limits overfitting."),
            HyperParameter("min_samples_split", "Min Samples Split", "int", 5, 2, 50, 1,
                          help_text="Minimum samples required to split an internal node."),
            HyperParameter("min_samples_leaf", "Min Samples Leaf", "int", 2, 1, 20, 1,
                          help_text="Minimum samples required at a leaf node."),
            HyperParameter("criterion", "Split Criterion", "select", "gini", options=["gini", "entropy"],
                          help_text="Function to measure split quality."),
        ],
        supports_probability=True,
        supports_decision_boundary=True,
        supports_feature_importance=True,
        default_params={"max_depth": 5, "min_samples_split": 5, "min_samples_leaf": 2,
                       "criterion": "gini", "random_state": 42},
    )


def _get_random_forest_info() -> ModelInfo:
    return ModelInfo(
        id=ModelID.RANDOM_FOREST,
        name="Random Forest Classifier",
        task_type=TaskType.CLASSIFICATION,
        description="An ensemble of decision trees that combines predictions via majority voting.",
        how_it_works="Random Forest builds many decision trees on bootstrapped samples of data. Each tree sees a random subset of features at each split. Final prediction is majority vote (classification) or average (regression).",
        when_to_use="Use for robust, accurate classification with minimal tuning. Handles non-linear patterns and feature interactions well.",
        limitations="Less interpretable than single trees. Can be slow with many trees. Doesn't extrapolate well. Memory intensive for large forests.",
        hyperparameters=[
            HyperParameter("n_estimators", "Number of Trees", "int", 100, 10, 500, 10,
                          help_text="Number of trees in the forest. More trees = better but slower."),
            HyperParameter("max_depth", "Max Depth", "int", 10, 1, 30, 1,
                          help_text="Maximum depth of each tree. None = fully grown."),
            HyperParameter("min_samples_split", "Min Samples Split", "int", 5, 2, 50, 1,
                          help_text="Minimum samples required to split an internal node."),
            HyperParameter("max_features", "Max Features per Split", "select", "sqrt",
                          options=["sqrt", "log2", None],
                          help_text="Number of features to consider at each split."),
        ],
        supports_probability=True,
        supports_decision_boundary=True,
        supports_feature_importance=True,
        default_params={"n_estimators": 100, "max_depth": 10, "min_samples_split": 5,
                       "max_features": "sqrt", "random_state": 42, "n_jobs": -1},
    )


def _get_knn_info() -> ModelInfo:
    return ModelInfo(
        id=ModelID.KNN,
        name="K-Nearest Neighbors Classifier",
        task_type=TaskType.CLASSIFICATION,
        description="Instance-based classifier that predicts based on the majority class of k nearest neighbors.",
        how_it_works="KNN stores all training data. For prediction, it finds the k closest training points (by distance metric) and assigns the majority class. Distance weighting gives closer neighbors more influence.",
        when_to_use="Use for simple baselines, when decision boundary is complex but local, or when you have enough data for nearest neighbors to be meaningful.",
        limitations="Slow at prediction time (scales with training data). Sensitive to feature scaling. Curse of dimensionality - degrades with many features. No explicit model learned.",
        hyperparameters=[
            HyperParameter("n_neighbors", "Number of Neighbors (k)", "int", 5, 1, 50, 1,
                          help_text="Number of neighbors to consider. Small k = noisy boundary, large k = smooth boundary."),
            HyperParameter("weights", "Weight Function", "select", "uniform", options=["uniform", "distance"],
                          help_text="'uniform' = equal weight, 'distance' = closer neighbors weighted more."),
            HyperParameter("metric", "Distance Metric", "select", "euclidean", options=["euclidean", "manhattan", "minkowski"],
                          help_text="Distance metric for finding neighbors."),
        ],
        supports_probability=True,
        supports_decision_boundary=True,
        supports_feature_importance=False,
        default_params={"n_neighbors": 5, "weights": "uniform", "metric": "euclidean"},
    )


def _get_svm_info() -> ModelInfo:
    return ModelInfo(
        id=ModelID.SVM,
        name="Support Vector Machine",
        task_type=TaskType.CLASSIFICATION,
        description="Finds the optimal separating hyperplane with maximum margin between classes.",
        how_it_works="SVM finds a hyperplane that maximizes the margin between classes. Kernels (linear, RBF, etc.) map data to higher dimensions where linear separation is possible. Support vectors are the critical points defining the boundary.",
        when_to_use="Use for high-dimensional data, clear margin separation, or when you need strong theoretical guarantees. RBF kernel for non-linear boundaries.",
        limitations="Sensitive to feature scaling. Slow on large datasets. No direct probability estimates (need calibration). Hard to interpret. RBF kernel has many hyperparameters.",
        hyperparameters=[
            HyperParameter("C", "Regularization (C)", "float", 1.0, 0.01, 100.0, 0.01,
                          help_text="Regularization parameter. High C = smaller margin, fewer misclassifications."),
            HyperParameter("kernel", "Kernel", "select", "rbf", options=["linear", "rbf", "poly", "sigmoid"],
                          help_text="Kernel type. 'linear' for linear boundary, 'rbf' for non-linear."),
            HyperParameter("gamma", "Gamma (RBF)", "float", 0.1, 0.001, 10.0, 0.01,
                          help_text="RBF kernel coefficient. High gamma = tighter fit, more overfitting."),
        ],
        supports_probability=True,
        supports_decision_boundary=True,
        supports_feature_importance=False,
        default_params={"C": 1.0, "kernel": "rbf", "gamma": "scale", "probability": True, "random_state": 42},
    )


def _get_linear_regression_info() -> ModelInfo:
    return ModelInfo(
        id=ModelID.LINEAR_REGRESSION,
        name="Linear Regression",
        task_type=TaskType.REGRESSION,
        description="Models the relationship between features and a continuous target using a linear equation.",
        how_it_works="Linear Regression finds coefficients that minimize the sum of squared residuals (OLS). The model assumes y = β₀ + β₁x₁ + ... + βₙxₙ + ε.",
        when_to_use="Use for interpretable regression, linear relationships, or as a baseline. Coefficients show feature impact direction and magnitude.",
        limitations="Assumes linear relationship. Sensitive to outliers. Assumes homoscedasticity and independence of errors. Can't capture complex non-linear patterns.",
        hyperparameters=[
            HyperParameter("fit_intercept", "Fit Intercept", "select", True, options=[True, False],
                          help_text="Whether to calculate the intercept (bias term)."),
        ],
        supports_probability=False,
        supports_decision_boundary=False,
        supports_feature_importance=True,
        default_params={"fit_intercept": True},
    )


def _get_kmeans_info() -> ModelInfo:
    return ModelInfo(
        id=ModelID.KMEANS,
        name="K-Means Clustering",
        task_type=TaskType.CLUSTERING,
        description="Partitions data into k clusters by minimizing within-cluster sum of squares (inertia).",
        how_it_works="K-Means iteratively assigns points to the nearest centroid and updates centroids as cluster means. It converges when assignments stop changing. Multiple random initializations avoid local minima.",
        when_to_use="Use for exploratory clustering, customer segmentation, or when you need fast, scalable clustering with spherical clusters.",
        limitations="Assumes spherical, equally-sized clusters. Requires specifying k. Sensitive to initialization and outliers. Doesn't handle varying densities or non-convex shapes. Cluster labels are arbitrary.",
        hyperparameters=[
            HyperParameter("n_clusters", "Number of Clusters (k)", "int", 3, 2, 10, 1,
                          help_text="Number of clusters to form. Use elbow method or silhouette to choose."),
            HyperParameter("n_init", "Number of Initializations", "int", 10, 1, 50, 1,
                          help_text="Number of random initializations. Best result is kept."),
            HyperParameter("max_iter", "Max Iterations", "int", 300, 50, 1000, 50,
                          help_text="Maximum iterations per initialization."),
        ],
        supports_probability=False,
        supports_decision_boundary=False,
        supports_feature_importance=False,
        default_params={"n_clusters": 3, "n_init": 10, "max_iter": 300, "random_state": 42},
    )


MODEL_REGISTRY: dict[ModelID, ModelInfo] = {
    ModelID.LOGISTIC_REGRESSION: _get_logistic_regression_info(),
    ModelID.DECISION_TREE: _get_decision_tree_info(),
    ModelID.RANDOM_FOREST: _get_random_forest_info(),
    ModelID.KNN: _get_knn_info(),
    ModelID.SVM: _get_svm_info(),
    ModelID.LINEAR_REGRESSION: _get_linear_regression_info(),
    ModelID.KMEANS: _get_kmeans_info(),
}


def get_model_info(model_id: ModelID) -> ModelInfo:
    return MODEL_REGISTRY[model_id]


def list_models(task_type: Optional[TaskType] = None) -> list[ModelInfo]:
    if task_type is None:
        return list(MODEL_REGISTRY.values())
    return [info for info in MODEL_REGISTRY.values() if info.task_type == task_type]


def create_model(model_id: ModelID, params: dict) -> Any:
    """Create a model instance with given parameters."""
    base_params = MODEL_REGISTRY[model_id].default_params.copy()
    base_params.update(params)

    if model_id == ModelID.LOGISTIC_REGRESSION:
        return LogisticRegression(**base_params)
    elif model_id == ModelID.DECISION_TREE:
        return DecisionTreeClassifier(**base_params)
    elif model_id == ModelID.RANDOM_FOREST:
        return RandomForestClassifier(**base_params)
    elif model_id == ModelID.KNN:
        return KNeighborsClassifier(**base_params)
    elif model_id == ModelID.SVM:
        return SVC(**base_params)
    elif model_id == ModelID.LINEAR_REGRESSION:
        return LinearRegression(**base_params)
    elif model_id == ModelID.KMEANS:
        return KMeans(**base_params)
    else:
        raise ValueError(f"Unknown model: {model_id}")


def create_pipeline(model_id: ModelID, params: dict, scale: bool = True) -> Pipeline:
    """Create a pipeline with optional scaling."""
    model = create_model(model_id, params)
    if scale and model_id != ModelID.KMEANS:
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    return Pipeline([("model", model)])