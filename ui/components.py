"""Streamlit UI components for ML Visualizer."""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import time

from ml_visualizer import (
    DatasetLoader,
    DatasetInfo,
    TaskType,
    ModelID,
    ModelInfo,
    HyperParameter,
    get_dataset_loader,
    get_model_info,
    list_models,
    split_and_scale,
    validate_data,
    train_model,
    predict,
    get_feature_importance,
    get_decision_boundary_data,
    compute_classification_metrics,
    compute_regression_metrics,
    compute_clustering_metrics,
    display_figure,
    plot_dataset_overview,
    plot_correlation_matrix,
    plot_confusion_matrix,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_regression_line,
    plot_feature_importance,
    plot_clusters,
    plot_silhouette,
    plot_decision_boundary,
    create_comparison_table,
    plot_model_comparison_bar,
    plot_metric_radar,
)


def render_header() -> None:
    """Render the main header."""
    st.markdown("""
    # 🤖 Algorithm Visualizer ML
    ### Interactive Machine Learning Model Explorer

    Learn how ML algorithms work by exploring datasets, training models, and visualizing results.
    """)
    st.divider()


def render_overview() -> None:
    """Render the overview page."""
    st.markdown("""
    ## Welcome to Algorithm Visualizer ML

    This application helps you understand **how machine learning algorithms work** through interactive exploration.

    ### 🎯 What You Can Do
    - **Explore Datasets** — Inspect features, targets, and data distributions
    - **Train Models** — Configure and train 7 different ML algorithms
    - **Visualize Results** — See decision boundaries, confusion matrices, regression plots, and clusters
    - **Compare Models** — Fairly compare multiple algorithms on the same data

    ### 📚 Supported Algorithms

    **Classification** (predicting categories):
    - **Logistic Regression** — Linear decision boundary with probabilities
    - **Decision Tree** — Interpretable rule-based splits
    - **Random Forest** — Ensemble of trees with voting
    - **K-Nearest Neighbors** — Instance-based classification
    - **Support Vector Machine** — Maximum margin separator

    **Regression** (predicting continuous values):
    - **Linear Regression** — Linear relationship modeling

    **Clustering** (unsupervised grouping):
    - **K-Means** — Centroid-based partitioning

    ### 🔬 Learning Concepts
    - **Supervised Learning** — Models learn from labeled examples (classification, regression)
    - **Unsupervised Learning** — Models find patterns without labels (clustering)
    - **Train/Test Split** — Evaluate on unseen data to measure generalization
    - **Feature Scaling** — Normalize features for fair distance-based algorithms
    - **Cross-Validation** — (Coming soon) Robust performance estimation

    ### 🚀 Quick Start
    1. Select **Dataset Explorer** from the sidebar
    2. Choose a dataset and explore its structure
    3. Go to **Model Explorer** and pick an algorithm
    4. Adjust hyperparameters and click **Train**
    5. Examine metrics, predictions, and visualizations
    6. Try **Model Comparison** to see multiple algorithms side-by-side
    """)


def render_dataset_explorer(loader: DatasetLoader) -> Tuple[str, DatasetInfo, pd.DataFrame, Optional[pd.Series]]:
    """Render dataset exploration page. Returns selected dataset name, info, X, y."""
    st.header("📊 Dataset Explorer")

    dataset_names = loader.list_datasets()
    dataset_options = {loader.get_info(name).name: name for name in dataset_names}

    selected_display = st.selectbox(
        "Select Dataset",
        options=list(dataset_options.keys()),
        index=0,
        help="Choose a built-in dataset to explore"
    )
    selected_name = dataset_options[selected_display]

    info = loader.get_info(selected_name)
    X, y, _ = loader.load(selected_name)

    # Dataset info cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Task Type", info.task_type.value.title())
    with col2:
        st.metric("Samples", info.n_samples)
    with col3:
        st.metric("Features", info.n_features)
    with col4:
        st.metric("Classes" if info.task_type == TaskType.CLASSIFICATION else "Target", 
                 len(info.target_names) if info.target_names else "Continuous")

    st.markdown(f"**Description:** {info.description}")

    # Data preview
    st.subheader("📋 Data Preview")
    preview = loader.get_preview(selected_name, n_rows=10)
    st.dataframe(preview, use_container_width=True)

    # Feature details
    with st.expander("🔍 Feature Details", expanded=False):
        feature_df = pd.DataFrame({
            "Feature": info.feature_names,
            "Type": ["numeric"] * len(info.feature_names),
            "Non-Null": [X[col].notna().sum() for col in info.feature_names],
            "Mean": [f"{X[col].mean():.3f}" for col in info.feature_names],
            "Std": [f"{X[col].std():.3f}" for col in info.feature_names],
            "Min": [f"{X[col].min():.3f}" for col in info.feature_names],
            "Max": [f"{X[col].max():.3f}" for col in info.feature_names],
        })
        st.dataframe(feature_df, use_container_width=True)

    # Visualizations
    st.subheader("📈 Data Visualizations")
    viz_tab1, viz_tab2 = st.tabs(["Distributions", "Correlations"])

    with viz_tab1:
        fig = plot_dataset_overview(X, y, info.target_name)
        display_figure(fig, "Feature Distributions by Class" if y is not None else "Feature Distributions")

    with viz_tab2:
        if len(X.columns) > 1:
            fig = plot_correlation_matrix(X)
            display_figure(fig, "Feature Correlation Matrix")
        else:
            st.info("Correlation matrix requires at least 2 features.")

    # Target info
    if y is not None and info.target_names:
        st.subheader("🎯 Target Distribution")
        target_counts = y.value_counts().sort_index()
        target_df = pd.DataFrame({
            "Class": [info.target_names[i] for i in target_counts.index],
            "Count": target_counts.values,
            "Percentage": [f"{v/len(y)*100:.1f}%" for v in target_counts.values]
        })
        st.dataframe(target_df, use_container_width=True)

    return selected_name, info, X, y


def render_model_explorer(
    loader: DatasetLoader,
    dataset_name: str,
    info: DatasetInfo,
    X: pd.DataFrame,
    y: Optional[pd.Series],
) -> None:
    """Render model training and evaluation page."""
    st.header("🤖 Model Explorer")

    # Model selection
    compatible_models = list_models(info.task_type)
    model_options = {m.name: m.id for m in compatible_models}

    selected_model_name = st.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        index=0,
        help="Choose a model compatible with the dataset task type"
    )
    model_id = model_options[selected_model_name]
    model_info = get_model_info(model_id)

    # Model description
    with st.expander("ℹ️ About This Model", expanded=True):
        st.markdown(f"**Description:** {model_info.description}")
        st.markdown(f"**How It Works:** {model_info.how_it_works}")
        st.markdown(f"**When to Use:** {model_info.when_to_use}")
        st.markdown(f"**Limitations:** {model_info.limitations}")

    # Hyperparameter controls
    st.subheader("⚙️ Hyperparameters")
    params = {}
    cols = st.columns(2)

    for i, hp in enumerate(model_info.hyperparameters):
        col = cols[i % 2]
        with col:
            if hp.param_type == "int":
                params[hp.name] = st.number_input(
                    hp.display_name,
                    min_value=int(hp.min_value) if hp.min_value else 0,
                    max_value=int(hp.max_value) if hp.max_value else 1000,
                    value=int(hp.default),
                    step=int(hp.step) if hp.step else 1,
                    help=hp.help_text,
                )
            elif hp.param_type == "float":
                params[hp.name] = st.number_input(
                    hp.display_name,
                    min_value=float(hp.min_value) if hp.min_value else 0.0,
                    max_value=float(hp.max_value) if hp.max_value else 100.0,
                    value=float(hp.default),
                    step=float(hp.step) if hp.step else 0.01,
                    format="%.3f",
                    help=hp.help_text,
                )
            elif hp.param_type == "select":
                params[hp.name] = st.selectbox(
                    hp.display_name,
                    options=hp.options,
                    index=hp.options.index(hp.default) if hp.default in hp.options else 0,
                    help=hp.help_text,
                )

    # Training controls
    st.subheader("🎯 Training Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05, help="Fraction of data for testing")
    with col2:
        random_state = st.number_input("Random Seed", 0, 10000, 42, help="For reproducibility")
    with col3:
        scale_features = st.checkbox("Scale Features", value=True, help="Standardize features (recommended)")

    # Feature selection for visualization
    if len(X.columns) >= 2:
        st.subheader("🔍 Visualization Features")
        feature_options = list(X.columns)
        default_idx = (0, 1) if len(feature_options) >= 2 else (0, 0)
        feat_col1, feat_col2 = st.columns(2)
        with feat_col1:
            viz_feature_1 = st.selectbox("Feature X", feature_options, index=default_idx[0])
        with feat_col2:
            viz_feature_2 = st.selectbox("Feature Y", feature_options, index=default_idx[1])
        viz_features = (feature_options.index(viz_feature_1), feature_options.index(viz_feature_2))
    else:
        viz_features = None

    # Train button
    train_clicked = st.button("🚀 Train Model", type="primary", use_container_width=True)

    if train_clicked:
        with st.spinner("Training model..."):
            try:
                # Validate data
                valid, msg = validate_data(X, y)
                if not valid:
                    st.error(f"Data validation failed: {msg}")
                    return

                # Split and scale
                split_result = split_and_scale(
                    X, y, test_size=test_size, random_state=random_state, scale=scale_features,
                    task_type=info.task_type
                )

                # Train
                training_result = train_model(
                    model_id, params, split_result,
                    target_name=info.target_name,
                    target_names=info.target_names,
                )

                # Store in session state
                st.session_state["training_result"] = training_result
                st.session_state["model_info"] = model_info
                st.session_state["viz_features"] = viz_features
                st.session_state["dataset_info"] = info

                st.success(f"✅ Model trained in {training_result.train_time:.2f}s!")

            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                return

    # Display results if available
    if "training_result" in st.session_state and st.session_state.get("model_info", {}).id == model_id:
        training_result = st.session_state["training_result"]
        model_info = st.session_state["model_info"]
        viz_features = st.session_state.get("viz_features")
        dataset_info = st.session_state.get("dataset_info", info)

        render_model_results(
            training_result, model_info, dataset_info, X, y, viz_features
        )


def render_model_results(
    training_result,
    model_info: ModelInfo,
    dataset_info: DatasetInfo,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    viz_features: Optional[Tuple[int, int]],
) -> None:
    """Render model training results and visualizations."""
    st.divider()
    st.header("📊 Results")

    model = training_result.model
    model_id = training_result.model_id
    X_test = training_result.X_test
    y_test = training_result.y_test
    y_train = training_result.y_train

    # Predictions
    y_pred, y_prob = predict(training_result)

    # Metrics based on task type
    if dataset_info.task_type == TaskType.CLASSIFICATION:
        metrics = compute_classification_metrics(y_test.values, y_pred, dataset_info.target_names)

        # Metric cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics.accuracy:.3f}")
        with col2:
            st.metric("Precision (weighted)", f"{metrics.precision:.3f}")
        with col3:
            st.metric("Recall (weighted)", f"{metrics.recall:.3f}")
        with col4:
            st.metric("F1 (weighted)", f"{metrics.f1:.3f}")

        # Confusion matrix
        st.subheader("📉 Confusion Matrix")
        fig = plot_confusion_matrix(metrics.confusion_matrix, dataset_info.target_names)
        display_figure(fig)

        # Classification report
        with st.expander("📋 Detailed Classification Report"):
            st.text(metrics.report)

        # Feature importance
        if model_info.supports_feature_importance:
            importance = get_feature_importance(training_result)
            if importance is not None:
                st.subheader("🎯 Feature Importance")
                fig = plot_feature_importance(importance)
                display_figure(fig, "Feature Importance")

        # Decision boundary
        if model_info.supports_decision_boundary and viz_features and len(X.columns) >= 2:
            st.subheader("🔮 Decision Boundary")
            try:
                xx, yy, Z = get_decision_boundary_data(training_result, viz_features)
                X_test_2d = X_test.iloc[:, list(viz_features)].values
                fig = plot_decision_boundary(
                    xx, yy, Z, X_test_2d, y_test.values,
                    (X.columns[viz_features[0]], X.columns[viz_features[1]]),
                    dataset_info.target_names,
                )
                display_figure(fig, f"Decision Boundary (Test Data)")
            except Exception as e:
                st.warning(f"Could not generate decision boundary: {e}")

        # Prediction preview
        st.subheader("🔮 Predictions Preview")
        pred_df = pd.DataFrame({
            "Actual": [dataset_info.target_names[v] if dataset_info.target_names else v for v in y_test.values[:20]],
            "Predicted": [dataset_info.target_names[v] if dataset_info.target_names else v for v in y_pred[:20]],
            "Match": ["✅" if a == p else "❌" for a, p in zip(y_test.values[:20], y_pred[:20])],
        })
        if y_prob is not None:
            pred_df["Max Prob"] = [f"{np.max(p):.3f}" for p in y_prob[:20]]
        st.dataframe(pred_df, use_container_width=True)

    elif dataset_info.task_type == TaskType.REGRESSION:
        metrics = compute_regression_metrics(y_test.values, y_pred)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAE", f"{metrics.mae:.3f}")
        with col2:
            st.metric("RMSE", f"{metrics.rmse:.3f}")
        with col3:
            st.metric("R²", f"{metrics.r2:.3f}")

        st.caption("Note: R² can be negative if the model fits worse than a horizontal line.")

        # Actual vs Predicted
        st.subheader("📈 Actual vs Predicted")
        fig = plot_actual_vs_predicted(y_test.values, y_pred)
        display_figure(fig)

        # Residuals
        st.subheader("📊 Residual Analysis")
        fig = plot_residuals(y_test.values, y_pred)
        display_figure(fig)

        # Regression line (if single feature or selected feature)
        if viz_features:
            st.subheader("📉 Regression Line")
            feat_idx = viz_features[0]
            feat_name = X.columns[feat_idx]
            X_test_feat = X_test.iloc[:, feat_idx].values
            sort_idx = np.argsort(X_test_feat)
            fig = plot_regression_line(
                X_test_feat[sort_idx],
                y_test.values[sort_idx],
                y_pred[sort_idx],
                feat_name,
                dataset_info.target_name or "target",
            )
            display_figure(fig)

        # Feature importance
        if model_info.supports_feature_importance:
            importance = get_feature_importance(training_result)
            if importance is not None:
                st.subheader("🎯 Feature Importance (Coefficient Magnitude)")
                fig = plot_feature_importance(importance)
                display_figure(fig)

    elif dataset_info.task_type == TaskType.CLUSTERING:
        # For clustering, we use the full dataset
        labels = y_pred
        inertia = model.named_steps["model"].inertia_ if hasattr(model.named_steps["model"], "inertia_") else 0

        metrics = compute_clustering_metrics(X.values, labels, inertia)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Clusters", metrics.n_clusters)
        with col2:
            st.metric("Inertia", f"{metrics.inertia:.2f}")
        with col3:
            st.metric("Silhouette" if metrics.silhouette else "Silhouette",
                     f"{metrics.silhouette:.3f}" if metrics.silhouette else "N/A")

        if metrics.silhouette is None:
            st.caption("Silhouette score undefined (need 2+ clusters and more samples than clusters)")

        # Cluster visualization
        if viz_features and X.shape[1] >= 2:
            st.subheader("🎯 Cluster Visualization")
            X_2d = X.iloc[:, list(viz_features)].values
            centroids = None
            if hasattr(model.named_steps["model"], "cluster_centers_"):
                # Need to transform centroids if scaled
                scaler = training_result.model.named_steps.get("scaler")
                if scaler:
                    centroids = scaler.inverse_transform(model.named_steps["model"].cluster_centers_)
                else:
                    centroids = model.named_steps["model"].cluster_centers_
                centroids = centroids[:, list(viz_features)]

            fig = plot_clusters(
                X_2d, labels, centroids,
                (X.columns[viz_features[0]], X.columns[viz_features[1]])
            )
            display_figure(fig, "K-Means Clusters")

            # Silhouette plot
            if metrics.silhouette_samples is not None:
                with st.expander("📊 Silhouette Analysis"):
                    fig = plot_silhouette(X_2d, labels, metrics.silhouette_samples)
                    display_figure(fig)

        # Cluster assignments
        st.subheader("🏷️ Cluster Assignments")
        cluster_df = pd.DataFrame({
            "Sample": range(len(labels)),
            "Cluster": labels,
        })
        st.dataframe(cluster_df.head(20), use_container_width=True)

    # Training info
    with st.expander("⚙️ Training Details"):
        st.json({
            "model": model_info.name,
            "parameters": training_result.params,
            "train_time_sec": round(training_result.train_time, 3),
            "train_samples": len(training_result.X_train),
            "test_samples": len(training_result.X_test),
            "features": training_result.feature_names,
        })


def render_model_comparison(
    loader: DatasetLoader,
    dataset_name: str,
    info: DatasetInfo,
    X: pd.DataFrame,
    y: Optional[pd.Series],
) -> None:
    """Render model comparison page."""
    st.header("⚖️ Model Comparison")

    compatible_models = list_models(info.task_type)
    model_options = {m.name: m.id for m in compatible_models}

    selected_names = st.multiselect(
        "Select Models to Compare",
        options=list(model_options.keys()),
        default=list(model_options.keys())[:3] if len(model_options) >= 3 else list(model_options.keys()),
        help="Choose 2+ models for fair comparison on the same data split"
    )

    if len(selected_names) < 2:
        st.info("Select at least 2 models to compare.")
        return

    # Shared training config
    st.subheader("🎯 Shared Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05, key="cmp_test_size")
    with col2:
        random_state = st.number_input("Random Seed", 0, 10000, 42, key="cmp_random_state")
    with col3:
        scale_features = st.checkbox("Scale Features", value=True, key="cmp_scale")

    # Per-model hyperparameters
    st.subheader("⚙️ Model Hyperparameters")
    model_params = {}

    for name in selected_names:
        model_id = model_options[name]
        model_info = get_model_info(model_id)
        with st.expander(f"{name} Parameters", expanded=False):
            params = {}
            cols = st.columns(2)
            for i, hp in enumerate(model_info.hyperparameters):
                col = cols[i % 2]
                with col:
                    if hp.param_type == "int":
                        params[hp.name] = st.number_input(
                            hp.display_name,
                            min_value=int(hp.min_value) if hp.min_value else 0,
                            max_value=int(hp.max_value) if hp.max_value else 1000,
                            value=int(hp.default),
                            step=int(hp.step) if hp.step else 1,
                            key=f"{model_id.value}_{hp.name}",
                            help=hp.help_text,
                        )
                    elif hp.param_type == "float":
                        params[hp.name] = st.number_input(
                            hp.display_name,
                            min_value=float(hp.min_value) if hp.min_value else 0.0,
                            max_value=float(hp.max_value) if hp.max_value else 100.0,
                            value=float(hp.default),
                            step=float(hp.step) if hp.step else 0.01,
                            format="%.3f",
                            key=f"{model_id.value}_{hp.name}",
                            help=hp.help_text,
                        )
                    elif hp.param_type == "select":
                        params[hp.name] = st.selectbox(
                            hp.display_name,
                            options=hp.options,
                            index=hp.options.index(hp.default) if hp.default in hp.options else 0,
                            key=f"{model_id.value}_{hp.name}",
                            help=hp.help_text,
                        )
            model_params[model_id] = params

    # Compare button
    if st.button("🚀 Compare Models", type="primary", use_container_width=True):
        with st.spinner("Training and evaluating models..."):
            try:
                # Single split for all models
                split_result = split_and_scale(
                    X, y, test_size=test_size, random_state=random_state, scale=scale_features,
                    task_type=info.task_type
                )

                results = {}
                for name in selected_names:
                    model_id = model_options[name]
                    params = model_params[model_id]

                    training_result = train_model(
                        model_id, params, split_result,
                        target_name=info.target_name,
                        target_names=info.target_names,
                    )

                    y_pred, _ = predict(training_result)

                    if info.task_type == TaskType.CLASSIFICATION:
                        metrics = compute_classification_metrics(
                            split_result.y_test.values, y_pred, info.target_names
                        )
                        results[name] = {"metrics": metrics, "training_result": training_result}
                    elif info.task_type == TaskType.REGRESSION:
                        metrics = compute_regression_metrics(split_result.y_test.values, y_pred)
                        results[name] = {"metrics": metrics, "training_result": training_result}
                    elif info.task_type == TaskType.CLUSTERING:
                        inertia = training_result.model.named_steps["model"].inertia_
                        metrics = compute_clustering_metrics(X.values, y_pred, inertia)
                        results[name] = {"metrics": metrics, "training_result": training_result}

                st.session_state["comparison_results"] = results
                st.session_state["comparison_config"] = {
                    "dataset": dataset_name,
                    "test_size": test_size,
                    "random_state": random_state,
                    "scale_features": scale_features,
                }
                st.success("✅ Comparison complete!")

            except Exception as e:
                st.error(f"Comparison failed: {str(e)}")
                return

    # Display comparison results
    if "comparison_results" in st.session_state:
        results = st.session_state["comparison_results"]
        config = st.session_state.get("comparison_config", {})

        st.divider()
        st.header("📊 Comparison Results")

        # Determine available metrics
        if info.task_type == TaskType.CLASSIFICATION:
            available_metrics = ["accuracy", "precision", "recall", "f1"]
        elif info.task_type == TaskType.REGRESSION:
            available_metrics = ["mae", "rmse", "r2"]
        else:
            available_metrics = ["inertia", "silhouette"]

        # Metric selector
        metric = st.selectbox("Primary Metric", available_metrics, index=0)

        # Comparison table
        st.subheader("📋 Metrics Table")
        comparison_data = []
        for name, result in results.items():
            metrics_obj = result["metrics"]
            row = {"Model": name}
            for m in available_metrics:
                if hasattr(metrics_obj, m):
                    val = getattr(metrics_obj, m)
                    row[m.capitalize()] = f"{val:.4f}" if val is not None else "N/A"
            comparison_data.append(row)

        if comparison_data:
            df = pd.DataFrame(comparison_data)
            # Highlight best
            if metric in ["accuracy", "precision", "recall", "f1", "r2", "silhouette"]:
                best_idx = df[metric.capitalize()].astype(float).idxmax()
            else:
                best_idx = df[metric.capitalize()].astype(float).idxmin()
            st.dataframe(df.style.highlight_max(axis=0, subset=[metric.capitalize()], color="lightgreen") 
                        if metric in ["accuracy", "precision", "recall", "f1", "r2", "silhouette"]
                        else df.style.highlight_min(axis=0, subset=[metric.capitalize()], color="lightgreen"),
                        use_container_width=True)

        # Bar chart
        st.subheader("📊 Bar Comparison")
        fig = plot_model_comparison_bar(results, metric)
        display_figure(fig, f"Model Comparison: {metric.capitalize()}")

        # Radar chart (for classification/regression with multiple metrics)
        if len(available_metrics) >= 3:
            st.subheader("🎯 Radar Comparison")
            fig = plot_metric_radar(results, available_metrics)
            display_figure(fig, "Multi-Metric Radar Chart")

        # Best model note
        st.info(f"""
        **Note:** The "best" model on {metric} may not be best for your specific use case.
        Consider: interpretability, training time, prediction latency, robustness, and domain requirements.
        """)


def render_sidebar() -> str:
    """Render sidebar navigation. Returns selected page."""
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        page = st.radio(
            "Select Page",
            ["Overview", "Dataset Explorer", "Model Explorer", "Model Comparison"],
            index=0,
        )

        st.divider()
        st.markdown("""
        ### About
        **Algorithm Visualizer ML** — Educational tool for learning ML algorithms.

        **Built with:** Streamlit, scikit-learn, pandas, matplotlib
        """)

    return page


def render_page(page: str) -> None:
    """Route to the appropriate page renderer."""
    loader = get_dataset_loader(random_state=42)

    if page == "Overview":
        render_overview()

    elif page == "Dataset Explorer":
        dataset_name, info, X, y = render_dataset_explorer(loader)
        st.session_state["current_dataset"] = dataset_name
        st.session_state["dataset_info"] = info
        st.session_state["X"] = X
        st.session_state["y"] = y

    elif page == "Model Explorer":
        if "current_dataset" not in st.session_state:
            st.warning("Please select a dataset first in the Dataset Explorer.")
            return
        dataset_name = st.session_state["current_dataset"]
        info = st.session_state["dataset_info"]
        X = st.session_state["X"]
        y = st.session_state["y"]
        render_model_explorer(loader, dataset_name, info, X, y)

    elif page == "Model Comparison":
        if "current_dataset" not in st.session_state:
            st.warning("Please select a dataset first in the Dataset Explorer.")
            return
        dataset_name = st.session_state["current_dataset"]
        info = st.session_state["dataset_info"]
        X = st.session_state["X"]
        y = st.session_state["y"]
        render_model_comparison(loader, dataset_name, info, X, y)