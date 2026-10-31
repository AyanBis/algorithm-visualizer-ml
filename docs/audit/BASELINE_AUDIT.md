# Baseline Audit — Algorithm Visualizer ML

**Date:** 2026-10-28
**Commit:** ca0b7f1 (main branch)
**Repository:** https://github.com/AyanBis/algorithm-visualizer-ml

---

## 1. Current Architecture

### 1.1 Overview

Algorithm Visualizer ML is a **Streamlit-based educational web application** for exploring machine learning algorithms. It follows a modular monolith architecture with clear separation between:

- **Core ML modules** (`ml_visualizer/`) — dataset loading, preprocessing, model registry, training, metrics, visualizations
- **UI layer** (`ui/`) — Streamlit page renderers and comparison utilities
- **Entry point** (`app.py`) — Streamlit application bootstrap

### 1.2 Module Map

| Module | Lines | Responsibility |
|--------|-------|----------------|
| `ml_visualizer/datasets.py` | 165 | Load 5 built-in scikit-learn datasets; metadata (task type, feature names, descriptions) |
| `ml_visualizer/preprocessing.py` | 129 | Leakage-safe train/test split; feature scaling with DataFrame column preservation |
| `ml_visualizer/model_registry.py` | 280 | Define 7 models with hyperparameters, descriptions, capabilities, default params |
| `ml_visualizer/training.py` | 175 | Model training, prediction, feature importance, decision boundary mesh generation |
| `ml_visualizer/metrics.py` | 306 | Classification, regression, clustering metrics + Matplotlib plotting functions |
| `ml_visualizer/visualizations.py` | 209 | Streamlit display utilities; dataset overview, correlation matrix, comparison charts |
| `ui/components.py` | 798 | Streamlit page renderers: Overview, Dataset Explorer, Model Explorer, Model Comparison |
| `ui/comparison.py` | 145 | Fair model comparison with shared data splits and metric extraction |

### 1.3 Data Flow

```
User Selection (Sidebar)
         │
         ▼
Dataset Loader ──► (X, y, DatasetInfo)
         │
         ▼
Preprocessing ──► SplitResult (X_train, X_test, y_train, y_test, scaler)
         │
         ▼
Model Registry ──► ModelInfo + HyperParameters
         │
         ▼
Training ──► TrainingResult (pipeline, params, timings, data splits)
         │
         ▼
Prediction ──► (y_pred, y_prob)
         │
         ▼
Metrics ──► ClassificationMetrics / RegressionMetrics / ClusteringMetrics
         │
         ▼
Visualizations ──► Matplotlib figures → Streamlit display
```

### 1.4 Technology Stack

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.10+ | Required |
| Streamlit | 1.35+ | Web framework |
| scikit-learn | 1.4+ | ML algorithms |
| pandas | 2.1+ | Data manipulation |
| NumPy | 1.26+ | Numerical computing |
| Matplotlib | 3.8+ | Static visualizations |
| pytest | 8.0+ | Testing |

---

## 2. Existing Features

### 2.1 Dataset Explorer
- **5 built-in datasets** from scikit-learn (no external downloads)
- Dataset metadata display: task type, samples, features, target info
- Data preview (first 10 rows with human-readable class names)
- Feature statistics table (mean, std, min, max, non-null counts)
- Feature distribution histograms (colored by class for classification)
- Correlation matrix heatmap
- Target class distribution for classification datasets

### 2.2 Model Explorer
- **7 algorithms** across 3 task types
- Model compatibility filtering (only shows models matching dataset task type)
- Educational explanations: description, how it works, when to use, limitations
- Interactive hyperparameter controls (int, float, select types)
- Training configuration: test size, random seed, feature scaling toggle
- Feature selection for 2D visualizations
- Real-time training with spinner
- Results display:
  - **Classification:** Accuracy, Precision, Recall, F1 (weighted), Confusion Matrix, Classification Report, Feature Importance, Decision Boundary (2D slice), Prediction Preview
  - **Regression:** MAE, RMSE, R², Actual vs Predicted, Residuals, Regression Line, Feature Importance
  - **Clustering:** Cluster count, Inertia, Silhouette, Cluster Scatter Plot, Silhouette Analysis, Test Set Assignments
- Training details expander (JSON)

### 2.3 Model Comparison
- Multi-select compatible models (2+)
- Shared train/test split for fair comparison
- Per-model hyperparameter configuration
- Metrics table with highlighted best model
- Bar chart comparison
- Radar chart (multi-metric)
- Explicit note that "best" depends on use case

### 2.4 Automated Testing
- **64 tests** covering datasets, metrics, models, app flow, comparison fairness, silhouette consistency
- All tests pass (verified 2026-10-28)

---

## 3. Existing Test Results

```
============================= test session starts ==============================
collected 64 items

tests/test_app_flow.py ...................... 18 passed
tests/test_datasets.py ...................... 10 passed
tests/test_metrics.py ....................... 12 passed
tests/test_models.py ........................ 24 passed

============================== 64 passed in 3.23s ==============================
```

**Coverage Areas:**
- Dataset loading, metadata, reproducibility, caching
- Classification metrics (binary, multiclass, perfect, all-wrong, zero-division)
- Regression metrics (normal, perfect, negative R²)
- Clustering metrics (valid, single cluster, each-point-own-cluster, silhouette samples)
- Model registry (all 7 models, listing by task, creation, pipelines)
- Model training (all 7 algorithms, feature importance, SVM hyperparameters, SVM calibration)
- App flow (classification, binary classification, regression, clustering workflows)
- Data validation (empty, NaN, inf, length mismatch)
- Comparison fairness (same split, consistent random state, missing metrics handling)
- Silhouette consistency (2D vs full space)

---

## 4. Known Defects (Verified)

### 4.1 Critical — None Found

### 4.2 High Severity

| ID | Defect | Location | Verification |
|----|--------|----------|--------------|
| H-01 | **SVM pipeline uses CalibratedClassifierCV inside Pipeline with StandardScaler** — the scaler is fitted on training data, but CalibratedClassifierCV does cross-validation internally, potentially causing data leakage during calibration if CV folds aren't handled correctly. | `model_registry.py:264-266` | Code inspection. The pipeline structure is `[("scaler", StandardScaler()), ("model", CalibratedClassifierCV(SVC(...)))]`. During `fit`, the scaler transforms training data, then CalibratedClassifierCV does CV on scaled data. This is correct *if* the CV is done within the training split. However, the calibration CV uses the *same* data the scaler was fitted on, which is acceptable. **Risk: Medium — needs verification with sklearn version.** |
| H-02 | **K-Means evaluation uses full dataset (X) instead of test split** — `compute_clustering_metrics(X.values, y_pred, inertia)` in `comparison.py:82` passes the full feature matrix, but `y_pred` is test-set predictions. This mixes train and test data for silhouette calculation. | `comparison.py:81-82`, `components.py:456` | Verified by code inspection. In `components.py:456`, clustering metrics use `training_result.X_train.values` with `labels` from training data — correct. But `comparison.py:82` uses `X.values` (full dataset) with `y_pred` (test predictions) — **INCORRECT**. |

### 4.3 Medium Severity

| ID | Defect | Location | Verification |
|----|--------|----------|--------------|
| M-01 | **Decision boundary uses test data for scatter but training data for mesh** — mesh grid bounds use `X_train` min/max, but scatter uses `X_test`. If test data falls outside train bounds, points won't appear in visualization. | `training.py:153-154` | Code inspection. Bounds: `X_train[f1_name].min() - 0.5`. Test points outside this range are clipped from view. |
| M-02 | **Clustering silhouette computation inconsistency** — `components.py` computes 2D silhouette for visualization but full-space silhouette for metrics. The UI shows both but labels them confusingly. | `components.py:492-513` | Code inspection. Two separate silhouette computations with different feature spaces. |
| M-03 | **No validation that hyperparameter values are compatible with sklearn version** — e.g., `max_features="sqrt"` for RandomForest may behave differently across versions. | `model_registry.py:125-127` | Known sklearn version dependency. Default params use string "sqrt" which works in current versions. |
| M-04 | **Session state persistence causes stale results** — Model Explorer checks `st.session_state.get("model_info", {}).id == model_id` but if user switches datasets without retraining, old results may display. | `components.py:307` | Code inspection. State keyed by model_id only, not dataset. |

### 4.4 Low Severity

| ID | Defect | Location | Verification |
|----|--------|----------|--------------|
| L-01 | **Correlation matrix text becomes unreadable with >10 features** — Breast Cancer has 30 features; text overlaps. | `visualizations.py:95-101` | Visual inspection. Font size 7, 30x30 matrix — labels overlap. |
| L-02 | **Feature importance for Linear Regression uses absolute coefficient mean** — for multiclass Logistic Regression, averages across classes. May misrepresent per-class importance. | `training.py:114-115` | Code inspection. `np.abs(coef).mean(axis=0)` for multi-class. |
| L-03 | **No caching invalidation for dataset loader** — `_cache` persists for process lifetime. If random_state changes, cached data is wrong. | `datasets.py:96, 147` | Code inspection. Cache keyed by dataset name only. |
| L-04 | **Hardcoded random_state=42 in multiple places** — reduces flexibility for experimentation. | Multiple files | Code inspection. |

---

## 5. Suspected Issues (Not Fully Verified)

| ID | Suspected Issue | How to Reproduce/Test |
|----|-----------------|------------------------|
| S-01 | **SVM CalibratedClassifierCV may leak data via cross-validation** — If the internal CV doesn't respect the train/test boundary established by the outer pipeline. | Train SVM on a dataset with known leakage-sensitive pattern; compare with manual CV. |
| S-02 | **KNN with k > n_train_samples raises ValueError at prediction time, not training** — Error surfacing is delayed. | Set k=50 on Iris (120 train samples); click Train → error only on Predict. |
| S-03 | **Decision boundary for >2 features uses mean of other features** — This creates a 2D slice that may not represent true decision surface. | Train on 4-feature Iris; visualize boundary on features 0,1; verify caption explains slice. |
| S-04 | **Streamlit session state not cleared on dataset change** — Model Explorer may show results from previous dataset. | Load Iris → Train → Load Diabetes → Model Explorer shows Iris results until retrain. |
| S-05 | **Comparison page doesn't clear results when dataset changes** — Stale comparison results persist. | Run comparison on Iris → Switch to Diabetes → Comparison still shows Iris results. |
| S-06 | **No protection against extremely large model configurations** — e.g., RandomForest n_estimators=5000 could hang UI. | Set n_estimators=5000; click Train; observe timeout/blocking. |

---

## 6. Reproducibility Issues

| Issue | Impact | Status |
|-------|--------|--------|
| Random state hardcoded to 42 in dataset loader default | Synthetic datasets always identical | By design for education |
| Model default_params include `random_state=42` for applicable models | Full reproducibility within session | Verified by `test_deterministic_results` |
| Dataset loader cache not keyed by random_state | Changing random_state after first load returns stale data | **Confirmed defect L-03** |
| Streamlit session state survives page navigation | Results persist across page switches | By design, but causes M-04 |

---

## 7. UX Problems

| Problem | Severity | Description |
|---------|----------|-------------|
| No visual workflow representation | **High** | Users cannot see the ML pipeline stages (load → split → scale → train → evaluate) |
| No execution timeline or stage durations | **High** | Training time shown only as total; no per-stage breakdown |
| No progress indication during training | **Medium** | Only spinner; no granular progress for slow models (RF, SVM) |
| Sidebar navigation resets on interaction | **Medium** | Streamlit reruns reset scroll position; no deep linking |
| No keyboard shortcuts | **Low** | Mouse-only interaction |
| Mobile layout not tested | **Medium** | Streamlit responsive but complex layouts may break |
| No accessibility features (ARIA, focus management) | **Medium** | Standard Streamlit components only |
| Color-only status indication (green/red metrics) | **Medium** | No icons/text for colorblind users |

---

## 8. Accessibility Problems

| Issue | WCAG Criterion | Status |
|-------|----------------|--------|
| No semantic HTML control | 1.3.1 Info and Relationships | Streamlit generates generic divs |
| No visible focus indicators | 2.4.7 Focus Visible | Default browser only |
| Color-only metric status | 1.4.1 Use of Color | Green/red metrics |
| No alt text for matplotlib charts | 1.1.1 Non-text Content | Images only |
| No reduced-motion support | 2.3.3 Animation from Interactions | No animations currently |
| No screen-reader announcements for dynamic updates | 4.1.3 Status Messages | Streamlit limitations |

---

## 9. Performance Problems

| Issue | Impact | Evidence |
|-------|--------|----------|
| Matplotlib figure generation on every render | Medium | Each chart creates new Figure; no caching |
| Full dataset correlation matrix (30x30) rendered as image | Low | Breast Cancer: 900 cells with text |
| Decision boundary meshgrid 100x100 = 10,000 predictions | Medium | Recomputed on every parameter change |
| No lazy loading for heavy visualizations | Low | All charts render on page load |
| SVM CalibratedClassifierCV with CV=3 does 3x training | High for large data | Default CV folds multiply training time |
| RandomForest n_estimators=100 default, no progress | Medium | Can take seconds; UI frozen |

---

## 10. Security Considerations

| Concern | Status | Notes |
|---------|--------|-------|
| Arbitrary code execution | **Not applicable** | No user code execution; no `eval`/`exec` |
| Data exposure | **Not applicable** | Only built-in sklearn datasets |
| XSS via `unsafe_allow_html` | **Low risk** | Only used for static CSS in `app.py:17-36` |
| Dependency vulnerabilities | **Unknown** | Requirements unpinned; no audit run |
| No authentication/authorization | **By design** | Single-user educational tool |

---

## 11. Migration Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **ML correctness regression** | High | Critical | Comprehensive regression tests comparing old vs new backend outputs |
| **SVM calibration behavior change** | Medium | High | Preserve exact CalibratedClassifierCV configuration; pin sklearn version |
| **Streamlit session state → React state migration** | High | High | Design explicit state management; use TanStack Query for server state |
| **Matplotlib → Recharts/WebGL visualization parity** | Medium | Medium | Accept visual differences; prioritize correctness over pixel-perfect match |
| **SSE event stream reliability** | Medium | High | Implement reconnection, event buffering, sequence numbers |
| **In-process background task management** | Medium | Medium | Use asyncio tasks with proper cancellation; document limitations |
| **Loss of educational explanations** | Low | High | Copy all markdown content verbatim; enhance with progressive disclosure |

---

## 12. Features That Must Be Preserved

1. **All 7 algorithms** with current hyperparameter spaces
2. **All 5 datasets** with identical loading behavior
3. **Leakage-safe preprocessing** (split before scale, fit on train only)
4. **Stratified splits for classification only**
5. **Weighted metrics for multiclass classification**
6. **SVM probability via CalibratedClassifierCV (Platt scaling)**
7. **Decision boundary 2D slice with mean-holding for other features**
7. **Clustering evaluation on training data (not test)**
8. **Silhouette computed in visualization feature space (2D)**
9. **Fair model comparison: same split, same preprocessing**
10. **All educational text (descriptions, how-it-works, limitations)**
11. **MIT License and attribution to Algo-Visualizer**
12. **Deterministic results with fixed random seeds**

---

## 13. Features That Should Be Improved

| Feature | Improvement | Priority |
|---------|-------------|----------|
| **Workflow visualization** | Add interactive node-graph showing execution stages with real-time status | **Critical (new feature)** |
| **Execution timeline** | Chronological event log with timestamps, durations, stage details | **Critical (new feature)** |
| **Frontend framework** | Migrate Streamlit → React + TypeScript + Vite + Tailwind | **Critical (migration)** |
| **Backend API** | Add FastAPI layer with REST + SSE for workflow events | **Critical (migration)** |
| **Design system** | Light theme, Antigravity-inspired, design tokens, responsive | **High** |
| **Accessibility** | Semantic HTML, keyboard nav, focus visible, screen-reader support | **High** |
| **Error handling** | User-friendly validation messages, recovery actions | **Medium** |
| **Performance** | Lazy loading, chart memoization, virtualized tables | **Medium** |
| **Mobile responsiveness** | Test and fix layouts on tablet/phone | **Medium** |

---

## 14. Features That Should NOT Be Implemented Yet

| Feature | Reason |
|---------|--------|
| User authentication/accounts | No multi-user requirement; adds complexity |
| Model persistence (save/load) | Out of scope for educational tool |
| Custom dataset upload | Requires file handling, validation, security |
| Hyperparameter tuning (grid/random search) | Educational focus on manual exploration |
| Cross-validation | Single split is intentional for simplicity |
| Distributed training / job queues | Over-engineering for local app |
| Database for experiment history | In-memory sufficient for session-based use |
| WebSocket bidirectional communication | SSE sufficient for one-way event stream |
| Real-time training progress percentages | sklearn doesn't expose intermediate progress reliably |
| Export to ONNX/PMML/other formats | Not an educational requirement |

---

## 15. Summary Assessment

| Dimension | Score | Notes |
|-----------|-------|-------|
| **ML Correctness** | 8/10 | Strong leakage prevention; minor issues in comparison clustering eval |
| **Code Quality** | 8/10 | Clean modular structure; good type hints; dataclasses used well |
| **Test Coverage** | 9/10 | 64 tests covering core logic; integration workflows tested |
| **UX/Design** | 4/10 | Functional but utilitarian; no workflow visualization |
| **Accessibility** | 3/10 | Streamlit defaults only; significant gaps |
| **Performance** | 6/10 | Acceptable for small datasets; no optimization |
| **Maintainability** | 8/10 | Clear separation of concerns; low coupling |
| **Migration Readiness** | 7/10 | ML logic well-isolated; UI tightly coupled to Streamlit |

---

## 16. Recommended Migration Approach

1. **Preserve ML core** (`ml_visualizer/`) as a Python package, expose via FastAPI
2. **Build React frontend** from scratch using design system
3. **Implement workflow event system** as new backend capability
4. **Migrate incrementally**: backend API → frontend pages → workflow studio
5. **Keep Streamlit running** until React version reaches feature parity
6. **Regression test** every model/dataset combination against baseline outputs

---

*End of Baseline Audit*