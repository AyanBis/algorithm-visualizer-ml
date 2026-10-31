# Migration Plan — Algorithm Visualizer ML

**Source:** Streamlit + Python monolith
**Target:** React/TypeScript frontend + FastAPI backend
**Baseline Commit:** ca0b7f1
**Date:** 2026-10-28

---

## 1. Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Process                         │
├─────────────────────────────────────────────────────────────┤
│  ui/components.py  ──►  ml_visualizer/*  ──►  sklearn       │
│  ui/comparison.py                                     │
│  app.py (entry)                                         │
└─────────────────────────────────────────────────────────────┘
```

**Characteristics:**
- Single-process, synchronous execution
- Streamlit manages UI state (session_state)
- Matplotlib generates static images for display
- All 7 models, 5 datasets, metrics, visualizations in-process
- 64 passing tests

---

## 2. Target Architecture

```
┌──────────────────────┐     HTTP + SSE      ┌──────────────────────┐
│   React Frontend     │ ◄─────────────────► │   FastAPI Backend    │
│  (Vite + TypeScript) │                     │  (Python + sklearn)  │
├──────────────────────┤                     ├──────────────────────┤
│  Pages               │                     │  API Endpoints       │
│  - Overview          │                     │  GET  /api/health    │
│  - Dataset Explorer  │                     │  GET  /api/datasets  │
│  - Model Explorer    │                     │  GET  /api/models    │
│  - Model Comparison  │                     │  POST /api/experiments│
│  - Workflow Studio   │                     │  GET  /api/experiments/{id}│
│                      │                     │  GET  /api/experiments/{id}/events/stream│
│  Components          │                     │  GET  /api/experiments/{id}/results│
│  - Workflow Graph    │                     │  POST /api/experiments/{id}/cancel│
│  - Timeline          │                     └──────────────────────┘
│  - Charts (Recharts) │                              │
│  - Forms (RHF + Zod) │                              ▼
│  State (TanStack Query)                    ┌──────────────────────┐
│  Routing (React Router)                    │  ML Core (Preserved) │
└──────────────────────┘                     │  ml_visualizer/*     │
                                              └──────────────────────┘
```

### 2.1 Frontend Stack

| Library | Purpose | Version Strategy |
|---------|---------|------------------|
| React | UI framework | Latest stable (18.x) |
| TypeScript | Type safety | Latest stable (5.x) |
| Vite | Build tool | Latest stable (5.x) |
| Tailwind CSS | Styling | Latest stable (3.x) |
| shadcn/ui | Component primitives | Latest compatible |
| Lucide React | Icons | Latest stable |
| React Router | Navigation | v6 (stable) |
| TanStack Query | Server state/cache | v5 (stable) |
| React Hook Form | Form handling | Latest stable |
| Zod | Validation | Latest stable |
| @xyflow/react | Workflow diagram | Latest stable |
| Recharts | Charts | Latest stable |
| Motion | Animation (minimal) | Latest stable |

### 2.2 Backend Stack

| Library | Purpose | Version Strategy |
|---------|---------|------------------|
| FastAPI | API framework | Latest stable (0.111+) |
| Pydantic | Data validation | v2 (stable) |
| Uvicorn | ASGI server | Latest stable |
| sse-starlette | SSE support | Latest stable |
| ml_visualizer | ML computation | **Preserved from current** |
| pytest | Testing | Latest stable |

### 2.3 Architecture Principles

1. **Modular monolith** — Single backend process, no microservices
2. **Preserve ML core** — `ml_visualizer/` package reused without rewrite
3. **SSE for workflow events** — One-way server→client event stream
4. **In-process background tasks** — asyncio tasks for experiment execution
5. **No database initially** — In-memory experiment store with TTL cleanup
6. **Stateless API** — All experiment state in backend memory, referenced by UUID

---

## 3. Frontend Component Plan

### 3.1 Page Structure

```
frontend/src/
├── app/
│   ├── App.tsx              # Root component, providers
│   ├── router.tsx           # Route definitions
│   └── providers.tsx        # TanStack Query, Theme, Toaster
├── pages/
│   ├── Overview.tsx         # Home page
│   ├── DatasetExplorer.tsx  # Dataset selection + exploration
│   ├── ModelExplorer.tsx    # Model config + training + results
│   ├── ModelComparison.tsx  # Multi-model comparison
│   └── WorkflowStudio.tsx   # Interactive workflow diagram + timeline
├── components/
│   ├── layout/
│   │   ├── Header.tsx
│   │   ├── Sidebar.tsx
│   │   └── Layout.tsx
│   ├── ui/                  # shadcn/ui primitives + custom
│   │   ├── Button.tsx
│   │   ├── Card.tsx
│   │   ├── Select.tsx
│   │   ├── Slider.tsx
│   │   ├── Checkbox.tsx
│   │   ├── Tabs.tsx
│   │   ├── Table.tsx
│   │   ├── Badge.tsx
│   │   └── ...
│   ├── charts/
│   │   ├── ConfusionMatrix.tsx
│   │   ├── DecisionBoundary.tsx
│   │   ├── ActualVsPredicted.tsx
│   │   ├── ResidualPlot.tsx
│   │   ├── RegressionLine.tsx
│   │   ├── FeatureImportance.tsx
│   │   ├── ClusterPlot.tsx
│   │   ├── SilhouettePlot.tsx
│   │   ├── ComparisonBar.tsx
│   │   └── ComparisonRadar.tsx
│   └── workflow/
│       ├── WorkflowGraph.tsx       # @xyflow/react diagram
│       ├── WorkflowNode.tsx        # Stage node component
│       ├── WorkflowEdge.tsx        # Connection edge
│       ├── StageDetailPanel.tsx    # Node click detail
│       ├── Timeline.tsx            # Chronological event log
│       ├── FlowControls.tsx        # Start/Cancel/Rerun/Reset
│       └── EventLog.tsx            # Expandable event list
├── features/
│   ├── datasets/
│   │   ├── api.ts           # Dataset API calls
│   │   ├── hooks.ts         # useDatasets, useDataset
│   │   └── types.ts
│   ├── models/
│   │   ├── api.ts
│   │   ├── hooks.ts
│   │   └── types.ts
│   ├── experiments/
│   │   ├── api.ts
│   │   ├── hooks.ts         # useExperiment, useExperimentEvents
│   │   ├── types.ts         # Experiment, Stage, Event types
│   │   └── sse.ts           # SSE connection manager
│   └── comparison/
│       ├── api.ts
│       ├── hooks.ts
│       └── types.ts
├── hooks/
│   ├── useDebounce.ts
│   ├── useLocalStorage.ts
│   └── useReducedMotion.ts
├── lib/
│   ├── api.ts               # Base API client + error handling
│   ├── utils.ts             # Formatters, helpers
│   └── validation.ts        # Zod schemas
├── types/
│   ├── api.ts               # Shared API types
│   ├── ml.ts                # ML domain types
│   └── workflow.ts          # Workflow event types
├── styles/
│   ├── globals.css          # Tailwind + design tokens
│   └── design-tokens.css    # CSS custom properties
└── main.tsx                 # Entry point
```

### 3.2 Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| `Overview` | Product intro, algorithm cards, quick-start |
| `DatasetExplorer` | Dataset list, metadata, preview, distributions, correlations |
| `ModelExplorer` | Model selection, hyperparameters, train config, results display |
| `ModelComparison` | Multi-model config, shared split, comparison charts |
| `WorkflowStudio` | Full-screen workflow graph + timeline + controls |
| `WorkflowGraph` | @xyflow/react diagram with pan/zoom/fit |
| `StageDetailPanel` | Side panel with stage I/O, timing, errors |
| `Timeline` | Chronological stage list with durations |
| `FlowControls` | Experiment lifecycle buttons |
| `Charts` | Recharts-based interactive visualizations |

---

## 4. Backend API Plan

### 4.1 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/datasets` | List all datasets with metadata |
| `GET` | `/api/datasets/{id}` | Get dataset detail + preview + statistics |
| `GET` | `/api/models` | List all models with hyperparameters |
| `GET` | `/api/models/{id}` | Get model detail (description, params, capabilities) |
| `POST` | `/api/experiments` | Create and start experiment |
| `GET` | `/api/experiments/{id}` | Get experiment status + config |
| `GET` | `/api/experiments/{id}/events` | Get all events (polling fallback) |
| `GET` | `/api/experiments/{id}/events/stream` | SSE stream for real-time events |
| `GET` | `/api/experiments/{id}/results` | Get final results + visualizations data |
| `POST` | `/api/experiments/{id}/cancel` | Request cancellation |

### 4.2 Request/Response Contracts

#### Dataset List Response
```json
{
  "datasets": [
    {
      "id": "iris",
      "name": "Iris",
      "task_type": "classification",
      "description": "...",
      "n_samples": 150,
      "n_features": 4,
      "feature_names": [...],
      "target_name": "species",
      "target_names": ["setosa", "versicolor", "virginica"]
    }
  ]
}
```

#### Model List Response
```json
{
  "models": [
    {
      "id": "logistic_regression",
      "name": "Logistic Regression",
      "task_type": "classification",
      "description": "...",
      "how_it_works": "...",
      "when_to_use": "...",
      "limitations": "...",
      "hyperparameters": [
        {"name": "C", "display_name": "Regularization Strength", "type": "float", "default": 1.0, "min": 0.01, "max": 100, "step": 0.01, "help_text": "..."}
      ],
      "supports_probability": true,
      "supports_decision_boundary": true,
      "supports_feature_importance": true,
      "default_params": {"C": 1.0, "max_iter": 200, "solver": "lbfgs", "random_state": 42}
    }
  ]
}
```

#### Experiment Create Request
```json
{
  "dataset_id": "iris",
  "model_id": "logistic_regression",
  "hyperparameters": {"C": 1.0, "max_iter": 200, "solver": "lbfgs"},
  "test_size": 0.2,
  "random_state": 42,
  "scale_features": true,
  "scaler_type": "standard",
  "viz_features": [0, 1]
}
```

#### Experiment Create Response
```json
{
  "experiment_id": "uuid",
  "status": "pending",
  "created_at": "ISO-8601"
}
```

#### Experiment Status Response
```json
{
  "experiment_id": "uuid",
  "status": "running|completed|failed|cancelled",
  "config": {...},
  "current_stage": "model_training",
  "progress": 0.6,
  "started_at": "ISO-8601",
  "completed_at": "ISO-8601|null",
  "total_duration_ms": 1250
}
```

#### Workflow Event (SSE)
```json
{
  "experiment_id": "uuid",
  "event_id": "uuid",
  "event_type": "stage_started|stage_completed|stage_failed|experiment_completed|experiment_failed",
  "stage_id": "dataset_loading|data_validation|train_test_split|preprocessing|model_initialization|model_training|prediction|evaluation|visualization",
  "stage_name": "Dataset Loading",
  "status": "pending|running|completed|failed|skipped|cancelled",
  "sequence": 2,
  "timestamp": "ISO-8601",
  "started_at": "ISO-8601",
  "completed_at": "ISO-8601|null",
  "duration_ms": 45.2,
  "message": "Dataset loaded successfully",
  "details": {"n_samples": 150, "n_features": 4, "task_type": "classification"},
  "error": null
}
```

#### Experiment Results Response
```json
{
  "experiment_id": "uuid",
  "status": "completed",
  "config": {...},
  "metrics": {
    "task_type": "classification",
    "accuracy": 0.96,
    "precision": 0.96,
    "recall": 0.96,
    "f1": 0.96,
    "confusion_matrix": [[10,0,0],[0,11,0],[0,0,9]],
    "per_class": {...}
  },
  "visualizations": {
    "decision_boundary": {"xx": [...], "yy": [...], "Z": [...], "X_test_2d": [...], "y_test": [...]},
    "feature_importance": [{"feature": "petal length", "importance": 0.85}, ...],
    "confusion_matrix": [[10,0,0],[0,11,0],[0,0,9]]
  },
  "predictions": [{"actual": 0, "predicted": 0, "probabilities": [0.99, 0.01, 0.0]}],
  "train_time_ms": 45,
  "total_time_ms": 1250,
  "stage_timings": {"dataset_loading": 5, "data_validation": 2, ...},
  "warnings": []
}
```

---

## 5. Workflow Event Schema

### 5.1 Stage Definitions (Fixed Order)

| Seq | Stage ID | Stage Name | Description |
|-----|----------|------------|-------------|
| 1 | `experiment_config` | Experiment Configuration | Validate inputs, prepare config |
| 2 | `dataset_loading` | Dataset Loading | Load dataset, extract metadata |
| 3 | `data_validation` | Data Validation | Check NaN, inf, alignment, compatibility |
| 4 | `train_test_split` | Train/Test Split | Stratified split (classification) or random |
| 5 | `preprocessing` | Preprocessing | Fit scaler on train, transform both |
| 6 | `model_initialization` | Model Initialization | Instantiate model with hyperparameters |
| 7 | `model_training` | Model Training | Fit model on training data |
| 8 | `prediction` | Prediction / Assignment | Predict on test set (or assign clusters) |
| 9 | `evaluation` | Evaluation | Compute metrics |
| 10 | `visualization` | Visualization | Generate chart data |
| 11 | `results_ready` | Results Ready | Finalize, emit terminal event |

### 5.2 Event Types

| Event Type | When Emitted | Payload |
|------------|--------------|---------|
| `stage_started` | Stage begins | `stage_id`, `stage_name`, `started_at`, `sequence` |
| `stage_completed` | Stage succeeds | `stage_id`, `completed_at`, `duration_ms`, `details` |
| `stage_failed` | Stage fails | `stage_id`, `completed_at`, `duration_ms`, `error` |
| `stage_skipped` | Stage not applicable | `stage_id`, `reason` |
| `experiment_completed` | All stages done | `total_duration_ms`, `stage_timings`, `final_status` |
| `experiment_failed` | Any stage fails | `failed_stage`, `error`, `partial_results` |

### 5.3 Event Ordering Guarantees

- Events emitted in strict sequence order (monotonic `sequence` field)
- SSE delivers events in order
- Reconnection: client sends `last_event_id` (Last-Event-ID header), server replays missed events
- Event buffer: last 1000 events per experiment retained in memory

---

## 6. Data Contracts

### 6.1 Shared Types (Frontend ↔ Backend)

```typescript
// Frontend types (TypeScript) mirror backend Pydantic models

type TaskType = "classification" | "regression" | "clustering";

type HyperParameter = 
  | { type: "int"; name: string; display_name: string; default: number; min: number; max: number; step: number; help_text: string }
  | { type: "float"; name: string; display_name: string; default: number; min: number; max: number; step: number; help_text: string }
  | { type: "select"; name: string; display_name: string; default: string | number | boolean; options: (string | number | boolean)[]; help_text: string };

interface ModelInfo {
  id: string;
  name: string;
  task_type: TaskType;
  description: string;
  how_it_works: string;
  when_to_use: string;
  limitations: string;
  hyperparameters: HyperParameter[];
  supports_probability: boolean;
  supports_decision_boundary: boolean;
  supports_feature_importance: boolean;
  default_params: Record<string, unknown>;
}

interface DatasetInfo {
  id: string;
  name: string;
  task_type: TaskType;
  description: string;
  n_samples: number;
  n_features: number;
  feature_names: string[];
  target_name: string | null;
  target_names: string[] | null;
}

interface ExperimentConfig {
  dataset_id: string;
  model_id: string;
  hyperparameters: Record<string, unknown>;
  test_size: number;
  random_state: number;
  scale_features: boolean;
  scaler_type: "standard" | "minmax";
  viz_features: [number, number] | null;
}

interface StageTiming {
  stage_id: string;
  stage_name: string;
  duration_ms: number;
  status: "completed" | "failed" | "skipped";
}
```

### 6.2 Metrics Types

```typescript
interface ClassificationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  confusion_matrix: number[][];
  per_class: Record<string, {precision: number; recall: number; f1: number; support: number}>;
}

interface RegressionMetrics {
  mae: number;
  rmse: number;
  r2: number;
}

interface ClusteringMetrics {
  inertia: number;
  silhouette: number | null;
  n_clusters: number;
  n_samples: number;
}
```

### 6.3 Visualization Data Types

```typescript
interface DecisionBoundaryData {
  xx: number[][];      // meshgrid X
  yy: number[][];      // meshgrid Y
  Z: number[][];       // predictions on grid
  X_test_2d: number[][];  // test points [n, 2]
  y_test: number[];    // true labels
  feature_names: [string, string];
  target_names: string[] | null;
}

interface FeatureImportanceData {
  feature: string;
  importance: number;
}[]

interface ClusterData {
  X_2d: number[][];
  labels: number[];
  centroids: number[][] | null;
  feature_names: [string, string];
}
```

---

## 7. Testing Strategy

### 7.1 Backend Tests (pytest)

| Test Module | Coverage |
|-------------|----------|
| `test_health.py` | Health endpoint |
| `test_datasets_api.py` | Dataset list, detail, preview |
| `test_models_api.py` | Model list, detail, compatibility |
| `test_experiment_lifecycle.py` | Create, status, events, results, cancel |
| `test_workflow_events.py` | Event ordering, schema, SSE reconnection |
| `test_ml_correctness.py` | Regression vs baseline outputs |
| `test_validation.py` | Input validation, error responses |
| `test_concurrency.py` | Multiple simultaneous experiments |

**Key Test Principles:**
- Use deterministic seeds (42)
- Compare new backend outputs against saved baseline JSON
- Mock time for duration assertions
- Test SSE with `httpx` async client
- Test cancellation at safe boundaries (between stages)

### 7.2 Frontend Tests (Vitest + React Testing Library)

| Test Module | Coverage |
|-------------|----------|
| `App.test.tsx` | Routing, providers |
| `pages/*.test.tsx` | Page rendering, user interactions |
| `components/workflow/*.test.tsx` | Graph rendering, node selection, zoom/pan |
| `hooks/*.test.ts` | Custom hooks (useExperiment, useSSE) |
| `lib/api.test.ts` | API client, error handling |

**Key Test Principles:**
- Mock API with MSW (Mock Service Worker)
- Test keyboard navigation
- Test reduced-motion preference
- Test error boundaries

### 7.3 End-to-End Tests (Playwright)

| Scenario | Steps |
|----------|-------|
| Classification workflow | Load Iris → Select Logistic Regression → Train → Verify workflow events → Check metrics → View confusion matrix |
| Regression workflow | Load Diabetes → Select Linear Regression → Train → Verify metrics → View residual plot |
| Clustering workflow | Load Synthetic Blobs → Select K-Means → Train → Verify cluster plot + silhouette |
| Model comparison | Load Iris → Select 3 models → Compare → Verify bar/radar charts |
| Validation error | Load Iris → Select incompatible model (if possible) → Verify error handling |
| SSE reconnection | Start experiment → Disconnect network → Reconnect → Verify events resume |
| Mobile layout | Resize viewport → Verify navigation, graphs, charts usable |

### 7.4 Regression Testing (Critical)

After backend implementation, run **deterministic comparison**:

```python
# For each (dataset, model, config) combination:
# 1. Run old Streamlit backend (save outputs to baseline.json)
# 2. Run new FastAPI backend
# 3. Compare: metrics, predictions, visualizations data
# 4. Allow tolerance: 1e-6 for floats, exact for integers
# 5. Document any differences with root cause
```

Baseline combinations (15 total):
- 5 classification models × Iris + Breast Cancer = 10
- 1 regression model × Diabetes + Synthetic Regression = 2
- 1 clustering model × Synthetic Blobs = 1
- Model comparison: 3-classification on Iris = 1
- Model comparison: clustering = 1 (if multiple clustering models added)
- Edge cases: SVM calibration, KNN k>n_samples, KMeans k>samples = 2

---

## 8. Migration Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ML numerical differences | Medium | High | Regression test suite; document acceptable tolerances |
| SVM CalibratedClassifierCV behavior change | Low | High | Pin sklearn version; preserve exact pipeline structure |
| SSE connection drops | Medium | Medium | Auto-reconnect with Last-Event-ID; event replay buffer |
| React state ↔ Server state sync issues | Medium | Medium | TanStack Query for server state; local state only for UI |
| @xyflow/react performance with many nodes | Low | Low | Only 11 fixed nodes; virtualization not needed |
| Matplotlib → Recharts visual parity | Medium | Low | Accept differences; prioritize correctness |
| In-process task cleanup on crash | Medium | Medium | Try/finally for task cancellation; TTL cleanup job |
| Memory leak from abandoned experiments | Low | Medium | Experiment TTL (1 hour); max 50 concurrent |
| Frontend bundle size | Low | Medium | Code-split pages; lazy load WorkflowStudio |

---

## 9. Milestones

| Milestone | Deliverables | Duration | Dependencies |
|-----------|--------------|----------|--------------|
| **M1: Audit & Baseline** | `BASELINE_AUDIT.md`, `MIGRATION_PLAN.md`, test baseline | 1 day | — |
| **M2: Backend Refactoring** | FastAPI app, API endpoints, SSE, experiment executor, preserved ML core, backend tests | 3-4 days | M1 |
| **M3: React Foundation** | Vite+React+TS, Tailwind, design tokens, routing, API client, providers | 2 days | M1 |
| **M4: Core Pages** | Overview, Dataset Explorer, Model Explorer, Model Comparison (real API) | 3-4 days | M2, M3 |
| **M5: Workflow Studio** | Workflow graph, node details, timeline, SSE integration, controls | 3-4 days | M2, M4 |
| **M6: Reliability & Polish** | ML correctness fixes, accessibility, responsive, error handling, visual consistency | 2-3 days | M4, M5 |
| **M7: Testing & Validation** | All backend tests pass, frontend tests pass, E2E tests pass, regression comparison | 2 days | M6 |
| **M8: Documentation** | `ARCHITECTURE.md`, `API.md`, `WORKFLOW_EVENTS.md`, `TESTING.md`, `KNOWN_LIMITATIONS.md`, updated `README.md` | 1 day | M7 |

**Total Estimate:** 17-22 days

---

## 10. Acceptance Criteria

### 10.1 Functional

- [ ] React frontend builds and runs (`npm run dev`)
- [ ] FastAPI backend runs (`uvicorn main:app --reload`)
- [ ] Frontend communicates with backend via API
- [ ] All 7 algorithms available for compatible task types
- [ ] All 5 datasets load and display correctly
- [ ] Dataset Explorer: preview, statistics, distributions, correlations
- [ ] Model Explorer: config → train → results (metrics + visualizations)
- [ ] Model Comparison: 2+ models, shared split, bar + radar charts
- [ ] Classification: Accuracy, Precision, Recall, F1, Confusion Matrix, Decision Boundary
- [ ] Regression: MAE, RMSE, R², Actual vs Predicted, Residuals
- [ ] Clustering: Inertia, Silhouette, Cluster Plot, Silhouette Analysis
- [ ] Workflow Studio: 11 stages, real-time SSE updates, node details, timeline
- [ ] Validation failures stop pipeline and show error in workflow
- [ ] Experiment cancellation works at stage boundaries

### 10.2 Visual Design

- [ ] Light theme only (no dark mode)
- [ ] Antigravity-inspired: spacious, clean typography, restrained color
- [ ] Design tokens used consistently (no arbitrary colors/spacing)
- [ ] Responsive: desktop, tablet, mobile
- [ ] Workflow graph: pan, zoom, fit-to-view, horizontal scroll on mobile
- [ ] Charts readable on all screen sizes
- [ ] No cyberpunk/neon/hacker aesthetics

### 10.3 Quality

- [ ] All backend tests pass (`pytest -q`)
- [ ] All frontend tests pass (`npm test`)
- [ ] E2E tests pass (where implemented)
- [ ] Production build succeeds (`npm run build`)
- [ ] No console errors in browser
- [ ] No backend errors in logs
- [ ] Regression test: new backend matches old within tolerance
- [ ] Accessibility: keyboard nav, focus visible, semantic HTML, reduced-motion
- [ ] License and attribution preserved

### 10.4 Documentation

- [ ] `README.md` updated with new architecture diagram and run instructions
- [ ] `docs/audit/BASELINE_AUDIT.md` complete
- [ ] `docs/MIGRATION_PLAN.md` complete
- [ ] `docs/ARCHITECTURE.md` describes final system
- [ ] `docs/API.md` documents all endpoints with examples
- [ ] `docs/WORKFLOW_EVENTS.md` documents event schema
- [ ] `docs/TESTING.md` explains how to run all tests
- [ ] `docs/KNOWN_LIMITATIONS.md` lists unresolved issues

---

## 11. Rollback Strategy

### 11.1 Git Branching

```
main (Streamlit) ──────────────────────────────────►
  │
  └── migration/react-fastapi (this work)
        ├── M1-baseline
        ├── M2-backend
        ├── M3-frontend-foundation
        ├── M4-core-pages
        ├── M5-workflow-studio
        ├── M6-polish
        ├── M7-testing
        └── M8-docs
```

### 11.2 Rollback Triggers

- Critical ML correctness regression not fixable in 1 day
- SSE instability causing data loss
- Performance degradation >10x vs Streamlit
- Frontend framework incompatibility blocking progress

### 11.3 Rollback Procedure

1. Tag current migration branch: `git tag migration-checkpoint-<milestone>`
2. Switch to `main`: `git checkout main`
3. Streamlit app remains functional at `ca0b7f1`
4. Document blockers in `docs/KNOWN_LIMITATIONS.md`
5. Resume from last stable checkpoint

### 11.4 No Delete Policy

**Do not delete Streamlit code** until:
- React version passes all acceptance criteria
- Stakeholder sign-off obtained
- Migration branch merged to main with PR review

---

## 12. Implementation Order Details

### Milestone 1 — Audit and Baseline (Day 1)
- [x] Inspect repository structure
- [x] Run existing tests (64 passed)
- [x] Start Streamlit app, verify behavior
- [x] Document baseline audit
- [x] Create migration plan
- [ ] Save baseline ML outputs for regression testing

### Milestone 2 — Backend Refactoring (Days 2-5)
1. Create FastAPI project structure
2. Expose `ml_visualizer` as importable package
3. Implement `/api/health`, `/api/datasets`, `/api/models`
4. Implement experiment execution engine:
   - `ExperimentManager` class with in-memory store
   - Background task runner (asyncio)
   - Stage executor with event emission
5. Implement SSE endpoint with reconnection support
6. Add cancellation support (asyncio task cancellation at stage boundaries)
7. Write backend tests
8. Verify all 7 models train via API

### Milestone 3 — React Foundation (Days 6-7)
1. `npm create vite@latest frontend -- --template react-ts`
2. Install dependencies (Tailwind, shadcn/ui, TanStack Query, React Router, @xyflow/react, Recharts, React Hook Form, Zod, Lucide React, Motion)
3. Configure Tailwind with design tokens
4. Set up CSS custom properties for design system
5. Create `App.tsx`, `router.tsx`, `providers.tsx`
6. Build layout components (Header, Sidebar, Layout)
7. Create base UI components (Button, Card, Select, Slider, Checkbox, Tabs, Table, Badge)
8. Implement typed API client with error handling
9. Add TanStack Query configuration

### Milestone 4 — Core Pages (Days 8-11)
**Overview Page:**
- Hero section with product name, description
- Algorithm cards (3 task types)
- Quick-start steps
- Visual workflow summary (static diagram)

**Dataset Explorer:**
- Dataset selector (cards or dropdown)
- Metadata cards (task, samples, features, classes)
- Description
- Data preview table
- Feature statistics table
- Distributions chart (Recharts histogram)
- Correlation matrix (Recharts heatmap)
- Target distribution (classification)

**Model Explorer:**
- Dataset context (from previous page or selector)
- Compatible model selector
- Model info panel (description, how it works, when to use, limitations)
- Hyperparameter form (RHF + Zod, dynamic fields per param type)
- Training config (test size, random seed, scale toggle, scaler type)
- Visualization feature selectors
- Train button → calls POST /api/experiments
- Workflow Studio link / embedded mini-workflow
- Results display (metrics cards, charts, prediction table)
- All chart types as Recharts components

**Model Comparison:**
- Dataset context
- Multi-model selector (compatible only)
- Shared config (test size, seed, scale)
- Per-model hyperparameter forms (accordions)
- Compare button → multiple experiment calls or batch endpoint
- Results: metrics table, bar chart, radar chart
- Best model note with caveats

### Milestone 5 — Workflow Studio (Days 12-15)
1. **Workflow Graph** (`@xyflow/react`):
   - 11 fixed nodes in topological order
   - Edges showing flow (including validation failure branch)
   - Node status colors + icons (pending/running/completed/failed/skipped)
   - Pan, zoom, fit-to-view, mini-map
   - Responsive: horizontal scroll container on mobile

2. **Stage Detail Panel**:
   - Slides in on node click
   - Shows: name, description, status, timestamps, duration, input summary, output summary, warnings/errors, technical details

3. **Timeline**:
   - Chronological list of stages
   - Visual duration bars
   - Expandable event log per stage

4. **Flow Controls**:
   - Start Experiment (disabled if running)
   - Cancel Experiment (enabled during running)
   - Rerun Experiment (enabled after completion/failure)
   - Reset Workflow
   - Fit to Screen
   - Toggle Auto-follow (center on active node)
   - Toggle Event Log
   - Expand/Collapse All Details

5. **SSE Integration**:
   - Connect on experiment start
   - Update node statuses in real-time
   - Handle reconnection with Last-Event-ID
   - Buffer events during disconnect
   - Clean up on unmount

6. **Historical Replay** (stretch):
   - "Replay" button on completed experiment
   - Animates through recorded events at 2x speed

### Milestone 6 — Reliability & Polish (Days 16-18)
- Fix verified ML defects (H-02, M-01, M-02, M-04)
- Add input validation on frontend (Zod) and backend (Pydantic)
- Improve error messages (user-friendly, actionable)
- Accessibility audit:
  - Semantic HTML
  - ARIA labels on interactive elements
  - Focus visible styles
  - Color contrast (WCAG AA)
  - Reduced-motion support
  - Screen-reader announcements for workflow events
- Responsive testing: 320px, 768px, 1024px, 1440px
- Visual consistency: spacing, typography, colors per design tokens
- Performance: lazy-load WorkflowStudio, memoize chart components, virtualize large tables

### Milestone 7 — Testing & Validation (Days 19-20)
- Run all backend tests
- Run all frontend tests
- Run Playwright E2E tests
- Build production frontend (`npm run build`)
- Start both servers, verify end-to-end
- Run regression comparison (new vs old backend)
- Check browser console, backend logs
- Git status clean

### Milestone 8 — Documentation & Handoff (Day 21)
- Write all required docs
- Add real screenshots to `docs/screenshots/`
- Final report with executive summary, architecture, features, defects, testing, performance, run instructions, limitations, git changes

---

## 13. File Structure After Migration

```
algorithm-visualizer-ml/
├── .gitignore
├── LICENSE
├── README.md
├── backend/
│   ├── main.py                 # FastAPI entry point
│   ├── api/
│   │   ├── __init__.py
│   │   ├── health.py
│   │   ├── datasets.py
│   │   ├── models.py
│   │   ├── experiments.py
│   │   └── events.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── experiment_manager.py
│   │   ├── event_emitter.py
│   │   └── task_runner.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   ├── models.py
│   │   ├── experiments.py
│   │   └── workflow.py
│   ├── ml_visualizer/          # PRESERVED - copied or symlinked
│   ├── tests/
│   │   ├── test_health.py
│   │   ├── test_datasets_api.py
│   │   ├── test_models_api.py
│   │   ├── test_experiment_lifecycle.py
│   │   ├── test_workflow_events.py
│   │   ├── test_ml_correctness.py
│   │   └── test_concurrency.py
│   ├── requirements.txt
│   └── pyproject.toml
├── frontend/
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── index.html
│   ├── public/
│   └── src/                    # (structure from Section 3.1)
├── docs/
│   ├── audit/
│   │   └── BASELINE_AUDIT.md
│   ├── MIGRATION_PLAN.md
│   ├── ARCHITECTURE.md
│   ├── API.md
│   ├── WORKFLOW_EVENTS.md
│   ├── TESTING.md
│   ├── KNOWN_LIMITATIONS.md
│   └── screenshots/
└── legacy-streamlit/           # Preserved Streamlit app
    ├── app.py
    ├── ui/
    ├── ml_visualizer/          # Symlink to backend/ml_visualizer
    └── requirements.txt
```

---

## 14. Dependencies to Add

### Backend (backend/requirements.txt additions)
```
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
pydantic>=2.7.0
sse-starlette>=2.1.0
httpx>=0.27.0  # for testing
pytest-asyncio>=0.23.0
```

### Frontend (frontend/package.json additions)
```json
{
  "dependencies": {
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "react-router-dom": "^6.26.0",
    "@tanstack/react-query": "^5.51.0",
    "react-hook-form": "^7.52.0",
    "@hookform/resolvers": "^3.9.0",
    "zod": "^3.23.0",
    "@xyflow/react": "^12.3.0",
    "recharts": "^2.12.0",
    "lucide-react": "^0.436.0",
    "motion": "^11.3.0",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.5.0",
    "date-fns": "^3.6.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "typescript": "^5.5.0",
    "vite": "^5.4.0",
    "tailwindcss": "^3.4.0",
    "postcss": "^8.4.0",
    "autoprefixer": "^10.4.0",
    "vitest": "^2.0.0",
    "@testing-library/react": "^16.0.0",
    "@testing-library/jest-dom": "^6.4.0",
    "msw": "^2.3.0",
    "playwright": "^1.46.0"
  }
}
```

---

## 15. Open Questions for Stakeholder

1. **Deployment target:** Local development only, or containerized deployment?
2. **Persistence:** Is in-memory experiment history sufficient, or is SQLite needed?
3. **Concurrency:** Expected simultaneous users? (Affects in-process task limits)
4. **Authentication:** Any future multi-user plans? (Affects API design)
5. **Visualization fidelity:** Must Recharts match Matplotlib exactly, or is functional parity sufficient?
6. **Browser support:** IE11? (Affects polyfills, CSS features)

---

*End of Migration Plan*