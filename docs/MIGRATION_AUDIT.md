# Migration Audit - Algorithm Visualizer ML

**Date:** 2024-01-15  
**Repository:** https://github.com/AyanBis/algorithm-visualizer-ml  
**Current State:** Streamlit application with partial FastAPI/React migration artifacts

---

## 1. Existing Architecture

### Current Stack (Streamlit)
- **Frontend:** Streamlit (Python)
- **Backend:** Integrated in Streamlit (single process)
- **ML Engine:** scikit-learn (7 algorithms, 5 datasets)
- **Visualization:** Matplotlib (static images rendered in Streamlit)
- **State Management:** Streamlit session_state
- **Testing:** pytest (64 tests passing)

### Partial Migration Artifacts (Incomplete)
- **backend/**: FastAPI skeleton with partial API endpoints
- **frontend/**: React + TypeScript + Vite scaffold with partial pages
- **ml_visualizer/**: Core ML engine (reusable)

---

## 2. Existing ML Capabilities

### Algorithms (7)
| Algorithm | Task Type | Status |
|-----------|-----------|--------|
| Logistic Regression | Classification | ✅ Complete |
| Decision Tree | Classification | ✅ Complete |
| Random Forest | Classification | ✅ Complete |
| K-Nearest Neighbors | Classification | ✅ Complete |
| Support Vector Machine | Classification | ✅ Complete |
| Linear Regression | Regression | ✅ Complete |
| K-Means | Clustering | ✅ Complete |

### Datasets (5)
| Dataset | Task | Samples | Features |
|---------|------|---------|----------|
| Iris | Classification | 150 | 4 |
| Breast Cancer Wisconsin | Classification | 569 | 30 |
| Diabetes | Regression | 442 | 10 |
| Synthetic Regression | Regression | 200 | 5 |
| Synthetic Blobs | Clustering | 300 | 2 |

### Core ML Features
- ✅ Dataset loading & metadata
- ✅ Leakage-safe train/test split (scaler fit on train only)
- ✅ Preprocessing (StandardScaler, MinMaxScaler)
- ✅ Model training with hyperparameter configuration
- ✅ Prediction with probability estimates
- ✅ Feature importance extraction
- ✅ Decision boundary mesh generation
- ✅ Classification metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
- ✅ Regression metrics (MAE, RMSE, R²)
- ✅ Clustering metrics (Inertia, Silhouette)
- ✅ Model comparison with shared data splits
- ✅ Educational explanations for each algorithm
- ✅ Feature distributions, correlation matrix visualizations

---

## 3. Existing Backend (FastAPI) - Partial Implementation

### Implemented
- ✅ FastAPI app with CORS, lifespan management
- ✅ Dataset endpoints (list, detail, preview, statistics)
- ✅ Model endpoints (list, detail with hyperparameters)
- ✅ Experiment endpoints (create, status, events SSE, results, cancel)
- ✅ Experiment manager with in-memory storage + cleanup
- ✅ Event emitter with SSE streaming
- ✅ Experiment runner with 11-stage workflow
- ✅ Pydantic schemas for all API contracts
- ✅ Background task execution

### Missing/Incomplete
- ❌ Comparison endpoint (POST /api/v1/experiments/compare)
- ❌ Model compatibility validation in some endpoints
- ❌ Proper error handling for edge cases
- ❌ Input validation for all endpoints
- ❌ Request/response logging
- ❌ Rate limiting
- ❌ Authentication (not required per spec)

---

## 4. Existing Frontend (React) - Partial Implementation

### Implemented Pages
- ✅ Overview page with algorithm cards
- ✅ Dataset Explorer (list, preview, statistics, visualizations)
- ✅ Model Explorer (model selection, hyperparameters, training, results)
- ✅ Model Comparison (multi-model selection, shared config, results table/charts)
- ✅ Workflow Studio (interactive workflow diagram with @xyflow/react)

### Implemented Components
- ✅ Layout with sidebar navigation
- ✅ UI components (Button, Card, Select, Slider, Checkbox, Tabs, Table, Badge)
- ✅ API client with TanStack Query
- ✅ TypeScript types for all API contracts
- ✅ SSE hook for real-time workflow events
- ✅ Tailwind CSS + shadcn/ui components
- ✅ Recharts integration (partial)
- ✅ @xyflow/react for workflow diagram

### Missing/Incomplete
- ❌ Dashboard page (not implemented, only Overview)
- ❌ Experiment Details page (WorkflowStudio exists but not linked from comparison)
- ❌ Recharts visualizations (mostly placeholder implementations)
- ❌ Model comparison charts (bar/radar - placeholder only)
- ❌ Decision boundary visualization (placeholder)
- ❌ Confusion matrix visualization (placeholder)
- ❌ Residual plots, actual vs predicted (placeholder)
- ❌ Cluster visualization (placeholder)
- ❌ Silhouette plot (placeholder)
- ❌ Feature importance charts (placeholder)
- ❌ Progress indicators during training
- ❌ Error boundaries
- ❌ Loading skeletons
- ❌ Responsive design testing
- ❌ Accessibility (ARIA labels, keyboard navigation)
- ❌ Dark/light theme toggle

---

## 5. Components That Can Be Reused

### ML Engine (ml_visualizer/) - FULLY REUSABLE
All core ML logic is well-encapsulated and should be ported directly:
- `datasets.py` - Dataset loading & metadata
- `preprocessing.py` - Leakage-safe splitting & scaling
- `model_registry.py` - Model definitions, hyperparameters, capabilities
- `training.py` - Training, prediction, feature importance, decision boundaries
- `metrics.py` - All metrics computation + matplotlib plotting
- `visualizations.py` - Dataset overview, correlation matrix, comparison charts

### Backend (Partial Reuse)
- Schema definitions (Pydantic models) - reusable
- Experiment manager - reusable with minor fixes
- Event emitter - reusable
- Task runner - reusable with minor fixes
- API route structure - keep, complete missing endpoints

### Frontend (Partial Reuse)
- TypeScript types (api.ts) - reusable
- API client (api.ts) - reusable
- Hooks (useExperimentSSE, hooks in features/) - reusable
- UI components (Button, Card, Select, etc.) - reusable with shadcn/ui
- Layout component - reusable with modifications
- Tailwind config - reusable
- Type definitions (api.ts) - reusable

---

## 6. Components That Must Be Replaced

### Remove Completely
- Streamlit application (app.py, ui/, Streamlit dependencies)
- Streamlit-specific visualization code (visualizations.py display_figure)
- Streamlit session_state usage
- Streamlit-specific components (st.selectbox, st.button, etc.)

### Rewrite Completely
- Frontend pages (Dashboard, DatasetExplorer, ModelExplorer, ModelComparison, ExperimentDetails)
- Visualization components (use Recharts instead of matplotlib)
- Workflow diagram (use @xyflow/react properly with real-time updates)
- API integration (replace Streamlit direct calls with API client)

---

## 6. Existing Defects (from code review)

### Critical
1. **Silhouette score zero handling** - `if metrics.silhouette:` treats 0.0 as falsy (components.py:464, comparison.py:126)
2. **Clustering comparison data leakage** - Uses `X.values` (full dataset) instead of test split for silhouette
3. **Missing screenshots** - README references non-existent files in `docs/screenshots/`
4. **HTML injection risk** - `unsafe_allow_html=True` with dynamic content in `app.py` and `ui/components.py`

### High
1. **Clustering comparison wrong feature space** - Uses full dataset instead of test split
2. **Missing HTML escaping** - Dynamic content in `st.markdown` without escaping
3. **Missing comparison endpoint** - No POST /api/v1/experiments/compare

### Medium
1. README heading "##Screenshots" missing space
2. Empty `docs/screenshots/` directory
3. Missing GitHub Actions CI
4. Frontend placeholder visualizations (no Recharts implementation)

---

## 7. Migration Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| ML engine behavior changes | High | Comprehensive regression tests comparing old vs new outputs |
| Data leakage in migration | High | Reuse preprocessing.py exactly; add regression tests |
| Frontend-backend contract mismatch | Medium | Shared TypeScript/Pydantic schemas; contract tests |
| WebSocket/SSE reliability | Medium | Implement reconnection, fallback polling |
| Long-running training blocking | Medium | Background tasks with status polling/SSE |
| Visualization parity | Medium | Recharts ≠ Matplotlib; accept differences, focus on accuracy |
| React/TypeScript learning curve | Low | Use existing patterns, shadcn/ui components |

---

## 7. Final Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      React Frontend (Vite)                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │  Dashboard  │ │   Dataset   │ │   Model     │  Pages     │
│  │   Page      │ │  Explorer   │ │  Explorer   │            │
│  └─────────────┘ └─────────────┘ └─────────────┘            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │   Model     │ │ Experiment  │ │  Workflow   │            │
│  │  Comparison │ │  Details    │ │   Studio    │            │
│  └─────────────┘ └─────────────┘ └─────────────┘            │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Shared Components (shadcn/ui)               │ │
│  │  Charts (Recharts) │ Layout │ Forms │ Tables │ Charts   │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              API Client (TanStack Query)                 │ │
│  │  REST API │ SSE │ Cache │ Validation │ Error Handling   │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/JSON + SSE
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (Uvicorn)                  │
│  ┌──────────┐ ┌────────┐ ┌──────────┐ ┌────────────────┐    │
│  │ Datasets │ │ Models │ │Experiments│ │  Comparison   │    │
│  └──────────┘ └────────┘ └──────────┘ └────────────────┘    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              Experiment Manager + Event Emitter         │  │
│  └────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              Task Runner (11-stage workflow)            │  │
│  └────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                    ML Engine (ml_visualizer)             │  │
│  │  Datasets │ Preprocessing │ Models │ Training │ Metrics │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ML Engine (scikit-learn)                  │
│  7 Algorithms │ 5 Datasets │ Metrics │ Visualization Data  │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Final Target Stack

| Layer | Technology |
|-------|------------|
| Frontend Framework | React 18 + TypeScript + Vite |
| UI Library | shadcn/ui + Tailwind CSS |
| Routing | React Router v6 |
| State Management | TanStack Query (server) + React Context (UI) |
| Charts | Recharts |
| Diagrams | @xyflow/react |
| Forms | React Hook Form + Zod |
| API Client | TanStack Query + custom API client |
| Real-time | SSE (EventSource) |
| Backend | FastAPI + Uvicorn |
| ML Engine | scikit-learn (existing ml_visualizer) |
| Validation | Pydantic v2 (backend) + Zod (frontend) |
| Testing | pytest (backend), Vitest + React Testing Library (frontend) |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Code Quality | ruff, black, TypeScript strict, oxlint |

---

## 9. Migration Phase Summary

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Audit | ✅ Complete | This document |
| 2. Architecture | 📋 Planned | Documented above |
| 3. Backend | 🟡 Partial | ~60% complete; needs comparison endpoint, validation, tests |
| 4. ML Engine | ✅ Reusable | Zero changes needed |
| 5. Frontend | 🟡 Partial | ~40% complete; needs visualizations, pages, integration |
| 6. UI/UX | 🔴 Not Started | Needs full Recharts implementation, theme, accessibility |
| 7. Visualizations | 🔴 Not Started | Replace matplotlib with Recharts |
| 8. Integration | 🔴 Not Started | Frontend-backend connection incomplete |
| 7. Security | 🔴 Not Started | CORS, validation, error handling |
| 8. Testing | 🟡 Partial | Backend tests exist; frontend tests missing |
| 8. Docker | 🔴 Not Started | docker-compose.yml, Dockerfiles needed |
| 9. CI/CD | 🟡 Partial | GitHub Actions exists but needs update |
| 13. Streamlit Removal | 🔴 Not Started | Remove after React+FastAPI verified |
| 14. Documentation | 🟡 Partial | README exists; need ARCHITECTURE.md, API_REFERENCE.md, etc. |
| 15. Final Verification | 🔴 Not Started | Full integration test needed |

---

**Next Step:** Begin Phase 3 - Complete Backend Implementation