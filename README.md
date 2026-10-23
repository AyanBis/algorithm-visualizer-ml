# Algorithm Visualizer ML 🤖

An interactive web application for learning and exploring machine learning algorithms. Built with Streamlit, scikit-learn, and matplotlib.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Features

- **Dataset Explorer** — Browse built-in datasets, inspect features, targets, distributions, and correlations
- **Model Explorer** — Train and evaluate 7 ML algorithms with configurable hyperparameters
- **Rich Visualizations** — Decision boundaries, confusion matrices, regression plots, cluster visualizations
- **Model Comparison** — Fairly compare multiple algorithms on the same data split
- **Educational Explanations** — Learn how each algorithm works, when to use it, and its limitations

## 📚 Supported Algorithms

### Classification
| Algorithm | Description |
|-----------|-------------|
| **Logistic Regression** | Linear classifier with probability estimates |
| **Decision Tree** | Interpretable rule-based splits |
| **Random Forest** | Ensemble of trees with majority voting |
| **K-Nearest Neighbors** | Instance-based classification |
| **Support Vector Machine** | Maximum margin separator (linear & RBF kernels) |

### Regression
| Algorithm | Description |
|-----------|-------------|
| **Linear Regression** | Linear relationship modeling |

### Clustering
| Algorithm | Description |
|-----------|-------------|
| **K-Means** | Centroid-based partitioning |

## 📊 Supported Datasets

| Dataset | Task | Samples | Features | Description |
|---------|------|---------|----------|-------------|
| **Iris** | Classification | 150 | 4 | Classic flower species classification |
| **Breast Cancer Wisconsin** | Classification | 569 | 30 | Binary cancer diagnosis |
| **Diabetes** | Regression | 442 | 10 | Disease progression prediction |
| **Synthetic Regression** | Regression | 200 | 5 | Generated regression data |
| **Synthetic Blobs** | Clustering | 300 | 2 | Generated cluster data |

## 🛠 Technology Stack

- **Python** 3.10+
- **Streamlit** — Web application framework
- **scikit-learn** — Machine learning algorithms
- **pandas** — Data manipulation
- **NumPy** — Numerical computing
- **matplotlib** — Visualizations
- **pytest** — Testing

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/AyanBis/algorithm-visualizer-ml.git
cd algorithm-visualizer-ml

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## ▶️ Running the Application

```bash
# Activate virtual environment
source .venv/bin/activate

# Launch Streamlit app
streamlit run app.py
```

The application will open at `http://localhost:8501`.

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest -q

# Run with coverage
python -m pytest --cov=ml_visualizer --cov=ui
```

## 🏗 Architecture

```
algorithm-visualizer-ml/
├── app.py                    # Streamlit entry point
├── requirements.txt          # Dependencies
├── .gitignore
├── README.md
├── ml_visualizer/           # Core ML modules
│   ├── __init__.py
│   ├── datasets.py          # Dataset loading & metadata
│   ├── preprocessing.py     # Train/test split & scaling
│   ├── model_registry.py    # Model definitions & hyperparameters
│   ├── training.py          # Training & prediction
│   ├── metrics.py           # Evaluation metrics
│   └── visualizations.py    # Plotting utilities
├── ui/                      # Streamlit UI components
│   ├── __init__.py
│   ├── components.py        # Page renderers
│   └── comparison.py        # Model comparison utilities
├── tests/                   # Unit & integration tests
│   ├── test_datasets.py
│   ├── test_models.py
│   ├── test_metrics.py
│   └── test_app_flow.py
└── docs/
    └── demo_checklist.md    # 5-minute demo guide
```

### Module Responsibilities

- **datasets.py** — Loads built-in sklearn datasets, provides metadata (task type, feature names, descriptions)
- **preprocessing.py** — Leakage-safe train/test splitting, feature scaling with DataFrame preservation
- **model_registry.py** — Defines all 7 models with hyperparameters, descriptions, and capabilities
- **training.py** — Model training, prediction, feature importance extraction, decision boundary generation
- **metrics.py** — Classification (accuracy, precision, recall, F1, confusion matrix), regression (MAE, RMSE, R²), clustering (inertia, silhouette)
- **visualizations.py** — Matplotlib-based charts for Streamlit display
- **components.py** — Streamlit page renderers (Overview, Dataset Explorer, Model Explorer, Model Comparison)
- **comparison.py** — Fair model comparison with shared data splits

## 🎓 Example Workflows

### 1. Classification with Iris
1. Go to **Dataset Explorer** → Select "Iris"
2. Go to **Model Explorer** → Select "Logistic Regression"
3. Adjust C (regularization) if desired
4. Click **Train Model**
5. View accuracy, confusion matrix, and decision boundary

### 2. Regression with Diabetes
1. Go to **Dataset Explorer** → Select "Diabetes"
2. Go to **Model Explorer** → Select "Linear Regression"
3. Click **Train Model**
4. View MAE, RMSE, R², actual vs predicted, and residual plots

### 3. Clustering with Synthetic Blobs
1. Go to **Dataset Explorer** → Select "Synthetic Blobs"
2. Go to **Model Explorer** → Select "K-Means"
3. Set number of clusters (k)
4. Click **Train Model**
5. View cluster assignments, centroids, and silhouette analysis

### 4. Model Comparison
1. Select a dataset in **Dataset Explorer**
2. Go to **Model Comparison**
3. Select 2+ models of the same task type
4. Click **Compare Models**
5. View metric tables, bar charts, and radar plots

## ⚠️ Limitations

- **Educational focus** — Not optimized for production ML pipelines
- **Built-in datasets only** — No custom data upload (yet)
- **No cross-validation** — Single train/test split for simplicity
- **Limited hyperparameter tuning** — Manual configuration only
- **2D visualizations** — Decision boundaries only for 2 selected features
- **No model persistence** — Models not saved between sessions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `python -m pytest -q`
5. Commit: `git commit -m "Add feature"`
6. Push: `git push origin feature-name`
7. Open a Pull Request

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Attribution

Inspired by [Algo-Visualizer](https://github.com/Ishwarpatra/Algo-Visualizer) by Ishwar Patra.

Built for educational purposes to help students and practitioners understand ML algorithms through interactive visualization.