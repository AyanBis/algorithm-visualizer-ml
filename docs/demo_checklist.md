# 5-Minute Demo Checklist

## Pre-Demo Setup
- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] App running (`streamlit run app.py`)
- [ ] Browser open at `http://localhost:8501`

---

## 1. Dataset Exploration (1 minute)

### Goal
Show how to explore and understand datasets before modeling.

### Steps
1. **Navigate to Dataset Explorer** (sidebar)
2. **Select "Iris" dataset**
   - Observe: Task type = Classification, 150 samples, 4 features, 3 classes
   - Read description: "Classic dataset for flower species classification"
3. **View Data Preview** (first 10 rows with target)
4. **Expand "Feature Details"**
   - Show statistics: mean, std, min, max for each feature
5. **Click "Distributions" tab**
   - Point out: Class separation visible in petal features
6. **Click "Correlations" tab**
   - Note: High correlation between petal length/width

### Talking Points
- "Always explore data before modeling"
- "Feature distributions reveal class separability"
- "Correlations help identify redundant features"

---

## 2. Classification Demo (1.5 minutes)

### Goal
Train a classifier and interpret results.

### Steps
1. **Navigate to Model Explorer**
2. **Select "Logistic Regression"**
   - Read model description (expanded by default)
   - Note hyperparameters: C (regularization), max_iter, solver
3. **Keep defaults**, set:
   - Test Size = 0.2
   - Random Seed = 42
   - Scale Features = ✓
4. **Select visualization features**: "petal length (cm)" and "petal width (cm)"
5. **Click "Train Model"**
6. **Observe Results**:
   - Metric cards: Accuracy ~0.97, Precision/Recall/F1 ~0.97
   - Confusion Matrix: Nearly diagonal
   - Decision Boundary: Linear separation in petal space
   - Feature Importance: Petal features most important
   - Predictions Preview: Actual vs Predicted with probabilities

### Talking Points
- "Logistic regression learns linear decision boundaries"
- "Feature scaling is critical for regularization"
- "Decision boundary shows how model separates classes in 2D"
- "Probabilities provide confidence in predictions"

### Variation (30 sec)
- Change to **Random Forest**, retrain
- Note: Non-linear boundary, higher accuracy, feature importance differs
- Explain: "Ensemble of trees captures non-linear patterns"

---

## 3. Regression Demo (1 minute)

### Goal
Show regression workflow and evaluation.

### Steps
1. **Dataset Explorer** → Select "Diabetes"
   - Task type = Regression, 442 samples, 10 features
2. **Model Explorer** → Select "Linear Regression"
   - Only hyperparameter: Fit Intercept
3. **Train Model**
4. **Observe Results**:
   - MAE, RMSE, R² (~0.4-0.5)
   - **Actual vs Predicted**: Points near diagonal = good predictions
   - **Residuals vs Predicted**: Should be random (no pattern)
   - **Residual Distribution**: Approximately normal
   - Note: "R² can be negative if worse than horizontal line"

### Talking Points
- "Regression predicts continuous values, not classes"
- "Residuals reveal model biases and outliers"
- "R² = 1 means perfect fit, 0 means no better than mean"

---

## 4. Clustering Demo (1 minute)

### Goal
Show unsupervised learning without labels.

### Steps
1. **Dataset Explorer** → Select "Synthetic Blobs"
   - Task type = Clustering, no target column
   - Features: feature_1, feature_2 (2D for easy visualization)
2. **Model Explorer** → Select "K-Means"
   - Set n_clusters = 3 (matches data generation)
   - Train
3. **Observe Results**:
   - Inertia: Within-cluster sum of squares
   - Silhouette Score: ~0.7 (good separation)
   - **Cluster Plot**: Clear separation, centroids marked
   - **Silhouette Analysis** (expanded): Per-cluster silhouette widths
   - Cluster Assignments table

### Talking Points
- "No target labels — algorithm discovers structure"
- "k must be chosen (elbow method, silhouette)"
- "Cluster labels are arbitrary (0,1,2 ≠ true classes)"
- "Silhouette near 1 = well-separated clusters"

---

## 5. Model Comparison (1.5 minutes)

### Goal
Demonstrate fair comparison across algorithms.

### Steps
1. **Model Comparison** page
2. **Select 3 Classification Models**:
   - Logistic Regression
   - Decision Tree
   - Random Forest
3. **Keep shared config**:
   - Test Size = 0.2, Random Seed = 42, Scale = ✓
4. **Use default hyperparameters** for each
5. **Click "Compare Models"**
6. **Observe**:
   - **Metrics Table**: Accuracy, Precision, Recall, F1 for each
   - **Bar Chart**: Visual comparison on selected metric
   - **Radar Chart**: Multi-metric view
   - Best model highlighted

### Talking Points
- "Same train/test split for fair comparison"
- "Random Forest typically wins on accuracy"
- "But consider: interpretability, speed, overfitting risk"
- "No single 'best' model — depends on use case"

---

## Post-Demo

### Clean Up
- [ ] Stop Streamlit server (Ctrl+C)
- [ ] Deactivate virtual environment

### Known Incomplete Features
| Feature | Status | Notes |
|---------|--------|-------|
| Cross-validation | Not implemented | Single train/test split only |
| Custom data upload | Not implemented | Built-in datasets only |
| Hyperparameter search | Manual only | No grid/random search |
| Model export | Not implemented | No joblib/pickle save |
| ROC/PR curves | Not implemented | For classification |
| Learning curves | Not implemented | For training dynamics |

### Troubleshooting
- **Import errors**: Ensure `.venv` activated and `pip install -r requirements.txt` run
- **Port in use**: `streamlit run app.py --server.port 8502`
- **Slow SVM training**: Reduce dataset size or increase timeout
- **Memory issues**: Reduce `n_estimators` for Random Forest