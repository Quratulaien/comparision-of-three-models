# ML Algorithm Comparison Assignment
### KNN · Decision Tree · Naïve Bayes on the Iris Dataset

---

## Overview

This project implements and compares three classical supervised machine-learning classifiers on the **Iris flower dataset** (UCI):

| Algorithm | Strategy | Key Hyperparameter |
|---|---|---|
| K-Nearest Neighbors | Instance-based, distance voting | k = 5 neighbors |
| Decision Tree | Recursive binary splitting (Gini) | max_depth = 5 |
| Naïve Bayes (Gaussian) | Probabilistic, feature independence | var_smoothing |

**Dataset** — 150 samples · 4 features · 3 classes (Setosa, Versicolor, Virginica)

---

## Project Structure

```
ml_assignment/
├── main.py                  ← Core training, evaluation & all plots
├── run_all.py               ← Run everything in one command
├── requirements.txt
├── data/
│   └── preprocessing.py     ← EDA, pair plots, PCA, box plots
├── models/
│   └── train_evaluate.py    ← GridSearchCV tuning + learning curves
├── ui/
│   └── app.py               ← Interactive Tkinter GUI
└── results/                 ← All generated plots (created on run)
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run everything
python run_all.py

# 3. Launch the interactive UI
python ui/app.py
```

---

## Pipeline

### 1. Data Preprocessing (`data/preprocessing.py`)
- Load Iris via `sklearn.datasets.load_iris`
- Summary statistics (min, max, mean, std) per feature
- Missing value check (none)
- Visualisations: pair plots, box plots, PCA 2D projection

### 2. Feature Selection
All 4 features retained (sepal/petal length & width). Decision tree feature importances confirm **petal length** and **petal width** dominate (~95% combined importance). `StandardScaler` applied (fit on train, transform test).

### 3. Train/Test Split
- 80 % training, 20 % test, **stratified** by class
- Fixed `random_state=42` for reproducibility

### 4. Model Training (`main.py`)
- KNN with k=5, Euclidean distance
- Decision Tree with max_depth=5, Gini criterion
- Gaussian Naïve Bayes with default var_smoothing

### 5. Hyperparameter Tuning (`models/train_evaluate.py`)
- `GridSearchCV` with 5-fold cross-validation
- KNN: k ∈ [1,20] × weights × metric
- DT: max_depth × min_samples_split × criterion
- NB: var_smoothing over 20 log-spaced values

### 6. Evaluation Metrics
Per model, per class and macro-averaged:
- **Accuracy** — overall correct predictions
- **Precision** — TP / (TP + FP)
- **Recall** — TP / (TP + FN)
- **F1 Score** — harmonic mean of precision & recall
- **Confusion Matrix** — per-class breakdown
- **5-fold CV score** — generalisation estimate

---

## Generated Plots

| File | Content |
|---|---|
| `01_metrics_comparison.png` | Accuracy / Precision / Recall / F1 bar chart |
| `02_confusion_matrices.png` | 3-panel confusion matrices |
| `03_cross_validation.png` | 5-fold CV mean ± std |
| `04_feature_analysis.png` | DT feature importances + correlation heatmap |
| `05_decision_tree_plot.png` | Decision tree structure (depth ≤ 3) |
| `06_knn_k_tuning.png` | Accuracy vs k (1–20) |
| `07_per_class_f1.png` | Per-class F1 grouped bars |
| `08_pair_plot.png` | Feature pair plot |
| `09_box_plots.png` | Box plots per class |
| `10_pca_projection.png` | PCA 2D scatter |
| `11_learning_curves.png` | Learning curves (tuned models) |
| `12_default_vs_tuned.png` | Default vs tuned accuracy |

---

## Interactive UI (`ui/app.py`)

- Sliders for all 4 Iris features with live value display
- Algorithm selector (radio buttons): KNN / Decision Tree / Naïve Bayes
- **Classify** button → shows predicted class, confidence, probability bars
- All 3 models predict simultaneously for comparison
- Quick preset buttons for representative Setosa / Versicolor / Virginica samples
- Live accuracy badges for all three models

---

## Key Findings

| Metric | KNN (k=5) | Decision Tree | Naïve Bayes |
|---|---|---|---|
| Test Accuracy | ~97% | ~97% | ~93% |
| Macro F1 | ~0.97 | ~0.97 | ~0.93 |
| Setosa F1 | 1.00 | 1.00 | 1.00 |
| Train time | fast | fast | very fast |
| Interpretable | ✗ | ✓ | partial |

- **Setosa** is linearly separable; all models achieve 100 % on it.
- **Versicolor / Virginica** overlap slightly; NB makes occasional errors there.
- **KNN and DT** are comparable; DT has the advantage of interpretability.
- **Naïve Bayes** is fastest and simplest, slightly lower accuracy due to feature-independence assumption (petal features are highly correlated).

---

*Submitted for: Machine Learning Algorithms Assignment*
*Dataset: UCI Iris · sklearn 1.3+ · Python 3.9+*
