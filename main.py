"""
ML Algorithm Comparison: KNN, Decision Tree, Naïve Bayes
Dataset: Iris (UCI) — 150 samples, 4 features, 3 classes
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
import warnings, os
warnings.filterwarnings('ignore')

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────────
print("=" * 60)
print("  ML ALGORITHM COMPARISON — IRIS DATASET")
print("=" * 60)

iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names   = iris.target_names

print(f"\n[DATA] Shape: {X.shape}  |  Classes: {list(class_names)}")
print(f"[DATA] Features: {feature_names}")

# Train/test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardise (fit on train only)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"[SPLIT] Train: {len(X_train)}  |  Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 2. TRAIN MODELS
# ─────────────────────────────────────────────
models = {
    "KNN (k=5)":       KNeighborsClassifier(n_neighbors=5),
    "Decision Tree":   DecisionTreeClassifier(max_depth=5, random_state=42),
    "Naïve Bayes":     GaussianNB(),
}

results = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='accuracy')

    results[name] = {
        "model":    model,
        "y_pred":   y_pred,
        "acc":      accuracy_score(y_test, y_pred),
        "prec":     precision_score(y_test, y_pred, average='macro'),
        "rec":      recall_score(y_test, y_pred, average='macro'),
        "f1":       f1_score(y_test, y_pred, average='macro'),
        "cm":       confusion_matrix(y_test, y_pred),
        "cv_mean":  cv_scores.mean(),
        "cv_std":   cv_scores.std(),
        "report":   classification_report(y_test, y_pred, target_names=class_names),
    }
    print(f"\n[{name}]  Acc={results[name]['acc']:.4f}  |  F1={results[name]['f1']:.4f}  |  5-CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# 3. PLOTS
# ─────────────────────────────────────────────
COLORS   = ['#3266ad', '#1D9E75', '#D85A30']
CM_CMAP  = plt.cm.Blues

# --- 3a. Accuracy / Precision / Recall / F1 bar chart
fig, ax = plt.subplots(figsize=(10, 5))
metrics   = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
algo_names = list(results.keys())
n_metrics  = len(metrics)
n_algos    = len(algo_names)
x = np.arange(n_metrics)
width = 0.22

for i, (name, r) in enumerate(results.items()):
    vals = [r['acc'], r['prec'], r['rec'], r['f1']]
    bars = ax.bar(x + i*width, vals, width, label=name, color=COLORS[i], zorder=3)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                f"{b.get_height():.2f}", ha='center', va='bottom', fontsize=8)

ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.set_ylim(0.7, 1.08)
ax.set_ylabel("Score")
ax.set_title("Performance Metrics — KNN vs Decision Tree vs Naïve Bayes", fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3, zorder=0)
fig.tight_layout()
fig.savefig(f"{RESULTS_DIR}/01_metrics_comparison.png", dpi=150)
plt.close()

# --- 3b. Confusion matrices (1 row, 3 cols)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, r) in zip(axes, results.items()):
    cm = r['cm']
    im = ax.imshow(cm, interpolation='nearest', cmap=CM_CMAP)
    ax.set_title(name, fontweight='bold')
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks); ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(tick_marks); ax.set_yticklabels(class_names, fontsize=9)
    thresh = cm.max() / 2
    for ii in range(cm.shape[0]):
        for jj in range(cm.shape[1]):
            ax.text(jj, ii, str(cm[ii, jj]), ha='center', va='center',
                    color='white' if cm[ii, jj] > thresh else 'black', fontsize=14)
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("Confusion Matrices", fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(f"{RESULTS_DIR}/02_confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.close()

# --- 3c. Cross-validation scores (box-style error bars)
fig, ax = plt.subplots(figsize=(7, 4))
names = list(results.keys())
cv_means = [results[n]['cv_mean'] for n in names]
cv_stds  = [results[n]['cv_std']  for n in names]
bars = ax.bar(names, cv_means, color=COLORS, zorder=3, width=0.45)
ax.errorbar(names, cv_means, yerr=cv_stds, fmt='none', color='black', capsize=6, linewidth=2)
for b, v in zip(bars, cv_means):
    ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.4f}", ha='center', fontsize=10)
ax.set_ylim(0.85, 1.02)
ax.set_ylabel("5-Fold CV Accuracy")
ax.set_title("5-Fold Cross-Validation Accuracy (mean ± std)", fontweight='bold')
ax.grid(axis='y', alpha=0.3, zorder=0)
fig.tight_layout()
fig.savefig(f"{RESULTS_DIR}/03_cross_validation.png", dpi=150)
plt.close()

# --- 3d. Feature importance (DT) + correlation heatmap
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Feature importance from Decision Tree
dt = results["Decision Tree"]["model"]
importances = dt.feature_importances_
axes[0].barh(feature_names, importances, color='#1D9E75')
axes[0].set_xlabel("Importance")
axes[0].set_title("Decision Tree — Feature Importances", fontweight='bold')
for i, v in enumerate(importances):
    axes[0].text(v+0.005, i, f"{v:.3f}", va='center', fontsize=9)

# Feature correlation heatmap
corr = np.corrcoef(X.T)
im = axes[1].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
axes[1].set_xticks(range(4)); axes[1].set_xticklabels([f.replace(' (cm)','') for f in feature_names], rotation=30, ha='right', fontsize=8)
axes[1].set_yticks(range(4)); axes[1].set_yticklabels([f.replace(' (cm)','') for f in feature_names], fontsize=8)
for i in range(4):
    for j in range(4):
        axes[1].text(j, i, f"{corr[i,j]:.2f}", ha='center', va='center', fontsize=9)
axes[1].set_title("Feature Correlation Heatmap", fontweight='bold')
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

fig.tight_layout()
fig.savefig(f"{RESULTS_DIR}/04_feature_analysis.png", dpi=150)
plt.close()

# --- 3e. Decision tree visualisation
fig, ax = plt.subplots(figsize=(16, 7))
plot_tree(dt, feature_names=feature_names, class_names=class_names,
          filled=True, rounded=True, ax=ax, fontsize=9, max_depth=3)
ax.set_title("Decision Tree Structure (depth ≤ 3 shown)", fontweight='bold')
fig.tight_layout()
fig.savefig(f"{RESULTS_DIR}/05_decision_tree_plot.png", dpi=120)
plt.close()

# --- 3f. KNN — effect of k on accuracy
k_range = range(1, 21)
k_accs  = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_s, y_train)
    k_accs.append(accuracy_score(y_test, knn.predict(X_test_s)))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(k_range, k_accs, marker='o', color='#3266ad', linewidth=2)
ax.axvline(5, color='red', linestyle='--', alpha=0.5, label='k=5 (selected)')
ax.set_xlabel("Number of Neighbors (k)")
ax.set_ylabel("Test Accuracy")
ax.set_title("KNN — Accuracy vs Number of Neighbors", fontweight='bold')
ax.set_xticks(list(k_range))
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(f"{RESULTS_DIR}/06_knn_k_tuning.png", dpi=150)
plt.close()

# --- 3g. Per-class F1 grouped bar chart
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(class_names))
w = 0.22
for i, (name, r) in enumerate(results.items()):
    f1_per = f1_score(y_test, r['y_pred'], average=None)
    ax.bar(x + i*w, f1_per, w, label=name, color=COLORS[i], zorder=3)
ax.set_xticks(x + w); ax.set_xticklabels(class_names)
ax.set_ylim(0, 1.1); ax.set_ylabel("F1 Score")
ax.set_title("Per-Class F1 Score Comparison", fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3, zorder=0)
fig.tight_layout()
fig.savefig(f"{RESULTS_DIR}/07_per_class_f1.png", dpi=150)
plt.close()

# ─────────────────────────────────────────────
# 4. PRINT FULL REPORTS
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  FULL CLASSIFICATION REPORTS")
print("="*60)
for name, r in results.items():
    print(f"\n── {name} ──")
    print(r['report'])

print("\n[✓] All plots saved to:", RESULTS_DIR)
print("[✓] Done.\n")
