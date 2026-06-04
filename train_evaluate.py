"""
models/train_evaluate.py — Hyperparameter Tuning & Extended Evaluation
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

iris = load_iris()
X, y = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler().fit(X_tr)
Xs_tr = sc.transform(X_tr)
Xs_te = sc.transform(X_te)

print("=" * 55)
print("  HYPERPARAMETER TUNING")
print("=" * 55)

# ── KNN grid search ────────────────────────
knn_params = {'n_neighbors': list(range(1, 21)),
              'weights': ['uniform', 'distance'],
              'metric': ['euclidean', 'manhattan']}
knn_gs = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring='accuracy', n_jobs=-1)
knn_gs.fit(Xs_tr, y_tr)
print(f"\nKNN best params : {knn_gs.best_params_}")
print(f"KNN best CV acc : {knn_gs.best_score_:.4f}")

# ── Decision Tree grid search ──────────────
dt_params = {'max_depth': [2, 3, 4, 5, 6, 8, None],
             'min_samples_split': [2, 5, 10],
             'criterion': ['gini', 'entropy']}
dt_gs = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5, scoring='accuracy', n_jobs=-1)
dt_gs.fit(Xs_tr, y_tr)
print(f"\nDT best params  : {dt_gs.best_params_}")
print(f"DT best CV acc  : {dt_gs.best_score_:.4f}")

# Naïve Bayes has var_smoothing as the key param
nb_params = {'var_smoothing': np.logspace(-12, 0, 20)}
nb_gs = GridSearchCV(GaussianNB(), nb_params, cv=5, scoring='accuracy', n_jobs=-1)
nb_gs.fit(Xs_tr, y_tr)
print(f"\nNB best params  : {nb_gs.best_params_}")
print(f"NB best CV acc  : {nb_gs.best_score_:.4f}")

# ── Learning curves ────────────────────────
COLORS = ['#3266ad', '#1D9E75', '#D85A30']
models_tuned = [
    ("KNN (tuned)",          knn_gs.best_estimator_),
    ("Decision Tree (tuned)", dt_gs.best_estimator_),
    ("Naïve Bayes (tuned)",   nb_gs.best_estimator_),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, (name, model), color in zip(axes, models_tuned, COLORS):
    sizes, tr_scores, cv_scores = learning_curve(
        model, Xs_tr, y_tr, cv=5, train_sizes=np.linspace(0.2, 1.0, 10),
        scoring='accuracy', n_jobs=-1)
    tr_mean, tr_std = tr_scores.mean(1), tr_scores.std(1)
    cv_mean, cv_std = cv_scores.mean(1), cv_scores.std(1)
    ax.plot(sizes, tr_mean, 'o-', color=color, label='Train')
    ax.fill_between(sizes, tr_mean-tr_std, tr_mean+tr_std, alpha=0.15, color=color)
    ax.plot(sizes, cv_mean, 's--', color='gray', label='CV')
    ax.fill_between(sizes, cv_mean-cv_std, cv_mean+cv_std, alpha=0.1, color='gray')
    ax.set_title(name, fontweight='bold', fontsize=10)
    ax.set_xlabel("Training samples")
    ax.set_ylim(0.7, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel("Accuracy")
fig.suptitle("Learning Curves (tuned models)", fontweight='bold')
fig.tight_layout()
fig.savefig(f"{RESULTS_DIR}/11_learning_curves.png", dpi=150)
plt.close()

# ── Final comparison: default vs tuned ─────
default_models = [
    KNeighborsClassifier(n_neighbors=5),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    GaussianNB(),
]
names_short = ['KNN', 'Decision Tree', 'Naïve Bayes']
default_accs = [accuracy_score(y_te, m.fit(Xs_tr,y_tr).predict(Xs_te)) for m in default_models]
tuned_accs   = [accuracy_score(y_te, m.predict(Xs_te)) for _, m in models_tuned]

x = np.arange(3)
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x-0.2, default_accs, 0.35, label='Default', color=[c+'99' for c in ['#3266ad','#1D9E75','#D85A30']], zorder=3)
ax.bar(x+0.2, tuned_accs,   0.35, label='Tuned',   color=COLORS, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(names_short)
ax.set_ylim(0.85, 1.04)
ax.set_ylabel("Test Accuracy")
ax.set_title("Default vs Tuned Accuracy", fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3, zorder=0)
for i,(d,t) in enumerate(zip(default_accs, tuned_accs)):
    ax.text(i-0.2, d+0.003, f"{d:.3f}", ha='center', fontsize=8)
    ax.text(i+0.2, t+0.003, f"{t:.3f}", ha='center', fontsize=8)
fig.tight_layout()
fig.savefig(f"{RESULTS_DIR}/12_default_vs_tuned.png", dpi=150)
plt.close()

print("\n[✓] Tuning & learning curve plots saved.")
