"""
data/preprocessing.py — Data Exploration & Preprocessing Report
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

iris = load_iris()
X, y = iris.data, iris.target
fn   = iris.feature_names
cn   = iris.target_names

print("=" * 55)
print("  DATA EXPLORATION & PREPROCESSING")
print("=" * 55)
print(f"\nDataset shape  : {X.shape}")
print(f"Classes        : {list(cn)}")
print(f"Features       : {fn}")
print(f"Missing values : {np.isnan(X).sum()}")
print(f"\nClass distribution:")
for i, c in enumerate(cn):
    print(f"  {c:12s}: {(y==i).sum()} samples")

print("\nFeature statistics (raw):")
header = f"{'Feature':<28} {'Min':>6} {'Max':>6} {'Mean':>7} {'Std':>6}"
print(header)
print("-" * len(header))
for i, f in enumerate(fn):
    print(f"  {f:<26} {X[:,i].min():>6.2f} {X[:,i].max():>6.2f} {X[:,i].mean():>7.3f} {X[:,i].std():>6.3f}")

# ── Pair plot ───────────────────────────────
COLORS = ['#3266ad', '#1D9E75', '#D85A30']
fig, axes = plt.subplots(4, 4, figsize=(12, 10))
short = [f.replace(' (cm)', '').replace('sepal','sep').replace('petal','pet')
          .replace(' length','_len').replace(' width','_wid') for f in fn]

for i in range(4):
    for j in range(4):
        ax = axes[i][j]
        if i == j:
            for c in range(3):
                data = X[y == c, i]
                ax.hist(data, bins=12, color=COLORS[c], alpha=0.6, edgecolor='none')
            ax.set_ylabel(short[i], fontsize=8)
        else:
            for c in range(3):
                ax.scatter(X[y==c, j], X[y==c, i], c=COLORS[c], s=10, alpha=0.7)
        ax.tick_params(labelsize=6)
        if i == 3: ax.set_xlabel(short[j], fontsize=8)

patches = [plt.Rectangle((0,0),1,1, color=COLORS[c]) for c in range(3)]
fig.legend(patches, cn, loc='upper right', fontsize=9)
fig.suptitle("Pair Plot — Iris Features", fontweight='bold', fontsize=12)
fig.tight_layout(rect=[0, 0, 0.93, 0.97])
fig.savefig(f"{RESULTS_DIR}/08_pair_plot.png", dpi=130)
plt.close()

# ── Box plots ───────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(13, 4))
for i, (ax, f) in enumerate(zip(axes, fn)):
    data = [X[y==c, i] for c in range(3)]
    bp = ax.boxplot(data, patch_artist=True, medianprops={'color':'white','linewidth':2})
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
    ax.set_xticklabels(cn, rotation=20, fontsize=8)
    ax.set_title(f.replace(' (cm)',''), fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
fig.suptitle("Feature Distribution by Class (Box Plots)", fontweight='bold')
fig.tight_layout()
fig.savefig(f"{RESULTS_DIR}/09_box_plots.png", dpi=130)
plt.close()

# ── PCA projection ──────────────────────────
sc = StandardScaler()
Xs = sc.fit_transform(X)
pca = PCA(n_components=2)
Xp  = pca.fit_transform(Xs)

fig, ax = plt.subplots(figsize=(7, 5))
for c in range(3):
    ax.scatter(Xp[y==c, 0], Xp[y==c, 1], c=COLORS[c], s=50, alpha=0.8,
               label=cn[c], edgecolors='white', linewidths=0.4)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
ax.set_title("PCA Projection (2D) — Iris Dataset", fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{RESULTS_DIR}/10_pca_projection.png", dpi=150)
plt.close()

print("\n[✓] Preprocessing plots saved to results/")
