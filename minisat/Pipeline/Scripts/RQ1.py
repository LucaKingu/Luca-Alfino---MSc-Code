import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')




# Configuration
DATASET = r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Pipeline\Data\dataset.csv"
FIGURES = r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Pipeline\Figures"

import os
os.makedirs(FIGURES, exist_ok=True)

df = pd.read_csv(DATASET)

FEATURES = ['mean_lbd', 'variance_lbd', 'lbd_trend', 'fraction_lbd2',
            'mean_clause_len', 'mean_dec_level', 'mean_props_per_conflict',
            'conflict_rate',
            'progress_at_k', 'learnts_per_conflict']

agg  = df[df['label'] == 'AGG']
cons = df[df['label'] == 'CONS']

colors = {'AGG': '#E74C3C', 'CONS': '#2E86AB'}



# PLOT 1: Box plots - all features side by side
import math
n_features = len(FEATURES)
n_cols = 4
n_rows = math.ceil(n_features / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3.5))
axes = axes.flatten()

# Hide unused panels
for i in range(n_features, len(axes)):
    axes[i].set_visible(False)
axes = axes.flatten()
fig.suptitle("Feature Distributions: AGG vs CONS\n(Early Search Dynamics at k=2000 Conflicts)",
             fontsize=14, fontweight='bold', y=1.01)

for i, feat in enumerate(FEATURES):
    ax = axes[i]

    # Use log scale for highly skewed features
    use_log = df[feat].max() / (df[feat].median() + 1e-9) > 50

    data_agg  = agg[feat].dropna()
    data_cons = cons[feat].dropna()

    if use_log:
        data_agg  = np.log1p(data_agg)
        data_cons = np.log1p(data_cons)
        ax.set_xlabel("log(1 + value)", fontsize=8)

    bp = ax.boxplot([data_agg, data_cons],
                    patch_artist=True,
                    labels=['AGG', 'CONS'],
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))

    bp['boxes'][0].set_facecolor('#E74C3C')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#2E86AB')
    bp['boxes'][1].set_alpha(0.7)

    # Mann-Whitney U test for significance
    stat, pval = stats.mannwhitneyu(agg[feat].dropna(),
                                     cons[feat].dropna(),
                                     alternative='two-sided')
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    title = f"{feat}\n(p={pval:.3f} {sig})"
    ax.set_title(title, fontsize=9, fontweight='bold' if pval < 0.05 else 'normal')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "eda_boxplots.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Boxplots saved.")



# PLOT 2: Top 4 most discriminative features - violin plots

# Pick 4 most significant features by Mann-Whitney p-value
pvals = {}
for feat in FEATURES:
    _, p = stats.mannwhitneyu(agg[feat].dropna(),
                               cons[feat].dropna(),
                               alternative='two-sided')
    pvals[feat] = p

top4 = sorted(pvals, key=pvals.get)[:4]
print(f"\nTop 4 most discriminative features: {top4}")

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle("Top 4 Discriminative Features: AGG vs CONS",
             fontsize=13, fontweight='bold')

for ax, feat in zip(axes, top4):
    plot_df = df[['label', feat]].copy()

    # Log transform if needed
    if df[feat].max() / (df[feat].median() + 1e-9) > 50:
        plot_df[feat] = np.log1p(plot_df[feat])
        ax.set_ylabel("log(1 + value)", fontsize=9)

    parts = ax.violinplot(
        [plot_df[plot_df['label']=='AGG'][feat].dropna().values,
         plot_df[plot_df['label']=='CONS'][feat].dropna().values],
        positions=[1, 2], showmedians=True, showextrema=True
    )

    for pc, color in zip(parts['bodies'], ['#E74C3C', '#2E86AB']):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)

    p = pvals[feat]
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    ax.set_title(f"{feat}\np={p:.4f} {sig}", fontsize=10, fontweight='bold')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['AGG', 'CONS'])
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "eda_violin_top4.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Violin plots saved.")




# PLOT 3: Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[FEATURES].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

import seaborn as sns
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, ax=ax,
            annot_kws={'size': 8},
            linewidths=0.5)
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "eda_correlation.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Correlation heatmap saved.")




# STATISTICAL SUMMARY TABLE
print("\n" + "="*65)
print("STATISTICAL SEPARABILITY SUMMARY")
print("="*65)
print(f"{'Feature':<30} {'AGG Median':>12} {'CONS Median':>12} {'p-value':>10} {'Sig':>5}")
print("-"*65)

for feat in sorted(pvals, key=pvals.get):
    p     = pvals[feat]
    sig   = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    a_med = agg[feat].median()
    c_med = cons[feat].median()
    print(f"{feat:<30} {a_med:>12.4f} {c_med:>12.4f} {p:>10.4f} {sig:>5}")