

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')
import os

RESULTS = r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Pipeline\Data\runtime_results.csv"
FIGURES = r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Pipeline\Figures"
TIMEOUT = 1200
os.makedirs(FIGURES, exist_ok=True)

df = pd.read_csv(RESULTS)
df['short_name'] = df['instance'].str.split('-').str[-1]\
                                 .str.replace('.cnf', '')
df['short_name'] = df['short_name'].str[:20]

# Exclude all-timeout instance
all_to   = ((df['status_agg']  == 'TIMEOUT') &
            (df['status_cons'] == 'TIMEOUT') &
            (df['status_none'] == 'TIMEOUT'))
df_valid = df[~all_to].copy()
print(f"Valid: {len(df_valid)}, Excluded: {len(df[all_to])}")




# PLOT 1: Adaptive vs Default MiniSAT - scatter
fig, ax = plt.subplots(figsize=(10, 8))

colors = df_valid['prediction_correct'].map(
    {True: '#2ECC71', False: '#E74C3C'})

ax.scatter(df_valid['time_none'],
           df_valid['time_predicted'],
           c=colors, s=100, alpha=0.85,
           edgecolors='black', linewidth=0.5, zorder=5)

ax.plot([0, TIMEOUT], [0, TIMEOUT],
        'k--', linewidth=1.2, alpha=0.6,
        label='Equal performance')

ax.fill_between([0, TIMEOUT], [0, TIMEOUT], [TIMEOUT, TIMEOUT],
                alpha=0.05, color='red')
ax.fill_between([0, TIMEOUT], [0, 0], [0, TIMEOUT],
                alpha=0.05, color='green')

green_patch = mpatches.Patch(color='#2ECC71', label='Correct prediction')
red_patch   = mpatches.Patch(color='#E74C3C', label='Wrong prediction')
ax.legend(handles=[green_patch, red_patch], fontsize=9, loc='upper left')

ax.set_xlabel("Default MiniSAT 2.2 Time (s)", fontsize=12)
ax.set_ylabel("Adaptive (Predicted) Policy Time (s)", fontsize=12)
ax.set_title("Adaptive Selector vs Default MiniSAT 2.2\n"
             "Points below diagonal = adaptive is faster",
             fontsize=12, fontweight='bold')
ax.set_xlim(0, 1300)
ax.set_ylim(0, 1300)
ax.text(200, 1100, 'Adaptive slower', color='red',   alpha=0.5, fontsize=9)
ax.text(700, 200,  'Adaptive faster', color='green', alpha=0.5, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "adaptive_vs_default_scatter.png"), dpi=150)
print("Plot 1 saved.")



# PLOT 2: Runtime savings/costs vs Default MiniSAT
df_valid['delta_vs_default'] = (df_valid['time_predicted'] -
                                 df_valid['time_none'])
df_sorted = df_valid.sort_values('delta_vs_default')

colors_bar = ['#2ECC71' if d <= 0 else '#E74C3C'
              for d in df_sorted['delta_vs_default']]

fig, ax = plt.subplots(figsize=(14, 7))
ax.barh(range(len(df_sorted)),
        df_sorted['delta_vs_default'],
        color=colors_bar, alpha=0.85,
        edgecolor='black', linewidth=0.3)

ax.axvline(x=0, color='black', linewidth=1.2)
ax.set_yticks(range(len(df_sorted)))
ax.set_yticklabels(df_sorted['short_name'], fontsize=8)
ax.set_xlabel("Time Difference (s): Adaptive − Default MiniSAT 2.2",
              fontsize=11)
ax.set_title("Runtime Savings (green) and Costs (red)\n"
             "Adaptive Selector vs Default MiniSAT 2.2 per Instance",
             fontsize=12, fontweight='bold')

green_patch = mpatches.Patch(color='#2ECC71',
                              label='Adaptive faster than default')
red_patch   = mpatches.Patch(color='#E74C3C',
                              label='Adaptive slower than default')
ax.legend(handles=[green_patch, red_patch], fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "savings_vs_default.png"), dpi=150)
print("Plot 2 saved.")





# PLOT 3: All 4 configurations mean time bar chart
config_keys = ['Default\nMiniSAT', 'AGG\nFixed',
               'CONS\nFixed', 'Adaptive\n(Predicted)']
config_vals = [df['time_none'].mean(),
               df['time_agg'].mean(),
               df['time_cons'].mean(),
               df['time_predicted'].mean()]

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.bar(config_keys, config_vals,
              color=['#95A5A6', '#E74C3C', '#2E86AB', '#2ECC71'],
              alpha=0.85, edgecolor='black', linewidth=0.5,
              width=0.5)

for bar, val in zip(bars, config_vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            f'{val:.1f}s', ha='center', va='bottom',
            fontsize=10, fontweight='bold')

bars[3].set_edgecolor('black')
bars[3].set_linewidth(2)

default_val   = config_vals[0]
default_label = f'Default baseline: {default_val:.1f}s'
ax.axhline(y=default_val, color='grey', linestyle='--',
           linewidth=1.2, alpha=0.7, label=default_label)

ax.set_ylabel("Mean Solve Time (s)", fontsize=12)
ax.set_title("Mean Solve Time Across All 24 Test Instances\n"
             "by Solver Configuration",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0, 750)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "mean_times_comparison.png"), dpi=150)
print("Plot 3 saved.")




# PLOT 4 REPLACEMENT: Top wins and losses vs Default MiniSAT
df_valid['delta_vs_default'] = (df_valid['time_predicted'] -
                                 df_valid['time_none'])

# Top 5 biggest savings (most negative delta)
top_wins   = df_valid.nsmallest(5, 'delta_vs_default')
# Top 5 biggest costs (most positive delta)
top_losses = df_valid.nlargest(5, 'delta_vs_default')

# Combine
df_highlight = pd.concat([top_wins, top_losses])
df_highlight = df_highlight.sort_values('delta_vs_default')

x     = np.arange(len(df_highlight))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 6))

bars_default  = ax.bar(x - width/2,
                        df_highlight['time_none'],
                        width, label='Default MiniSAT 2.2',
                        color='#95A5A6', alpha=0.85)
bars_adaptive = ax.bar(x + width/2,
                        df_highlight['time_predicted'],
                        width, label='Adaptive (Predicted)',
                        color='#2ECC71', alpha=0.85)

# Colour adaptive bars red where wrong
for i, (_, row) in enumerate(df_highlight.iterrows()):
    if not row['prediction_correct']:
        bars_adaptive[i].set_color('#E74C3C')
    marker = '✓' if row['prediction_correct'] else '✗'
    color  = 'darkgreen' if row['prediction_correct'] else 'red'
    ax.text(i + width/2, row['time_predicted'] + 15,
            marker, ha='center', fontsize=13,
            color=color, fontweight='bold')

# Add delta labels
for i, (_, row) in enumerate(df_highlight.iterrows()):
    delta = row['delta_vs_default']
    color = '#1a7a1a' if delta < 0 else '#cc0000'
    ax.text(i, max(row['time_none'],
                   row['time_predicted']) + 40,
            f'{delta:+.0f}s',
            ha='center', fontsize=9,
            color=color, fontweight='bold')

ax.axhline(y=TIMEOUT, color='black', linestyle='--',
           linewidth=1, alpha=0.5, label='Timeout (1200s)')
ax.axvline(x=4.5, color='black', linewidth=1.5,
           linestyle=':', alpha=0.7)
ax.text(1.5, 1280, 'Top 5 Savings', ha='center',
        fontsize=10, color='darkgreen', fontweight='bold')
ax.text(7.5, 1280, 'Top 5 Costs', ha='center',
        fontsize=10, color='darkred', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(df_highlight['short_name'],
                   rotation=30, ha='right', fontsize=9)
ax.set_ylabel("Solve Time (s)", fontsize=12)
ax.set_title("Top 5 Runtime Savings and Top 5 Runtime Costs\n"
             "Adaptive Selector vs Default MiniSAT 2.2",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0, 1380)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES,
            "top_wins_losses.png"), dpi=150)
print("Plot 4 replacement saved.")