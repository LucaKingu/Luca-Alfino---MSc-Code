import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv(r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Pipeline\Data\runtime_results.csv")

# Exclude all-timeout instance
df = df[~((df['status_agg']=='TIMEOUT') &
          (df['status_cons']=='TIMEOUT') &
          (df['status_none']=='TIMEOUT'))]

# Short names for x axis
df['short_name'] = df['instance'].str[:25]

x     = np.arange(len(df))
width = 0.2

fig, ax = plt.subplots(figsize=(18, 6))
ax.bar(x - 1.5*width, df['time_none'],      width, label='No Policy',  color='grey')
ax.bar(x - 0.5*width, df['time_agg'],       width, label='AGG Fixed',  color='#E74C3C')
ax.bar(x + 0.5*width, df['time_cons'],      width, label='CONS Fixed', color='#2E86AB')
ax.bar(x + 1.5*width, df['time_predicted'], width, label='Adaptive',   color='#2ECC71')

ax.set_xticks(x)
ax.set_xticklabels(df['short_name'], rotation=45, ha='right', fontsize=7)
ax.set_ylabel("Solve Time (s)")
ax.set_title("Per-Instance Runtime: Adaptive vs Fixed Policies\n(23 instances, spg_200_300 excluded - all timeout)")
ax.axhline(y=1200, color='black', linestyle='--', linewidth=0.8, label='Timeout (1200s)')
ax.legend()
plt.tight_layout()
plt.savefig(r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Pipeline\Figures\runtime_comparison.png",
            dpi=150)
plt.show()