# plot_epsilon_vs_accuracy.py
import os, pandas as pd, matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True)

# read CSV (try parent folder then local)
candidates = ["../results/eps_results.csv", "results/eps_results.csv", "results/eps_results_example.csv"]
csv_path = next((p for p in candidates if os.path.exists(p)), None)
if csv_path is None:
    raise FileNotFoundError("Could not find results/eps_results.csv. Run appendix script first.")
df = pd.read_csv(csv_path)

# Convert sigma None-like to label
df['sigma_label'] = df['sigma'].apply(lambda x: "baseline" if pd.isna(x) else f"σ={x}")

# Aggregate by sigma (exclude baseline for epsilon aggregation)
agg = df.groupby('sigma_label').agg(
    epsilon_mean = ('epsilon', lambda s: np.nanmean(s.values.astype(float))),
    epsilon_std  = ('epsilon', lambda s: np.nanstd(s.values.astype(float))),
    acc_mean     = ('test_acc', 'mean'),
    acc_std      = ('test_acc', 'std'),
).reset_index()

# Separate baseline row
baseline_row = agg[agg['sigma_label'] == 'baseline']
non_baseline = agg[agg['sigma_label'] != 'baseline'].copy()

# Plot epsilon vs accuracy (x=epsilon, y=accuracy) with error bars
plt.figure(figsize=(7,5))
# plot non-baseline points (epsilon vs acc)
plt.errorbar(non_baseline['epsilon_mean'], non_baseline['acc_mean'],
             xerr=non_baseline['epsilon_std'], yerr=non_baseline['acc_std'],
             fmt='o-', capsize=4, label='DP runs')

# mark baseline at far right with no epsilon (draw as a separate point at x = max_eps * 1.05)
if not baseline_row.empty:
    # pick a large x position for baseline (or max epsilon * 1.05)
    max_eps = non_baseline['epsilon_mean'].replace([np.nan, None], 0).max()
    if np.isnan(max_eps) or max_eps == 0:
        max_x = 1.0
    else:
        max_x = max_eps * 1.05
    plt.errorbar([max_x], baseline_row['acc_mean'], yerr=baseline_row['acc_std'],
                 fmt='s', label='Non-private baseline', markersize=8)

# annotate text labels for each DP point
for _, r in non_baseline.iterrows():
    plt.annotate(r['sigma_label'], (r['epsilon_mean'], r['acc_mean']), textcoords="offset points", xytext=(6,6))

plt.xlabel("ε (epsilon)")
plt.ylabel("Test accuracy")
plt.title("DP-SGD: Privacy (ε) vs Accuracy")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
png_path = "figures/epsilon_vs_accuracy.png"
svg_path = "figures/epsilon_vs_accuracy.svg"
plt.savefig(png_path, dpi=300)
plt.savefig(svg_path)
print(f"Wrote {png_path} and {svg_path}")
