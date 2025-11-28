# plot_epsilon_vs_accuracy_minimal.py
"""
Minimal, publication-ready IEEE-style Privacy–Utility plot (NO inset).
Cleanest possible version for Appendix A.

- Serif fonts (Times)
- Black curve with blue markers
- Subtle error bars
- No inset, no clutter
- Very light grid
- Per-seed points optional

Run: python plot_epsilon_vs_accuracy_minimal.py
Saves: figures/epsilon_vs_accuracy_minimal.(png/svg)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ---------- Config ----------
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)
OUT_PNG = OUT_DIR / "epsilon_vs_accuracy_minimal.png"
OUT_SVG = OUT_DIR / "epsilon_vs_accuracy_minimal.svg"
PLOT_PER_SEED = True

# ---------- Locate CSV ----------
this_file = Path(__file__).resolve()
project_root = this_file.parent
if project_root.name.lower() == "notebooks":
    project_root = project_root.parent
results_dir = project_root / "results"

if not results_dir.exists():
    raise FileNotFoundError(f"results/ directory not found at {results_dir}")

csv_exact = results_dir / "eps_results_final.csv"
if csv_exact.exists():
    csv_path = csv_exact
else:
    candidates = sorted(results_dir.glob("eps_results*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No eps_results*.csv found.")
    csv_path = candidates[0]

print(f"[Minimal plot] Using CSV: {csv_path}")
df = pd.read_csv(csv_path)

# Convert epsilon
df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce")
baseline_df = df[df["sigma"].isna()].copy()
dp_df = df[df["sigma"].notna()].copy()

# Aggregate DP runs
agg = (
    dp_df.groupby("sigma")
    .agg(
        eps_mean=("epsilon", "mean"),
        acc_mean=("test_acc", "mean"),
        acc_std=("test_acc", "std"),
    )
    .reset_index()
    .sort_values("eps_mean")
)

# ---------- Minimal IEEE Style ----------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.3,
        "figure.dpi": 300,
    }
)

fig, ax = plt.subplots(figsize=(6.5, 3.4))

# Per-seed points
if PLOT_PER_SEED:
    rng = np.random.default_rng(42)
    for _, r in dp_df.iterrows():
        x = r["epsilon"]
        if np.isnan(x):
            continue
        jitter = (0.002 * x if x > 0 else 0.002) * rng.normal()
        ax.scatter(
            x + jitter,
            r["test_acc"],
            s=14,
            alpha=0.5,
            color="#1f77b4",
            edgecolor="none",
            zorder=2,
        )

# Aggregated line + error bars
ax.errorbar(
    agg["eps_mean"],
    agg["acc_mean"],
    yerr=agg["acc_std"],
    fmt="-o",
    color="black",
    markerfacecolor="#1f77b4",
    markeredgecolor="black",
    markersize=6,
    capsize=3,
    elinewidth=0.9,
    label="DP runs (mean ± std)",
    zorder=3,
)

# Baseline
if len(baseline_df) > 0:
    baseline = baseline_df["test_acc"].mean()
    ax.axhline(
        baseline,
        color="0.35",
        linestyle="--",
        linewidth=1.0,
        label=f"Baseline ({baseline:.4f})",
        zorder=1,
    )

# X axis log-scale
ax.set_xscale("log")
ax.set_xlabel("ε (epsilon) [log scale]")
ax.set_ylabel("Test accuracy")
ax.set_title("DP-SGD: Privacy–Utility Curve (Minimal)")

# Limits
if len(agg) > 0:
    ymin, ymax = agg["acc_mean"].min(), agg["acc_mean"].max()
    pad = (ymax - ymin) * 0.25 if ymax > ymin else 0.0005
    ax.set_ylim(ymin - pad, ymax + pad)

# Light grid
ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.4)
ax.xaxis.set_minor_locator(AutoMinorLocator())

# Legend
leg = ax.legend(frameon=True, edgecolor="0.5")
leg.get_frame().set_linewidth(0.6)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
fig.savefig(OUT_SVG, bbox_inches="tight")

print(f"Saved minimal plots: {OUT_PNG} and {OUT_SVG}")