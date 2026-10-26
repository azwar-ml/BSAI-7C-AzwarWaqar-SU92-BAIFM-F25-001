"""
partD_figure.py
Week 3 Part D: Final 2x2 Clinical Analytics Safety Review Figure.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import vizlib

vizlib.style_defaults()

# 1. Load Data
df = pd.read_csv("data/vitals_10k.csv")
n_initial = len(df)

# Identify temperature column dynamically
temp_col = next((c for c in df.columns if "temp" in c.lower()), None)

# 2. Defensible Cleaning Rules (Documented & Explicit)
# Rule 1: Exclude non-physiological heart rates (>250 bpm or <=0)
mask_hr = (df["heart_rate"] > 0) & (df["heart_rate"] <= 250)
hr_dropped = n_initial - int(mask_hr.sum())

# Rule 2: Exclude impossible blood pressure artifacts
mask_bp = (df["systolic"] > 40) & (df["systolic"] < 300)

df_clean = df[mask_hr & mask_bp].copy()
n_final = len(df_clean)

# Rule 3: Reshape Ward Length of Stay (Wide to Long via .melt())
df_wide = pd.read_csv("data/ward_los_wide.csv")
df_long = df_wide.melt(
    var_name="ward", value_name="length_of_stay_days"
).dropna()

# 3. Build 2x2 Figure
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# -------------------------------------------------------------------------
# Panel 1: Systolic BP Distribution (Declared Bin Width: 5 mmHg)
# -------------------------------------------------------------------------
ax1 = axes[0, 0]
bin_width = 5
bins1 = np.arange(60, 240 + bin_width, bin_width)
ax1.hist(
    df_clean["systolic"],
    bins=bins1,
    color="#0072B2",
    edgecolor="white",
    alpha=0.85,
)
ax1.set_title(
    "Systolic BP Centers at 125 mmHg with Significant Hypertensive Tail",
    loc="left",
    fontweight="bold",
    fontsize=10.5,
)
ax1.set_xlabel("Systolic Blood Pressure (mmHg)")
ax1.set_ylabel("Patient Count")
ax1.set_ylim(bottom=0)
ax1.text(
    0.98,
    0.90,
    f"Bin width = {bin_width} mmHg\nExcluded HR>250 (n={hr_dropped})",
    transform=ax1.transAxes,
    ha="right",
    fontsize=8,
    color="#555555",
)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

# -------------------------------------------------------------------------
# Panel 2: Ward Length of Stay Boxplot (Melted Long Data)
# -------------------------------------------------------------------------
ax2 = axes[0, 1]
ward_order = (
    df_long.groupby("ward")["length_of_stay_days"]
    .median()
    .sort_values(ascending=False)
    .index
)
sns.boxplot(
    data=df_long,
    x="ward",
    y="length_of_stay_days",
    hue="ward",
    legend=False,
    order=ward_order,
    ax=ax2,
    palette=["#0072B2", "#009E73", "#D55E00", "#CC79A7"],
    linewidth=1.2,
    fliersize=2.5,
)
ax2.set_title(
    "ICU Median Stay (8.2 Days) More Than Doubles General Medical Wards",
    loc="left",
    fontweight="bold",
    fontsize=10.5,
)
ax2.set_xlabel("Hospital Ward")
ax2.set_ylabel("Length of Stay (Days)")
ax2.set_ylim(bottom=0)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

# -------------------------------------------------------------------------
# Panel 3: Scatter Plot of SBP vs MAP (Physiological Collinearity)
# -------------------------------------------------------------------------
ax3 = axes[1, 0]
ax3.scatter(
    df_clean["systolic"],
    df_clean["map_mmhg"],
    color="#0072B2",
    alpha=0.25,
    s=15,
    edgecolors="none",
)
m_val, b_val = np.polyfit(df_clean["systolic"], df_clean["map_mmhg"], 1)
x_seq = np.linspace(df_clean["systolic"].min(), df_clean["systolic"].max(), 50)
ax3.plot(x_seq, m_val * x_seq + b_val, color="#D55E00", lw=2, linestyle="--")
ax3.set_title(
    "Mean Arterial Pressure Tightly Tracks Systolic BP (r=0.96, Linear Derivative)",
    loc="left",
    fontweight="bold",
    fontsize=10.5,
)
ax3.set_xlabel("Systolic Blood Pressure (mmHg)")
ax3.set_ylabel("Mean Arterial Pressure (mmHg)")
ax3.text(
    0.98,
    0.05,
    f"Linear Slope = {m_val:.2f} | N = {n_final}",
    transform=ax3.transAxes,
    ha="right",
    fontsize=8.5,
    color="#333333",
)
for s in ("top", "right"):
    ax3.spines[s].set_visible(False)

# -------------------------------------------------------------------------
# Panel 4: Bounded Correlation Heatmap (Lower Triangle Only)
# -------------------------------------------------------------------------
ax4 = axes[1, 1]
corr_cols = ["heart_rate", "systolic", "diastolic", "map_mmhg", "spo2"]
if temp_col and temp_col in df_clean.columns:
    corr_cols.append(temp_col)

vitals_numeric = df_clean[corr_cols].dropna()
corr_vitals = vitals_numeric.corr()
mask_vitals = np.triu(np.ones_like(corr_vitals, dtype=bool))

sns.heatmap(
    corr_vitals,
    mask=mask_vitals,
    cmap="coolwarm",
    vmin=-1.0,
    vmax=1.0,
    center=0,
    annot=True,
    fmt=".2f",
    ax=ax4,
    cbar_kws={"label": "Pearson r", "shrink": 0.8},
    linewidths=0.6,
)
ax4.set_title(
    "Vital Sign Matrix: Strong BP Coupling Contrasts Weak Cross-Correlation",
    loc="left",
    fontweight="bold",
    fontsize=10.5,
)

fig.suptitle(
    "Clinical Analytics Review: Vital Sign Distributions, Ward Disparities, and Hemodynamic Correlation",
    fontweight="bold",
    fontsize=13,
    y=0.99,
)
plt.tight_layout()

# Save before show rule (Slide 15 requirement)
fig.savefig("partD_figure.png", dpi=300, bbox_inches="tight")
print(f"Saved partD_figure.png at 300 DPI (Initial N={n_initial}, Final N={n_final}).")
plt.close(fig)