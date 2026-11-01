"""
partD_pcp.py
Week 5 Part D2: Parallel Coordinates Plot with Greedy Correlation Chain & Brushing.
"""

import matplotlib.pyplot as plt
import pandas as pd
import vizlib

vizlib.style_defaults()

df = pd.read_csv("data/health_indicators.csv")

# Greedy order from self-check
chain_cols = ["healthcare_access", "cases_per_100k", "urban_pct", "median_age", "gdp_pc"]

# Percentile rank scaling preserves non-linear spreads across axes
df_scaled = df[chain_cols].rank(pct=True)

fig, ax = plt.subplots(figsize=(10, 6))

# Grey background for all observations
for i in range(len(df_scaled)):
    ax.plot(range(len(chain_cols)), df_scaled.iloc[i], color="#d0d0d0", alpha=0.45, lw=1.2)

# Brush one region (e.g., Sub-Saharan Africa or South Asia)
brush_region = df["region"].iloc[0]
brush_idx = df[df["region"] == brush_region].index

for idx in brush_idx:
    ax.plot(range(len(chain_cols)), df_scaled.loc[idx], color="#D55E00", alpha=0.9, lw=2.2)

ax.plot([], [], color="#D55E00", lw=2.2, label=f"Brushed: {brush_region}")
ax.plot([], [], color="#d0d0d0", lw=1.2, label="Other Regions (Greyed)")

ax.set_xticks(range(len(chain_cols)))
ax.set_xticklabels([c.replace("_", " ").title() for c in chain_cols], fontweight="bold", fontsize=9)
ax.set_yticks([0, 0.5, 1.0])
ax.set_yticklabels(["0% (Low)", "50% (Median)", "100% (High)"])
ax.set_title(
    "Parallel Coordinates: Strong Inverse Coupling in Healthcare vs Case Rate (r = -0.799)",
    fontweight="bold",
    loc="left",
)
ax.legend(frameon=False, loc="upper right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

plt.tight_layout()
vizlib.save_fig(fig, "partD_pcp.png")
plt.close(fig)
print("Saved partD_pcp.png at 300 DPI.")