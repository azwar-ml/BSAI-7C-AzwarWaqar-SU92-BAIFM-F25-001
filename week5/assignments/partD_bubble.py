"""
partD_bubble.py
Week 5 Part D1: 4-Variable Bubble Chart (300 DPI).
"""

import matplotlib.pyplot as plt
import pandas as pd
import vizlib

vizlib.style_defaults()

df = pd.read_csv("data/health_indicators.csv")

fig, ax = plt.subplots(figsize=(10, 6.5))

# Filter extreme GDP outlier or note on figure
# Linear area mapping for pop_m: A = pop_m * 4.0 + 30
sizes = df["pop_m"] * 3.8 + 25

regions = df["region"].unique()
pal = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
reg_colors = {r: pal[i % len(pal)] for i, r in enumerate(regions)}

for r in regions:
    sub = df[df["region"] == r]
    ax.scatter(
        sub["gdp_pc"],
        sub["healthcare_access"],
        s=sub["pop_m"] * 3.8 + 25,
        color=reg_colors[r],
        alpha=0.75,
        edgecolors="white",
        linewidth=0.8,
        label=r,
    )

ax.set_xscale("log")
ax.set_ylim(0, 105)
ax.set_xlabel("GDP per Capita ($ USD, Log Scale)")
ax.set_ylabel("Healthcare Access Index (0-100)")
ax.set_title(
    "Healthcare Access Scales with GDP per Capita; Regional Disparities Dominate Mass",
    fontweight="bold",
    loc="left",
)

# Reference Size Legend
geolib_size_pts = [10, 50, 150]
for p in geolib_size_pts:
    ax.scatter([], [], s=p * 3.8 + 25, c="#999999", alpha=0.6, edgecolors="white", label=f"{p}M Pop")

# Direct label key nations
targets = ["Norway", "United States", "Nigeria", "India", "China"]
for _, row in df.iterrows():
    if row.get("country", "") in targets:
        ax.annotate(row["country"], (row["gdp_pc"], row["healthcare_access"]), fontsize=8, fontweight="bold")

ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", title="Region & Population")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

plt.tight_layout()
vizlib.save_fig(fig, "partD_bubble.png")
plt.close(fig)
print("Saved partD_bubble.png at 300 DPI.")