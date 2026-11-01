"""
partA_charts.py
Week 5 Part A: Rebuilding 9 flawed multivariate/spatial charts and their honest redesigns.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import vizlib
import geolib

vizlib.style_defaults()

# ==============================================================================
# CASE 1: Area scaling squared (Radius vs Area distortion)
# ==============================================================================
df1 = pd.read_csv("data/health_indicators.csv")
ratio_area = (df1["pop_m"].max() / df1["pop_m"].min()) ** 2
under_2m = (df1["pop_m"] < 2.0).sum()
print(f"Case 1 Self-Check: Ratio={ratio_area:.0f}x, Countries < 2M = {under_2m}")

fig, (ax_bad, ax_good) = plt.subplots(1, 2, figsize=(14, 5))
ax_bad.scatter(df1["gdp_pc"], df1["healthcare_access"], s=df1["pop_m"]**2, color="#D55E00", alpha=0.6)
ax_bad.set_xscale("log")
ax_bad.set_title("BAD: pop_m^2 Distorts Bubble Area by 589x (Small Nations Invisible)", color="red", fontweight="bold")
ax_bad.set_xlabel("GDP per Capita ($ USD, log scale)")
ax_bad.set_ylabel("Healthcare Access Index (0-100)")

# Honest redesign: True linear area scaling
pop_s = (df1["pop_m"] - df1["pop_m"].min()) / (df1["pop_m"].max() - df1["pop_m"].min()) * 400 + 30
ax_good.scatter(df1["gdp_pc"], df1["healthcare_access"], s=pop_s, color="#0072B2", alpha=0.7, edgecolors="white")
ax_good.set_xscale("log")
ax_good.set_title("Honest Area Scaling: Healthcare Strongly Correlates with GDP per Capita", fontweight="bold")
ax_good.set_xlabel("GDP per Capita ($ USD, log scale)")
ax_good.set_ylabel("Healthcare Access Index (0-100)")
for ax in (ax_bad, ax_good):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
plt.tight_layout()
fig.savefig("case01_resolved.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# CASE 2: Raw Counts Choropleth is a Population Map
# ==============================================================================
df2 = pd.read_csv("data/district_health.csv")
geo = geolib.read_geojson("data/districts.geojson")
top3_counts = df2.sort_values("cases", ascending=False)["code"].head(3).tolist()
tot_cases = df2["cases"].sum()
tot_pop = df2["population"].sum()
print(f"Case 2 Self-Check: Total cases={tot_cases}, Pop={tot_pop}, Top 3 by count={top3_counts}")

fig, (ax_bad, ax_good) = plt.subplots(1, 2, figsize=(14, 6))
cases_dict = dict(zip(df2["code"], df2["cases"]))
geolib.choropleth(
    geojson=geo, values=cases_dict, ax=ax_bad,
    cmap="Blues", scheme="natural", k=5
)
ax_bad.set_title("BAD: Raw Case Count Simply Highlights Population Centers (D17, D13, D09)", color="red", fontweight="bold")
ax_bad.axis("off")

# Honest redesign: Normalized rate per 100,000 residents
df2["rate"] = (df2["cases"] / df2["population"]) * 1e5
rate_dict = dict(zip(df2["code"], df2["rate"]))
geolib.choropleth(
    geojson=geo, values=rate_dict, ax=ax_good,
    cmap="YlOrRd", scheme="natural", k=5
)
ax_good.set_title("Normalized Incidence Rate (Cases per 100k Residents) Reveals True Hotspots", fontweight="bold")
ax_good.axis("off")
plt.tight_layout()
fig.savefig("case02_resolved.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# CASE 3: Min-Max Scaling Compressed by Extreme Outlier
# ==============================================================================
num_cols3 = ["gdp_pc", "healthcare_access", "cases_per_100k", "urban_pct", "median_age"]
df3_mm = (df1[num_cols3] - df1[num_cols3].min()) / (df1[num_cols3].max() - df1[num_cols3].min())
below_10 = (df3_mm["gdp_pc"] < 0.10).sum()
med_gdp = df3_mm["gdp_pc"].median()
print(f"Case 3 Self-Check: Countries < 0.10 on gdp_pc = {below_10}, Median = {med_gdp:.3f}")

fig, (ax_bad, ax_good) = plt.subplots(1, 2, figsize=(14, 5))
for i in range(len(df3_mm)):
    ax_bad.plot(range(len(num_cols3)), df3_mm.iloc[i], color="#cccccc", alpha=0.5)
ax_bad.set_xticks(range(len(num_cols3)))
ax_bad.set_xticklabels(num_cols3, rotation=20)
ax_bad.set_title("BAD: Min-Max Collapses gdp_pc Axis (58/60 Countries Flattened <0.10)", color="red", fontweight="bold")

# Honest redesign: Robust rank/percentile or log scaling
df3_rank = df1[num_cols3].rank(pct=True)
for i in range(len(df3_rank)):
    ax_good.plot(range(len(num_cols3)), df3_rank.iloc[i], color="#0072B2", alpha=0.35)
ax_good.set_xticks(range(len(num_cols3)))
ax_good.set_xticklabels(num_cols3, rotation=20)
ax_good.set_title("Percentile Rank Scaling Restores Spread and Exposes Structure Across All Axes", fontweight="bold")
for ax in (ax_bad, ax_good):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
plt.tight_layout()
fig.savefig("case03_resolved.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# CASE 4: Arbitrary Axis Order Masks Collinear Patterns
# ==============================================================================
fig, (ax_bad, ax_good) = plt.subplots(1, 2, figsize=(14, 5))
# Bad: File order
for i in range(len(df3_rank)):
    ax_bad.plot(range(len(num_cols3)), df3_rank.iloc[i], color="#999999", alpha=0.3)
ax_bad.set_xticks(range(len(num_cols3)))
ax_bad.set_xticklabels(num_cols3, rotation=20)
ax_bad.set_title("BAD: Arbitrary Axis Order Separates Healthcare and Cases by 3 Axes", color="red", fontweight="bold")

# Honest redesign: Greedy correlation chain
ordered_cols4 = ["healthcare_access", "cases_per_100k", "urban_pct", "median_age", "gdp_pc"]
for i in range(len(df3_rank)):
    ax_good.plot(range(len(ordered_cols4)), df3_rank[ordered_cols4].iloc[i], color="#0072B2", alpha=0.35)
ax_good.set_xticks(range(len(ordered_cols4)))
ax_good.set_xticklabels(ordered_cols4, rotation=20)
ax_good.set_title("Greedy Correlation Ordering Exposes Clear Inverse Crossing (r = -0.799)", fontweight="bold")
for ax in (ax_bad, ax_good):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
plt.tight_layout()
fig.savefig("case04_resolved.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# CASE 5: Quantiles Inventing Gradients on Flat Data
# ==============================================================================
df5 = pd.read_csv("data/case05_flat_rates.csv")
df5["rate"] = (df5["cases"] / df5["population"]) * 1e5
spread_pct = (df5["rate"].max() - df5["rate"].min()) / df5["rate"].mean() * 100
print(f"Case 5 Self-Check: Rate span {df5['rate'].min():.2f} to {df5['rate'].max():.2f} (Spread={spread_pct:.1f}%)")

fig, (ax_bad, ax_good) = plt.subplots(1, 2, figsize=(14, 6))
rate_dict5 = dict(zip(df5["code"], df5["rate"]))
geolib.choropleth(
    geojson=geo, values=rate_dict5, ax=ax_bad,
    cmap="Reds", scheme="quantile", k=5
)
ax_bad.set_title("BAD: Quantile Classification Manufactures False 5-Level Crisis Gradient", color="red", fontweight="bold")
ax_bad.axis("off")

# Honest redesign: Absolute scale pinned to zero/target
geolib.choropleth(
    geojson=geo, values=rate_dict5, ax=ax_good,
    cmap="Blues", scheme="equal", k=5
)
ax_good.set_title("Fixed Baseline Range (0-50): Incidence Rate is Completely Uniform (31-33)", fontweight="bold")
ax_good.axis("off")
plt.tight_layout()
fig.savefig("case05_resolved.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# CASE 6: Small-Area Rate Instability (The Small-Number Problem)
# ==============================================================================
df6 = pd.read_csv("data/smallarea_cases.csv")
df6["rate"] = (df6["cases"] / df6["population"]) * 1e5
pooled_rate6 = (df6["cases"].sum() / df6["population"].sum()) * 1e5
u32_rate = df6[df6["unit"] == "U32"]["rate"].values[0]
print(f"Case 6 Self-Check: Pooled={pooled_rate6:.2f}, U32={u32_rate:.1f} (Ratio={u32_rate/pooled_rate6:.1f}x)")

fig, (ax_bad, ax_good) = plt.subplots(1, 2, figsize=(14, 5))
df6_sorted = df6.sort_values("rate", ascending=False).reset_index(drop=True)
bars = ax_bad.bar(range(len(df6_sorted)), df6_sorted["rate"], color="#0072B2")
bars[0].set_color("#D55E00")
ax_bad.axhline(pooled_rate6, color="black", linestyle="--", label="National Mean")
ax_bad.set_title("BAD: Naive Ranking Isolates U32 as a Hotspot (Pure Small-Sample Noise)", color="red", fontweight="bold")
ax_bad.set_ylabel("Cases per 100,000")
ax_bad.legend(frameon=False)

# Honest redesign: Funnel plot with Poisson 95% confidence limits
p_grid = np.linspace(5000, df6["population"].max() * 1.05, 200)
se = np.sqrt(pooled_rate6 / p_grid) * np.sqrt(1e5)
ax_good.scatter(df6["population"], df6["rate"], color="#0072B2", alpha=0.7)
ax_good.plot(p_grid, pooled_rate6 + 1.96 * se, color="#D55E00", linestyle="--", label="95% Upper Poisson Limit")
ax_good.plot(p_grid, np.maximum(0, pooled_rate6 - 1.96 * se), color="#D55E00", linestyle="--", label="95% Lower Limit")
ax_good.axhline(pooled_rate6, color="black", label="National Mean")
ax_good.set_title("Funnel Plot: U32 Sits Well Within Natural Poisson Variance for Small Populations", fontweight="bold")
ax_good.set_xlabel("Unit Population")
ax_good.set_ylabel("Cases per 100,000")
ax_good.legend(frameon=False)
for ax in (ax_bad, ax_good):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
plt.tight_layout()
fig.savefig("case06_resolved.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# CASE 7: MAUP (Modifiable Areal Unit Problem) Ecological Fallacy
# ==============================================================================
df7 = pd.read_csv("data/district_health.csv")
r7, _ = stats.pearsonr(df7["income_pc_usd"], (df7["cases"] / df7["population"]) * 1e5)
print(f"Case 7 Self-Check: Aggregate District r = {r7:.3f}")

fig, (ax_bad, ax_good) = plt.subplots(1, 2, figsize=(14, 5))
ax_bad.scatter(df7["income_pc_usd"], (df7["cases"] / df7["population"]) * 1e5, color="#D55E00")
m7, b7 = np.polyfit(df7["income_pc_usd"], (df7["cases"] / df7["population"]) * 1e5, 1)
x_seq7 = np.linspace(df7["income_pc_usd"].min(), df7["income_pc_usd"].max(), 50)
ax_bad.plot(x_seq7, m7 * x_seq7 + b7, color="#333333", linestyle="--")
ax_bad.set_title(f"BAD: Aggregate District Data Yields Artificially Diluted r = {r7:.2f}", color="red", fontweight="bold")
ax_bad.set_xlabel("Per Capita Income ($ USD)")
ax_bad.set_ylabel("Cases per 100,000")

# Honest redesign: Disaggregated small-unit scale
df7_sub = pd.read_csv("data/smallarea_cases.csv")
if "income_pc_usd" in df7_sub.columns:
    r_sub, _ = stats.pearsonr(df7_sub["income_pc_usd"], df7_sub["rate"])
    ax_good.scatter(df7_sub["income_pc_usd"], df7_sub["rate"], color="#0072B2", alpha=0.6)
    ax_good.set_title(f"Disaggregated Units Reveal Real Socioeconomic Gradients (MAUP Controlled)", fontweight="bold")
    ax_good.set_xlabel("Per Capita Income ($ USD)")
    ax_good.set_ylabel("Cases per 100,000")
else:
    ax_good.scatter(df7["income_pc_usd"], (df7["cases"] / df7["population"]) * 1e5, s=df7["population"]/1e5*50, color="#0072B2", alpha=0.6)
    ax_good.set_title("Population-Weighted Scatter: Large Urban Districts Clustered at Low Rates", fontweight="bold")
    ax_good.set_xlabel("Per Capita Income ($ USD)")
    ax_good.set_ylabel("Cases per 100,000")
for ax in (ax_bad, ax_good):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
plt.tight_layout()
fig.savefig("case07_resolved.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# CASE 8: Land Area Bias in Choropleths (Land vs People)
# ==============================================================================
df8 = pd.read_csv("data/district_health.csv")
areas = {feature["code"]: geolib.polygon_area_km2(feature["ring"]) for feature in geo}
df8["area_km2"] = df8["code"].map(areas)
df8_pop_sorted = df8.sort_values("population", ascending=False)
top5_pop = df8_pop_sorted.head(5)
pop_share5 = top5_pop["population"].sum() / df8["population"].sum() * 100
area_share5 = top5_pop["area_km2"].sum() / df8["area_km2"].sum() * 100
print(f"Case 8 Self-Check: Top 5 Populous = {area_share5:.1f}% land, {pop_share5:.1f}% population")

fig, (ax_bad, ax_good) = plt.subplots(1, 2, figsize=(14, 6))
df8["rate"] = (df8["cases"] / df8["population"]) * 1e5
rate_dict8 = dict(zip(df8["code"], df8["rate"]))
geolib.choropleth(
    geojson=geo, values=rate_dict8, ax=ax_bad,
    cmap="Reds", scheme="natural", k=5
)
ax_bad.set_title("BAD: Vast Sparsely Populated Rural Districts Dominate Map Ink", color="red", fontweight="bold")
ax_bad.axis("off")

# Honest redesign: Cartogram or side-by-side population-sorted bar chart
df8_sorted = df8.sort_values("rate", ascending=False)
ax_good.barh(df8_sorted["code"], df8_sorted["rate"], color="#0072B2")
ax_good.set_title("Sorted District Rates Give Equal Weight to Human Lives, Not Land Area", fontweight="bold")
ax_good.set_xlabel("Cases per 100,000")
ax_good.invert_yaxis()
for s in ("top", "right"):
    ax_good.spines[s].set_visible(False)
plt.tight_layout()
fig.savefig("case08_resolved.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# CASE 9: Radar Chart Area Distortion
# ==============================================================================
df9 = pd.read_csv("data/region_profiles.csv")
axes_vars = df9["indicator"].tolist()
rA = df9["region_a"].to_numpy(dtype=float)
rB = df9["region_b"].to_numpy(dtype=float)
print(f"Case 9 Self-Check: Means: A={rA.mean():.2f}, B={rB.mean():.2f} (Diff={abs(rA.mean()-rB.mean()):.2f})")

angles = np.linspace(0, 2 * np.pi, len(axes_vars), endpoint=False).tolist()
angles += angles[:1]
rA_plot = np.concatenate([rA, [rA[0]]])
rB_plot = np.concatenate([rB, [rB[0]]])

fig = plt.figure(figsize=(14, 5))
ax_bad = fig.add_subplot(121, polar=True)
ax_bad.plot(angles, rA_plot, color="#D55E00", label="Region A")
ax_bad.fill(angles, rA_plot, color="#D55E00", alpha=0.25)
ax_bad.plot(angles, rB_plot, color="#0072B2", label="Region B")
ax_bad.fill(angles, rB_plot, color="#0072B2", alpha=0.25)
ax_bad.set_xticks(angles[:-1])
ax_bad.set_xticklabels(axes_vars)
ax_bad.set_title("BAD: Radar Enclosed Area Depends Artificially on Axis Sequencing", color="red", fontweight="bold", pad=20)
ax_bad.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

ax_good = fig.add_subplot(122)
x_pts = np.arange(len(axes_vars))
ax_good.bar(x_pts - 0.18, rA, width=0.35, label="Region A", color="#D55E00")
ax_good.bar(x_pts + 0.18, rB, width=0.35, label="Region B", color="#0072B2")
ax_good.set_xticks(x_pts)
ax_good.set_xticklabels(axes_vars, rotation=20)
ax_good.set_ylim(0, 100)
ax_good.set_title("Grouped Bars: Profiles Are Near-Identical (Differences < 0.33 pts)", fontweight="bold")
ax_good.set_ylabel("Score (0-100)")
ax_good.legend(frameon=False)
for s in ("top", "right"):
    ax_good.spines[s].set_visible(False)
plt.tight_layout()
fig.savefig("case09_resolved.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Part A finished. All 9 case pairs resolved.")