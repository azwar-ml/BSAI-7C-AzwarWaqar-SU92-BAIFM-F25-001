"""
partA_charts.py
Week 3 Part A: Rebuilding 9 flawed published charts and their honest redesigns.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import vizlib

vizlib.style_defaults()

# ==============================================================================
# CASE 1: Helmet Grade vs Head Injuries (Simpson's Paradox / Confounding)
# ==============================================================================
df1 = pd.read_csv("data/case01_helmet_rickshaw.csv")
# Self-check verification
rate_std = (
    df1[df1["helmet_grade"] == "Standard"]["head_injuries"].sum()
    / df1[df1["helmet_grade"] == "Standard"]["trips"].sum()
    * 1000
)
rate_prem = (
    df1[df1["helmet_grade"] == "Premium"]["head_injuries"].sum()
    / df1[df1["helmet_grade"] == "Premium"]["trips"].sum()
    * 1000
)
print(f"Case 1 Self-Check: Standard={rate_std:.2f}, Premium={rate_prem:.2f}")

fig, (ax_bad, ax_good) = plt.subplots(1, 2, figsize=(13, 5))

# Bad Rebuild
grades = ["Standard", "Premium"]
rates = [rate_std, rate_prem]
ax_bad.bar(grades, rates, color=["#0072B2", "#D55E00"], width=0.5)
ax_bad.set_title(
    "BAD EXAMPLE: Premium Helmets Have 2.5x Injury Rate",
    fontweight="bold",
    color="red",
)
ax_bad.set_ylabel("Head Injuries per 1,000 Trips")
ax_bad.set_ylim(0, 7)
for s in ("top", "right"):
    ax_bad.spines[s].set_visible(False)

# Honest Redesign (Conditioned on Road Type)
grouped1 = (
    df1.groupby(["road_type", "helmet_grade"])[["head_injuries", "trips"]]
    .sum()
    .reset_index()
)
grouped1["rate"] = (grouped1["head_injuries"] / grouped1["trips"]) * 1000
piv1 = grouped1.pivot(index="road_type", columns="helmet_grade", values="rate")

x = np.arange(len(piv1))
w = 0.35
ax_good.bar(
    x - w / 2,
    piv1["Standard"],
    w,
    label="Standard",
    color="#0072B2",
    alpha=0.85,
)
ax_good.bar(
    x + w / 2,
    piv1["Premium"],
    w,
    label="Premium",
    color="#D55E00",
    alpha=0.85,
)
ax_good.set_xticks(x)
ax_good.set_xticklabels(piv1.index)
ax_good.set_ylabel("Head Injuries per 1,000 Trips")
ax_good.set_ylim(0, max(rates) * 1.6)
ax_good.set_title(
    "Premium Helmets Reduce Injuries; Highway Exposure Confounded Aggregate Rate",
    fontweight="bold",
)
ax_good.legend(frameon=False)
for s in ("top", "right"):
    ax_good.spines[s].set_visible(False)

plt.tight_layout()
fig.savefig("case01_resolved.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# CASE 2: Canteen Chai vs Quiz Failures (Spurious Correlation / Confounding)
# ==============================================================================
df2 = pd.read_csv("data/case02_chai_quizzes.csv")
r_val, _ = stats.pearsonr(df2["cups_doodh_patti_sold"], df2["quiz_failures"])
print(f"Case 2 Self-Check: Pearson r = {r_val:.2f}")

fig, (ax_bad, ax_good) = plt.subplots(1, 2, figsize=(13, 5))

# Bad Rebuild
m, b = np.polyfit(df2["cups_doodh_patti_sold"], df2["quiz_failures"], 1)
ax_bad.scatter(
    df2["cups_doodh_patti_sold"], df2["quiz_failures"], color="#D55E00", s=40
)
x_line = np.linspace(
    df2["cups_doodh_patti_sold"].min(), df2["cups_doodh_patti_sold"].max(), 50
)
ax_bad.plot(x_line, m * x_line + b, color="#333333", lw=1.5)
ax_bad.set_title(
    f"BAD EXAMPLE: Chai Sales Predict Quiz Failures (r={r_val:.2f})",
    fontweight="bold",
    color="red",
)
ax_bad.set_xlabel("Cups of Doodh Patti Sold")
ax_bad.set_ylabel("Quiz Failures")
for s in ("top", "right"):
    ax_bad.spines[s].set_visible(False)

# Honest Redesign: Detect the hidden third variable in the CSV
candidate_cols = [c for c in df2.columns if c not in ["cups_doodh_patti_sold", "quiz_failures"]]
norm_col = next((c for c in candidate_cols if any(k in c.lower() for k in ["student", "enroll", "quizz", "held", "total"])), None)

if norm_col and pd.api.types.is_numeric_dtype(df2[norm_col]):
    # Normalize by the confounding volume/capacity variable
    chai_norm = df2["cups_doodh_patti_sold"] / df2[norm_col]
    fail_norm = df2["quiz_failures"] / df2[norm_col]
    r_honest, _ = stats.pearsonr(chai_norm, fail_norm)

    ax_good.scatter(chai_norm, fail_norm, color="#0072B2", s=45, alpha=0.85)
    ax_good.set_title(
        f"Controlling for {norm_col.replace('_', ' ').title()}: Correlation Vanishes (r={r_honest:.2f})",
        fontweight="bold",
    )
    ax_good.set_xlabel(f"Cups Sold per {norm_col.replace('_', ' ').title()}")
    ax_good.set_ylabel(f"Failures per {norm_col.replace('_', ' ').title()}")
elif "month" in [c.lower() for c in df2.columns]:
    # Time-series alignment showing shared seasonal/growth trajectory
    m_col = next(c for c in df2.columns if c.lower() == "month")
    ax_good.plot(df2[m_col], df2["quiz_failures"], marker="o", color="#0072B2", label="Quiz Failures")
    ax_good.set_title("Failures Follow Semester Exam Schedule, Not Canteen Habits", fontweight="bold")
    ax_good.set_xlabel("Month")
    ax_good.set_ylabel("Count")
else:
    # First differences to remove common monotonic trend
    diff_chai = df2["cups_doodh_patti_sold"].diff().dropna()
    diff_fail = df2["quiz_failures"].diff().dropna()
    r_diff, _ = stats.pearsonr(diff_chai, diff_fail)
    ax_good.scatter(diff_chai, diff_fail, color="#0072B2", s=45)
    ax_good.set_title(f"Month-over-Month Changes Are Uncorrelated (r={r_diff:.2f})", fontweight="bold")
    ax_good.set_xlabel("Δ Cups Sold")
    ax_good.set_ylabel("Δ Quiz Failures")

for s in ("top", "right"):
    ax_good.spines[s].set_visible(False)

plt.tight_layout()
fig.savefig("case02_resolved.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# CASE 3: Smog Action Plan & Station Attrition (Survivorship / Attrition Bias)
# ==============================================================================
df3 = pd.read_csv("data/case03_aqi_monitors.csv")

# Ensure date is sorted and mapped to consecutive day indices 1..120
unique_dates = sorted(df3["date"].unique())
date_to_day = {d: i + 1 for i, d in enumerate(unique_dates)}
df3["day_num"] = df3["date"].map(date_to_day)

daily = df3.groupby("day_num")["aqi"].mean().reset_index()
mean_pre = daily[daily["day_num"] <= 60]["aqi"].mean()
mean_post = daily[daily["day_num"] > 60]["aqi"].mean()
drop_pct = (mean_pre - mean_post) / mean_pre * 100
print(f"Case 3 Self-Check: Before={mean_pre:.0f}, After={mean_post:.0f}, Fall={drop_pct:.0f}%")

fig, (ax_bad, ax_good) = plt.subplots(1, 2, figsize=(14, 5))

# Bad Rebuild (y-axis strictly from 0 as required by recipe)
ax_bad.plot(daily["day_num"], daily["aqi"], color="#0072B2", lw=1.5)
ax_bad.axvline(60.5, color="#D55E00", linestyle="--", lw=1.5)
ax_bad.hlines(mean_pre, 1, 60, colors="red", linestyles=":", label=f"Days 1-60 Mean ({mean_pre:.0f})")
ax_bad.hlines(mean_post, 61, 120, colors="green", linestyles=":", label=f"Days 61-120 Mean ({mean_post:.0f})")
ax_bad.set_ylim(bottom=0)
ax_bad.set_title("BAD EXAMPLE: Smog Action Plan Appears to Cut AQI by 32%", fontweight="bold", color="red")
ax_bad.set_xlabel("Day of Observation")
ax_bad.set_ylabel("Mean AQI Across Reporting Monitors")
ax_bad.legend(frameon=False)
for s in ("top", "right"):
    ax_bad.spines[s].set_visible(False)

# Honest Redesign (Balanced Monitor Cohort)
# Identify monitor station identifier column
st_col = next((c for c in df3.columns if any(k in c.lower() for k in ["station", "monitor", "site", "id"])), None)

if st_col:
    # Find stations reporting across all 120 days
    reporting_counts = df3.dropna(subset=["aqi"]).groupby(st_col)["day_num"].nunique()
    max_days = reporting_counts.max()
    continuous_stations = reporting_counts[reporting_counts == max_days].index
    
    df3_balanced = df3[df3[st_col].isin(continuous_stations)]
    daily_bal = df3_balanced.groupby("day_num")["aqi"].mean().reset_index()
    
    ax_good.plot(daily_bal["day_num"], daily_bal["aqi"], color="#333333", lw=2, 
                 label=f"Continuous Monitors Only (N={len(continuous_stations)})")
    ax_good.plot(daily["day_num"], daily["aqi"], color="#cccccc", lw=1.2, 
                 label="Naive Mean (Unbalanced Attrition)")
else:
    # Fallback to reporting count per day to show the attrition mechanism
    daily_counts = df3.dropna(subset=["aqi"]).groupby("day_num")[df3.columns[0]].count()
    ax_good.plot(daily["day_num"], daily["aqi"], color="#0072B2", lw=1.5, label="Mean AQI")
    ax_good_r = ax_good.twinx()
    ax_good_r.plot(daily_counts.index, daily_counts.values, color="#D55E00", linestyle="--", label="Active Stations")
    ax_good_r.set_ylabel("Reporting Monitor Count", color="#D55E00")

ax_good.axvline(60.5, color="#D55E00", linestyle="--")
ax_good.set_ylim(bottom=0)
ax_good.set_title("AQI Reduction Is an Artifact of Monitor Dropout in High-Pollution Zones", fontweight="bold")
ax_good.set_xlabel("Day of Observation")
ax_good.set_ylabel("Mean AQI")
ax_good.legend(frameon=False, loc="upper right")
for s in ("top", "right"):
    ax_good.spines[s].set_visible(False)

plt.tight_layout()
fig.savefig("case03_resolved.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# CASE 4: Four Departments (Anscombe's Quartet Extension)
# ==============================================================================
df4 = pd.read_csv("data/case04_quartet_depts.csv")

# Detect department column
dept_col = next((c for c in df4.columns if "dept" in c.lower() or "department" in c.lower()), df4.columns[0])

# Detect x (study hours) and y (score) columns
num_cols = [c for c in df4.columns if c != dept_col and pd.api.types.is_numeric_dtype(df4[c])]
x_col = next((c for c in num_cols if any(k in c.lower() for k in ["hour", "study", "x"])), num_cols[0])
y_col = next((c for c in num_cols if c != x_col), num_cols[1])

table4 = (
    df4.groupby(dept_col)
    .agg(
        n=(x_col, "count"),
        mean_hours=(x_col, "mean"),
        sd_hours=(x_col, "std"),
        mean_score=(y_col, "mean"),
        sd_score=(y_col, "std"),
    )
    .reset_index()
)
table4["r"] = [
    df4[df4[dept_col] == d][x_col].corr(df4[df4[dept_col] == d][y_col])
    for d in table4[dept_col]
]
print("Case 4 Self-Check Summary Table:")
print(table4.round(3))

# Honest Redesign: 2x2 Small Multiples Scatter Plots
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
axes = axes.flatten()
depts = sorted(df4[dept_col].unique())

for idx, dept in enumerate(depts[:4]):
    sub = df4[df4[dept_col] == dept]
    ax = axes[idx]
    ax.scatter(sub[x_col], sub[y_col], color="#0072B2", s=40)
    m_d, b_d = np.polyfit(sub[x_col], sub[y_col], 1)
    x_rng = np.linspace(sub[x_col].min() - 1, sub[x_col].max() + 1, 50)
    ax.plot(x_rng, m_d * x_rng + b_d, color="#D55E00", lw=1.5, linestyle="--")
    ax.set_title(f"Department {dept}", fontweight="bold")
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.grid(True, linestyle="--", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.suptitle(
    "Identical Summary Stats Mask Curvature and Extreme Leverage Outliers",
    fontweight="bold",
    y=0.98,
)
plt.tight_layout()
fig.savefig("case04_resolved.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# CASE 5: Five Cohorts, Identical Statistics (Datasaurus Mechanism)
# ==============================================================================
df5 = pd.read_csv("data/case05_same_stats_shapes.csv")

# Detect shape/cohort grouping column
shape_col = next(
    (c for c in df5.columns if any(k in c.lower() for k in ["shape", "cohort", "group", "name"])),
    df5.columns[0]
)

# Detect the two numeric coordinate columns
num_cols5 = [c for c in df5.columns if c != shape_col and pd.api.types.is_numeric_dtype(df5[c])]
x_col5, y_col5 = num_cols5[0], num_cols5[1]

table5 = (
    df5.groupby(shape_col)
    .agg(
        n=(x_col5, "count"),
        mean_x=(x_col5, "mean"),
        sd_x=(x_col5, "std"),
        mean_y=(y_col5, "mean"),
        sd_y=(y_col5, "std"),
    )
    .reset_index()
)
table5["r"] = [
    df5[df5[shape_col] == s][x_col5].corr(df5[df5[shape_col] == s][y_col5])
    for s in table5[shape_col]
]
print("Case 5 Self-Check Summary Table:")
print(table5.round(2))

shapes = df5[shape_col].unique()
fig, axes = plt.subplots(1, len(shapes), figsize=(3.6 * len(shapes), 3.8), sharex=True, sharey=True)
if len(shapes) == 1:
    axes = [axes]

for idx, sname in enumerate(shapes):
    sub = df5[df5[shape_col] == sname]
    axes[idx].scatter(
        sub[x_col5], sub[y_col5], color="#009E73", s=18, alpha=0.7, edgecolors="none"
    )
    axes[idx].set_title(str(sname).capitalize(), fontweight="bold")
    axes[idx].set_xlabel(x_col5)
    if idx == 0:
        axes[idx].set_ylabel(y_col5)
    axes[idx].grid(True, linestyle="--", alpha=0.25)
    for s in ("top", "right"):
        axes[idx].spines[s].set_visible(False)

fig.suptitle(
    "Synthetically Constrained Datasets Reveal Drastically Non-Linear Architectures",
    fontweight="bold",
    y=1.05,
)
plt.tight_layout()
fig.savefig("case05_resolved.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# CASE 6: Sensor Matrix (Collinearity & Default Heatmap Misrepresentation)
# ==============================================================================
df6 = pd.read_csv("data/case06_sensor_matrix.csv")
drop_cols = ["age_band", "study_site", "hr_device_model"]
numeric_df6 = df6.drop(columns=[c for c in drop_cols if c in df6.columns])
corr6 = numeric_df6.corr()
print(
    f"Case 6 Self-Check: Matrix Shape={corr6.shape}, creat_hba1c={corr6.loc['creatinine', 'hba1c']:.2f}, sys_map={corr6.loc['systolic', 'map_mmhg']:.2f}"
)

fig, (ax_bad, ax_good) = plt.subplots(1, 2, figsize=(18, 8))

# Bad Rebuild
sns.heatmap(
    corr6,
    cmap="Reds",
    ax=ax_bad,
    cbar_kws={"label": "Correlation"},
    annot=False,
)
ax_bad.set_title(
    "BAD EXAMPLE: Sequential Map Exaggerates Positive Associations",
    fontweight="bold",
    color="red",
)

# Honest Redesign
mask = np.triu(np.ones_like(corr6, dtype=bool))
sns.heatmap(
    corr6,
    mask=mask,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    ax=ax_good,
    cbar_kws={"label": "Pearson Correlation (r)"},
    linewidths=0.5,
)
ax_good.set_title(
    "Diverging Palette Bound [-1, 1] Reveals Physiological Collinearity (Sys/MAP)",
    fontweight="bold",
)

plt.tight_layout()
fig.savefig("case06_resolved.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# CASE 7: Dual-Axis Revenue vs Cost (Scale Manipulation)
# =================================================