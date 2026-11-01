"""
partA_charts.py
Week 4 Part A: Rebuild 9 broken time-series/proportion charts and their honest redesigns.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vizlib

vizlib.style_defaults()

# ==============================================================================
# CASE 1: Alphabetical string sorting instead of datetime sorting
# ==============================================================================
df1 = pd.read_csv("data/case01_orders_stringdate.csv")
# Bad rebuild: sorted by month column as strings
df1_sorted = df1.sort_values("month").reset_index(drop=True)
diffs = df1_sorted["orders"].diff()
drop1 = diffs.iloc[4]
drop2 = diffs.iloc[16]
print(f"Case 1 Self-Check: String-sorted drops = {drop1:.0f}, {drop2:.0f}")

fig, (ax_bad, ax_good) = plt.subplots(1, 2, figsize=(14, 5))
ax_bad.plot(
    range(len(df1_sorted)),
    df1_sorted["orders"],
    marker="o",
    color="#D55E00",
    lw=1.5,
)
ax_bad.set_xticks(range(len(df1_sorted)))
ax_bad.set_xticklabels(df1_sorted["month"], rotation=90, fontsize=8)
ax_bad.set_ylim(bottom=0)
ax_bad.set_title(
    "BAD: String Sort Arranges Apr 2024 -> Aug 2024 -> Dec 2025 (False Cliff)",
    color="red",
    fontweight="bold",
)
ax_bad.set_ylabel("Orders")

# Redesign: Proper chronological parse and sort
df1["dt"] = pd.to_datetime(df1["month"])
df1_chrono = df1.sort_values("dt").reset_index(drop=True)
ax_good.plot(
    df1_chrono["dt"],
    df1_chrono["orders"],
    marker="o",
    color="#0072B2",
    lw=2,
)
ax_good.set_ylim(bottom=0)
ax_good.set_title(
    "Honest Chronological Time-Series: Steady Growth with Mild Q4 Peak",
    fontweight="bold",
)
ax_good.set_xlabel("Calendar Date")
ax_good.set_ylabel("Orders per Month")
for ax in (ax_bad, ax_good):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
plt.tight_layout()
fig.savefig("case01_resolved.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# CASE 2: Row index plotting hides store closures and weekend/closure gaps
# ==============================================================================
df2 = pd.read_csv("data/case02_shop_closed.csv")
v_first = df2["sales_pkr_k"].iloc[0]
v_last = df2["sales_pkr_k"].iloc[-1]
rise_pct = (v_last - v_first) / v_first * 100
print(f"Case 2 Self-Check: First={v_first:.0f}, Last={v_last:.0f}, Rise=+{rise_pct:.0f}%")

fig, (ax_bad, ax_good) = plt.subplots(1, 2, figsize=(14, 5))
# Bad rebuild: plot against row index
m2, b2 = np.polyfit(range(len(df2)), df2["sales_pkr_k"], 1)
ax_bad.plot(range(len(df2)), df2["sales_pkr_k"], marker="o", color="#D55E00", lw=1.2)
ax_bad.plot(
    range(len(df2)),
    m2 * np.arange(len(df2)) + b2,
    color="#333333",
    linestyle="--",
)
ax_bad.set_title(
    "BAD: Plotting Row Index Removes Store Closure Gaps, Synthesizing False Growth",
    color="red",
    fontweight="bold",
)
ax_bad.set_xlabel("Row Number (Not Time)")
ax_bad.set_ylabel("Sales (PKR K)")

# Redesign: Reindex to full calendar to reveal missing closure periods
date_col2 = next(c for c in df2.columns if "date" in c.lower() or "day" in c.lower())
df2["dt"] = pd.to_datetime(df2[date_col2])
df2_idx = df2.set_index("dt").sort_index()
full_idx = pd.date_range(df2_idx.index.min(), df2_idx.index.max(), freq="D")
df2_full = df2_idx.reindex(full_idx)

ax_good.plot(
    df2_full.index,
    df2_full["sales_pkr_k"],
    marker="o",
    markersize=4,
    color="#0072B2",
    lw=1.5,
)
ax_good.set_title(
    "Calendar-Indexed Series: Sales Stagnated and Collapsed During Multi-Week Closure",
    fontweight="bold",
)
ax_good.set_xlabel("Calendar Date")
ax_good.set_ylabel("Sales (PKR K)")
for ax in (ax_bad, ax_good):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
plt.tight_layout()
fig.savefig("case02_resolved.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# CASE 3: Resampling an instantaneous LEVEL with .sum() instead of .mean() / .max()
# ==============================================================================
df3 = pd.read_csv("data/case03_bandwidth_levels.csv")
dt_col3 = next(c for c in df3.columns if "time" in c.lower() or "date" in c.lower() or "ts" in c.lower())
df3["dt"] = pd.to_datetime(df3[dt_col3])
df3_idx = df3.set_index("dt").sort_index()

# Bad rebuild: .sum() on instantaneous levels
daily_sum = df3_idx["concurrent_users"].resample("D").sum()
june_sum_mean = daily_sum["2025-06"].mean()
july_sum_mean = daily_sum["2025-07"].mean()
pct_jump = (july_sum_mean - june_sum_mean) / june_sum_mean * 100
print(f"Case 3 Self-Check: June Mean={june_sum_mean:.0f}, July Mean={july_sum_mean:.0f}, Jump=+{pct_jump:.0f}%")

fig, (ax_bad, ax_good) = plt.subplots(1, 2, figsize=(14, 5))
ax_bad.plot(daily_sum.index, daily_sum.values, color="#D55E00", lw=1.5)
ax_bad.set_ylim(bottom=0)
ax_bad.set_title(
    "BAD: Resampling Concurrent Users (Level) with .sum() Creates +43% Phantom Surge",
    color="red",
    fontweight="bold",
)
ax_bad.set_ylabel("Sum of Concurrent Users (Nonsensical Unit)")

# Redesign: Levels take .mean() or .max()
daily_mean = df3_idx["concurrent_users"].resample("D").mean()
daily_max = df3_idx["concurrent_users"].resample("D").max()
ax_good.plot(
    daily_mean.index,
    daily_mean.values,
    color="#0072B2",
    lw=1.8,
    label="Daily Mean Concurrent Users",
)
ax_good.plot(
    daily_max.index,
    daily_max.values,
    color="#009E73",
    lw=1.2,
    linestyle="--",
    label="Daily Peak Concurrent Users",
)
ax_good.set_ylim(bottom=0)
ax_good.set_title(
    "Correct Level Aggregation: Sampling Frequency Doubled; Actual User Load Was Flat",
    fontweight="bold",
)
ax_good.set_ylabel("Concurrent Users (Mean / Peak)")
ax_good.legend(frameon=False)
for ax in (ax_bad, ax_good):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
plt.tight_layout()
fig.savefig("case03_resolved.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# CASE 4: Trailing rolling mean phase lag (31-day window delays inflection)
# ==============================================================================
df4 = pd.read_csv("data/case04_intervention_lag.csv")
dt_col4 = next(c for c in df4.columns if "date" in c.lower() or "day" in c.lower() or "ts" in c.lower())
df4["dt"] = pd.to_datetime(df4[dt_col4])
df4 = df4.sort_values("dt").reset_index(drop=True)
df4["rolling_31_trailing"] = df4["conversion_pct"].rolling(31).mean()
bottom_date = df4.loc[df4["rolling_31_trailing"].idxmin(), "dt"]
print(f"Case 4 Self-Check: Trailing mean bottoms out on {bottom_date.strftime('%Y-%m-%d')}")

fig, (ax_bad, ax_good) = plt.subplots(1, 2, figsize=(14, 5))
ax_bad.plot(df4["dt"], df4["conversion_pct"], color="#cccccc", lw=1, label="Raw")
ax_bad.plot(
    df4["dt"],
    df4["rolling_31_trailing"],
    color="#D55E00",
    lw=2,
    label="31-day Trailing Mean",
)
ax_bad.axvline(bottom_date, color="black", linestyle="--")
ax_bad.set_title(
    f"BAD: 31-day Trailing Window Artificially Delays Turning Point to {bottom_date.strftime('%d %b')}",
    color="red",
    fontweight="bold",
)
ax_bad.set_ylabel("Conversion Rate (%)")
ax_bad.legend(frameon=False)

# Redesign: Centered rolling mean removes the 15-day phase shift
df4["rolling_31_centered"] = df4["conversion_pct"].rolling(31, center=True).mean()
true_bottom = df4.loc[df4["rolling_31_centered"].idxmin(), "dt"]
ax_good.plot(df4["dt"], df4["conversion_pct"], color="#cccccc", lw=0.9, label="Raw Daily")
ax_good.plot(
    df4["dt"],
    df4["rolling_31_centered"],
    color="#0072B2",
    lw=2,
    label="31-day Centred Rolling Mean",
)
ax_good.axvline(
    true_bottom,
    color="#009E73",
    linestyle="--",
    lw=2,
    label=f"True Inflection ({true_bottom.strftime('%d %b')})",
)
ax_good.set_title(
    "Centred Window Aligns Inflection Precisely with Late-August Checkout Rollout",
    fontweight="bold",
)
ax_good.set_ylabel("Conversion Rate (%)")
ax_good.legend(frameon=False)
for ax in (ax_bad, ax_good):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
plt.tight_layout()
fig.savefig("case04_resolved.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# CASE 5: Cherry-picked window truncates seasonal annual cycle
# ==============================================================================
df5 = pd.read_csv("data/case05_revenue_window.csv")
dt_col5 = next(c for c in df5.columns if "date" in c.lower() or "month" in c.lower())
df5["dt"] = pd.to_datetime(df5[dt_col5])
df5 = df5.sort_values("dt").reset_index(drop=True)

# Bad rebuild: Subselect May to Nov 2025
df5_sub = df5[(df5["dt"] >= "2025-05-01") & (df5["dt"] <= "2025-11-30")].copy()
v_start5 = df5_sub["revenue_pkr_m"].iloc[0]
v_end5 = df5_sub["revenue_pkr_m"].iloc[-1]
pct_fall5 = (v_end5 - v_start5) / v_start5 * 100
print(f"Case 5 Self-Check: Start={v_start5:.1f}, End={v_end5:.1f}, Fall={pct_fall5:.1f}%")

fig, (ax_bad, ax_good) = plt.subplots(1, 2, figsize=(14, 5))
ax_bad.plot(
    df5_sub["dt"],
    df5_sub["revenue_pkr_m"],
    marker="o",
    color="#D55E00",
    lw=2,
)
ax_bad.set_ylim(bottom=0)
ax_bad.set_title(
    f"BAD: Cherry-Picked 7-Month Window Shows −41.9% Disaster",
    color="red",
    fontweight="bold",
)
ax_bad.set_ylabel("Revenue (PKR M)")

# Redesign: Full multi-year context exposes standard summer off-season dip
ax_good.plot(
    df5["dt"],
    df5["revenue_pkr_m"],
    marker="o",
    markersize=4,
    color="#0072B2",
    lw=1.8,
)
ax_good.axvspan(
    pd.to_datetime("2025-05-01"),
    pd.to_datetime("2025-11-30"),
    color="#D55E00",
    alpha=0.15,
    label="Cherry-Picked Window",
)
ax_good.set_ylim(bottom=0)
ax_good.set_title(
    "Full Time Span Reveals Annual Summer Seasonality; Year-over-Year Trend is Up",
    fontweight="bold",
)
ax_good.set_ylabel("Revenue (PKR M)")
ax_good.legend(frameon=False)
for ax in (ax_bad, ax_good):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
plt.tight_layout()
fig.savefig("case05_resolved.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# CASE 6: Incomplete / partial final period plotted alongside completed periods
# ==============================================================================
df6 = pd.read_csv("data/case06_monthly_partial.csv")
dt_col6 = next(c for c in df6.columns if "date" in c.lower() or "day" in c.lower() or "ts" in c.lower())
df6["dt"] = pd.to_datetime(df6[dt_col6])
monthly6 = df6.set_index("dt")["orders"].resample("MS").sum()
sep_orders = monthly6.loc["2025-09-01"]
oct_orders = monthly6.loc["2025-10-01"]
print(f"Case 6 Self-Check: Sep={sep_orders:.0f}, Oct={oct_orders:.0f} (Ratio={oct_orders/sep_orders*100:.0f}%)")

fig, (ax_bad, ax_good) = plt.subplots(1, 2, figsize=(14, 5))
m_labels = [d.strftime("%b") for d in monthly6.index]
ax_bad.bar(m_labels, monthly6.values, color="#D55E00", width=0.6)
ax_bad.set_ylim(bottom=0)
ax_bad.set_title(
    "BAD: Partial Month Plotted as Full Month Simulates a 63% Demand Collapse",
    color="red",
    fontweight="bold",
)
ax_bad.set_ylabel("Orders (Sum)")

# Redesign: Convert to daily run-rate or hatch partial month with clear annotation
days_in_month = monthly6.index.days_in_month.values
# October only had partial recorded days
max_date_oct = df6[df6["dt"] >= "2025-10-01"]["dt"].max()
oct_days_recorded = max_date_oct.day if pd.notnull(max_date_oct) else 11
run_rate = monthly6.values.copy().astype(float)
run_rate[:-1] = run_rate[:-1] / days_in_month[:-1]
run_rate[-1] = run_rate[-1] / oct_days_recorded

bars = ax_good.bar(m_labels[:-1], run_rate[:-1], color="#0072B2", width=0.6, label="Complete Months")
ax_good.bar(
    m_labels[-1],
    run_rate[-1],
    color="none",
    edgecolor="#0072B2",
    hatch="//",
    linewidth=1.5,
    width=0.6,
    label=f"Oct Run-Rate ({oct_days_recorded} Days)",
)
ax_good.set_ylim(bottom=0)
ax_good.set_title(
    "Normalized Daily Order Rate Proves October Performance Was Strong and Growing",
    fontweight="bold",
)
ax_good.set_ylabel("Average Orders per Day")
ax_good.legend(frameon=False)
for ax in (ax_bad, ax_good):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
plt.tight_layout()
fig.savefig("case06_resolved.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# CASE 7: Stacked area charts distort upper series baselines
# ==============================================================================
df7 = pd.read_csv("data/case07_stacked_products.csv")
dt_col7 = next(c for c in df7.columns if "date" in c.lower() or "month" in c.lower() or "day" in c.lower())
df7["dt"] = pd.to_datetime(df7[dt_col7])
prods = [c for c in df7.columns if c.startswith("product_")]
totals7 = df7[prods].sum(axis=1)
print(f"Case 7 Self-Check: Total envelope min={totals7.min():.0f}, max={totals7.max():.0f}")

fig, (ax_bad, ax_good) = plt.subplots(1, 2, figsize=(14, 5))
ax_bad.stackplot(
    df7["dt"],
    [df7[p] for p in prods],
    labels=prods,
    colors=["#0072B2", "#009E73", "#F0E442", "#D55E00"],
    alpha=0.85,
)
ax_bad.set_title(
    "BAD: Stacking Distorts Product D Baseline (Volatilities of A-C Transferred to D)",
    color="red",
    fontweight="bold",
)
ax_bad.set_ylabel("Sales Volume")
ax_bad.legend(loc="upper left", frameon=False)

# Redesign: Unstacked lines (small multiples / direct lines from shared zero)
for p, c in zip(prods, ["#0072B2", "#009E73", "#56B4E9", "#D55E00"]):
    lw = 2.5 if p == "product_d" else 1.2
    alpha = 1.0 if p == "product_d" else 0.65
    ax_good.plot(df7["dt"], df7[p], label=p.replace("_", " ").title(), color=c, lw=lw, alpha=alpha)
ax_good.set_ylim(bottom=0)
ax_good.set_title(
    "Independent Zero Baselines Reveal Product D Was Completely Stable (Product B Was Volatile)",
    fontweight="bold",
)
ax_good.set_ylabel("Units Sold per Month")
ax_good.legend(frameon=False)
for ax in (ax_bad, ax_good):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
plt.tight_layout()
fig.savefig("case07_resolved.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# CASE 8: Multi-touch attribution shares that sum to > 100% in a pie chart
# ==============================================================================
df8 = pd.read_csv("data/case08_channel_attrib.csv")
touch_cols = [c for c in df8.columns if c.startswith("touch_")]
cust_count = len(df8)
shares8 = {c.replace("touch_", ""): (df8[c].sum() / cust_count) * 100 for c in touch_cols}
print("Case 8 Self-Check:", {k: round(v, 1) for k, v in shares8.items()}, f"Sum = {sum(shares8.values()):.1f}%")

fig, (ax_bad, ax_good) = plt.subplots(1, 2, figsize=(14, 5))
ax_bad.pie(
    shares8.values(),
    labels=shares8.keys(),
    autopct="%1.1f%%",
    colors=["#0072B2", "#D55E00", "#009E73", "#CC79A7"],
)
ax_bad.set_title(
    f"BAD: Pie Forced on Overlapping Multi-Touch Channels (Sum = {sum(shares8.values()):.1f}%)",
    color="red",
    fontweight="bold",
)

# Redesign: Horizontal bar chart representing independent customer touch reach
y_pos = np.arange(len(shares8))
s_keys = list(shares8.keys())
s_vals = list(shares8.values())
sorted_idx = np.argsort(s_vals)
ax_good.barh(
    [s_keys[i].title() for i in sorted_idx],
    [s_vals[i] for i in sorted_idx],
    color="#0072B2",
    height=0.55,
)
ax_good.set_xlim(0, 50)
for i in sorted_idx:
    ax_good.text(s_vals[i] + 0.8, s_keys[i].title(), f"{s_vals[i]:.1f}%", va="center", fontweight="bold")
ax_good.set_title(
    "Independent Touchpoint Penetration (% of Total Converted Customers Touched)",
    fontweight="bold",
)
ax_good.set_xlabel("% of Total Customers")
for s in ("top", "right"):
    ax_good.spines[s].set_visible(False)
plt.tight_layout()
fig.savefig("case08_resolved.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# CASE 9: Ratio distortion / denominator collapse in premium revenue share
# ==============================================================================
df9 = pd.read_csv("data/case09_premium_share.csv")
df9["share"] = (df9["premium_revenue_pkr_m"] / df9["total_revenue_pkr_m"]) * 100
s_first = df9["share"].iloc[0]
s_last = df9["share"].iloc[-1]
print(f"Case 9 Self-Check: First Share={s_first:.1f}%, Last Share={s_last:.1f}%")

fig, (ax_bad, ax_good) = plt.subplots(1, 2, figsize=(14, 5))
q_labels = df9[df9.columns[0]].astype(str)
ax_bad.plot(q_labels, df9["share"], marker="o", color="#D55E00", lw=2)
ax_bad.set_ylim(bottom=0, top=50)
ax_bad.set_title(
    "BAD: Premium Share Looks Like a Massive Success (Rises from 24% to 38%)",
    color="red",
    fontweight="bold",
)
ax_bad.set_ylabel("Premium Share of Revenue (%)")

# Redesign: Plot absolute revenues side by side to reveal standard service collapse
ax_good.plot(
    q_labels,
    df9["total_revenue_pkr_m"],
    marker="o",
    color="#333333",
    lw=2,
    label="Total Revenue",
)
ax_good.plot(
    q_labels,
    df9["premium_revenue_pkr_m"],
    marker="s",
    color="#D55E00",
    lw=2,
    label="Premium Revenue (Flat / Stagnant)",
)
ax_good.fill_between(
    range(len(df9)),
    df9["premium_revenue_pkr_m"],
    df9["total_revenue_pkr_m"],
    color="#0072B2",
    alpha=0.15,
    label="Standard Tier (Severe Loss)",
)
ax_good.set_ylim(bottom=0)
ax_good.set_title(
    "Denominator Fallacy: Premium Revenue Did Not Grow; Standard Tier Revenue Collapsed",
    fontweight="bold",
)
ax_good.set_ylabel("Revenue (PKR Millions)")
ax_good.legend(frameon=False)
for ax in (ax_bad, ax_good):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
plt.tight_layout()
fig.savefig("case09_resolved.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Part A completed successfully. All 9 case pairs generated.")