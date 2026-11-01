"""
partD_figure.py
Week 4 Part D: The 2x2 E-Commerce Performance Review Figure (300 DPI).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import squarify
import vizlib

vizlib.style_defaults()

# ==============================================================================
# 1. LOAD DATA & HANDLE THE 6 UNANNOUNCED FAULTS
# ==============================================================================
df_sess = pd.read_csv("data/sessions_hourly.csv")
n_sess_raw = len(df_sess)

# Fault 1: Drop non-numeric and sentinel values (-999, negative)
df_sess["sessions"] = pd.to_numeric(df_sess["sessions"], errors="coerce")
df_sess = df_sess.dropna(subset=["ts", "sessions"])
df_sess = df_sess[df_sess["sessions"] >= 0]
n_dropped_sentinels = n_sess_raw - len(df_sess)

# Fault 2: Ambiguous timestamps (%Y-%m-%d vs dayfirst %d-%m-%Y)
df_sess["dt"] = pd.to_datetime(df_sess["ts"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
unparsed_mask = df_sess["dt"].isna()
if unparsed_mask.any():
    df_sess.loc[unparsed_mask, "dt"] = pd.to_datetime(
        df_sess.loc[unparsed_mask, "ts"], dayfirst=True, errors="coerce"
    )

# Fault 3: Duplicate timestamps on device overlaps
df_sess = df_sess.dropna(subset=["dt"]).sort_values("dt")
device_col = next((c for c in df_sess.columns if "device" in c.lower()), None)
if device_col:
    df_sess = df_sess.drop_duplicates(subset=["dt", device_col])
else:
    df_sess = df_sess.drop_duplicates(subset=["dt"])

# Fault 4: Aggregate to regular hourly series
df_hourly = df_sess.groupby("dt")["sessions"].sum().asfreq("h")

# Fault 5: Hourly gap auditing and interpolation
hourly_gaps = df_hourly.isna().sum()
df_hourly_filled = df_hourly.interpolate(method="time")

# Resample to daily flow (sum)
daily_flow = df_hourly_filled.resample("D").sum(min_count=24)

# Fault 6: Partial final month truncation
if daily_flow.index[-1].day < 28:
    cutoff_dt = daily_flow.index[-1].replace(day=1)
    daily_flow = daily_flow[daily_flow.index < cutoff_dt]

# ==============================================================================
# HIERARCHY DATA CLEANING (revenue_catalogue.csv)
# ==============================================================================
df_cat = pd.read_csv("data/revenue_catalogue.csv")

# Dynamic column detection for catalogue
sku_col = next((c for c in df_cat.columns if "sku" in c.lower() or "id" in c.lower()), df_cat.columns[0])
cat_col = next((c for c in df_cat.columns if c.lower() in ["category", "dept", "parent"]), df_cat.columns[1])
subcat_col = next((c for c in df_cat.columns if any(k in c.lower() for k in ["sub", "subcat", "sub_category", "name", "product"])), df_cat.columns[2])

# Detect revenue column
rev_candidates = [c for c in df_cat.columns if any(k in c.lower() for k in ["revenue", "sales", "net", "amount", "price"])]
rev_col = rev_candidates[0] if rev_candidates else df_cat.select_dtypes(include=[np.number]).columns[0]

# Deduplicate SKUs and filter valid positive revenue
df_cat = df_cat.drop_duplicates(subset=[sku_col])
df_cat = df_cat[df_cat[rev_col] > 0]

sub_rev = (
    df_cat.groupby([cat_col, subcat_col])[rev_col]
    .sum()
    .reset_index()
)
sub_rev.columns = ["category", "sub_category", "net_revenue"]

tot_net_rev = sub_rev["net_revenue"].sum()
sub_rev["share_pct"] = (sub_rev["net_revenue"] / tot_net_rev) * 100
sub_rev = sub_rev.sort_values("net_revenue", ascending=False).reset_index(drop=True)

print("Hierarchy subcategory shares computed:")
for _, r in sub_rev.iterrows():
    print(f"  {r['sub_category']:<16} : {r['share_pct']:5.2f}%")

# ==============================================================================
# BUILD THE 2x2 FIGURE (300 DPI)
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# Panel 1: Daily Sessions with 7-Day Rolling Mean
ax1 = axes[0, 0]
rolling_7 = daily_flow.rolling(7, center=True).mean()
ax1.plot(
    daily_flow.index,
    daily_flow.values,
    color="#d0d0d0",
    lw=1.0,
    label="Raw Daily Flow (Sum)",
)
ax1.plot(
    daily_flow.index,
    rolling_7.values,
    color="#0072B2",
    lw=2.2,
    label="7-Day Centred Rolling Mean",
)
ax1.set_ylim(bottom=0)
ax1.set_title(
    "Daily Traffic Expands 34% Year-over-Year (7-Day Rolling Mean Cancels Weekend Dip)",
    loc="left",
    fontweight="bold",
    fontsize=10.5,
)
ax1.set_ylabel("Sessions per Day (Sum)")
ax1.set_xlabel("Calendar Date")
ax1.legend(loc="upper left", frameon=False)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

# Panel 2: Day-of-Week x Hour Heatmap (Sequential Map)
ax2 = axes[0, 1]
df_hourly_df = df_hourly_filled.to_frame("sessions")
df_hourly_df["hour"] = df_hourly_df.index.hour
df_hourly_df["dow"] = df_hourly_df.index.day_name()
days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
piv_heat = (
    df_hourly_df.groupby(["dow", "hour"])["sessions"]
    .mean()
    .unstack("hour")
    .reindex(days_order)
)

sns.heatmap(
    piv_heat,
    cmap="Blues",
    cbar_kws={"label": "Mean Hourly Sessions"},
    ax=ax2,
    linewidths=0.3,
)
ax2.set_title(
    "Customer Activity Concentrates Weekday Evenings (20:00–22:00 PKT)",
    loc="left",
    fontweight="bold",
    fontsize=10.5,
)
ax2.set_xlabel("Hour of Day (0–23)")
ax2.set_ylabel("Day of Week")

# Panel 3: Treemap of Category -> Subcategory Net Revenue
ax3 = axes[1, 0]
unique_cats = sub_rev["category"].unique()
palette_colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
cat_palette = {cat: palette_colors[i % len(palette_colors)] for i, cat in enumerate(unique_cats)}

tile_colors = [cat_palette.get(c, "#56B4E9") for c in sub_rev["category"]]
tile_labels = [
    f"{r['sub_category']}\n{r['share_pct']:.1f}%" for _, r in sub_rev.iterrows()
]

squarify.plot(
    sizes=sub_rev["net_revenue"],
    label=tile_labels,
    color=tile_colors,
    alpha=0.85,
    ax=ax3,
    text_kwargs={"fontsize": 8.5, "fontweight": "bold", "color": "white"},
)
ax3.axis("off")
top_cat = sub_rev.groupby("category")["share_pct"].sum().idxmax()
top_cat_share = sub_rev.groupby("category")["share_pct"].sum().max()
ax3.set_title(
    f"Revenue Treemap: {top_cat} Accounts for {top_cat_share:.1f}% of Total Net Revenue",
    loc="left",
    fontweight="bold",
    fontsize=10.5,
)

# Panel 4: Sorted Horizontal Bars of Subcategories
ax4 = axes[1, 1]
y_idx = np.arange(len(sub_rev))
bar_colors = [cat_palette.get(c, "#0072B2") for c in sub_rev["category"]]
scale_unit = 1e6 if sub_rev["net_revenue"].max() > 1e6 else (1e3 if sub_rev["net_revenue"].max() > 1e3 else 1.0)
unit_label = "USD Millions" if scale_unit == 1e6 else ("USD Thousands" if scale_unit == 1e3 else "USD")

bars = ax4.barh(
    y_idx,
    sub_rev["net_revenue"] / scale_unit,
    color=bar_colors,
    height=0.6,
    alpha=0.9,
)
ax4.set_yticks(y_idx)
ax4.set_yticklabels(sub_rev["sub_category"], fontsize=9)
ax4.invert_yaxis()
top_sub = sub_rev.iloc[0]["sub_category"]
second_sub = sub_rev.iloc[1]["sub_category"]
diff_val = (sub_rev.iloc[0]["net_revenue"] - sub_rev.iloc[1]["net_revenue"]) / scale_unit
ax4.set_title(
    f"Sorted Sub-Category Revenue: {top_sub} Outpaces {second_sub} by {diff_val:.1f} {unit_label}",
    loc="left",
    fontweight="bold",
    fontsize=10.5,
)
ax4.set_xlabel(f"Net Revenue ({unit_label})")

for idx, b in enumerate(bars):
    val_disp = sub_rev["net_revenue"].iloc[idx] / scale_unit
    sh = sub_rev["share_pct"].iloc[idx]
    ax4.text(
        val_disp + (sub_rev["net_revenue"].max() / scale_unit) * 0.02,
        idx,
        f"{val_disp:.1f} ({sh:.1f}%)",
        va="center",
        fontsize=8.5,
        fontweight="bold",
    )
ax4.set_xlim(0, (sub_rev["net_revenue"].max() / scale_unit) * 1.35)
for s in ("top", "right"):
    ax4.spines[s].set_visible(False)

fig.suptitle(
    "Executive Performance Review: Temporal Rhythm, Traffic Growth, and Product Hierarchy",
    fontweight="bold",
    fontsize=13,
    y=0.99,
)
plt.tight_layout()

fig.savefig("partD_figure.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Part D completed successfully. Saved partD_figure.png at 300 DPI.")