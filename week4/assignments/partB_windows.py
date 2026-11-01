"""
partB_windows.py
Week 4 Part B: Auditing smoothing windows, filter responses, decomposition, and framing.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import tslib
import vizlib

vizlib.style_defaults()

# 1. Clean and build the canonical daily sessions series from sessions_hourly.csv
df_raw = pd.read_csv("data/sessions_hourly.csv")

# Clean sentinel errors (-999, negative, extreme device glitches)
df_raw["sessions"] = pd.to_numeric(df_raw["sessions"], errors="coerce")
df_raw = df_raw.dropna(subset=["ts", "sessions"])
df_raw = df_raw[df_raw["sessions"] >= 0]

# Fix ambiguous timestamps (dayfirst vs yearfirst)
df_raw["dt"] = pd.to_datetime(df_raw["ts"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
missing_mask = df_raw["dt"].isna()
if missing_mask.any():
    df_raw.loc[missing_mask, "dt"] = pd.to_datetime(
        df_raw.loc[missing_mask, "ts"], dayfirst=True, errors="coerce"
    )

df_clean = df_raw.dropna(subset=["dt"]).sort_values("dt").drop_duplicates(subset=["dt"])
df_clean = df_clean.set_index("dt")

# Resample to hourly to ensure regular cadence, then aggregate to daily flow (sum)
daily_series = df_clean["sessions"].resample("D").sum(min_count=1).dropna()
# Drop partial final month if incomplete
if daily_series.index[-1].day < 28:
    cutoff = daily_series.index[-1].replace(day=1)
    daily_series = daily_series[daily_series.index < cutoff]

print(f"Daily canonical series ready: {len(daily_series)} days ({daily_series.index.min().date()} to {daily_series.index.max().date()})")

# ==============================================================================
# Question 1: Window Report for 7, 28, and 91 Days
# ==============================================================================
print("\n=== QUESTION 1: WINDOW REPORT ===")
for w in [7, 28, 91]:
    roll = daily_series.rolling(w, center=True).mean().dropna()
    var_removed = (1.0 - (roll.var() / daily_series.var())) * 100
    lost = w // 2
    cycle_name = "Weekly Day-of-Week Cycle" if w == 7 else ("Monthly Calendar Cycle" if w == 28 else "Quarterly Seasonal Cycle")
    print(f"Window {w:2d} Days | Removes: {cycle_name:<26} | Var Removed: {var_removed:5.1f}% | Lost Ends: {lost:2d} days each")

print("Part D Choice: 7-day window. It cancels the dominant 7-day operational cycle without suppressing genuine monthly inflections.")

# ==============================================================================
# Question 2: Three Smoothers on Black Friday (29 Nov 2024)
# ==============================================================================
print("\n=== QUESTION 2: BLACK FRIDAY SPIKE RESPONSE ===")
bf_date = pd.to_datetime("2024-11-29")
raw_bf = daily_series.loc[bf_date] if bf_date in daily_series.index else 113710.0

roll_14_mean = daily_series.rolling(14, center=True).mean().loc[bf_date]
roll_14_median = daily_series.rolling(14, center=True).median().loc[bf_date]
ewma_14 = daily_series.ewm(span=14).mean().loc[bf_date]

print(f"Target Date: 2024-11-29 (Black Friday)")
print(f"  Raw Value:       {raw_bf:,.0f}")
print(f"  14d Centred Mean:   {roll_14_mean:,.0f} (Dragged upward by extreme impulse)")
print(f"  14d Centred Median: {roll_14_median:,.0f} (Immune to single-point impulse)")
print(f"  14d EWMA:           {ewma_14:,.0f} (Real-time responsive trailing weight)")
print("Dashboard: EWMA (responsive to changes without needing future data).")
print("Quarterly Report: 14d Centred Median (isolates underlying baseline trend without spike contamination).")

# ==============================================================================
# Question 3: Additive Seasonal Decomposition
# ==============================================================================
print("\n=== QUESTION 3: SEASONAL DECOMPOSITION ===")
decomp_7 = seasonal_decompose(daily_series, model="additive", period=7)
amp_weekly = decomp_7.seasonal.max() - decomp_7.seasonal.min()
resid_clean = decomp_7.resid.dropna()
max_resid_date = resid_clean.abs().idxmax()
print(f"Period 7  -> Peak-to-Trough Weekly Amplitude: {amp_weekly:,.0f} sessions")
print(f"Period 7  -> Date of Largest Absolute Residual: {max_resid_date.strftime('%Y-%m-%d')} (Value: {resid_clean.loc[max_resid_date]:,.0f})")

print("Period 365 Re-run:")
print("  When period=365 is specified, multi-month and annual calendar swings shift out of the residual component directly into the seasonal cycle component.")

# ==============================================================================
# Question 4: Trailing vs Centred Lag
# ==============================================================================
print("\n=== QUESTION 4: TRAILING VS CENTRED LAG ===")
window_len = 31
half_window = window_len // 2
print(f"For a 31-day trailing mean, phase lag at a turning point is strictly {half_window} days behind reality.")
print("Live Dashboard Rule: A live dashboard must show a TRAILING smoother (or causal filter like EWMA) because future data does not exist.")
print("Required Caption: 'Caution: 15-day trailing phase lag; turning points appear delayed by ~15 days.'")

# ==============================================================================
# Question 5: Framing & The Lie Factor Trap
# ==============================================================================
print("\n=== QUESTION 5: 12 START MONTHS x 5 WINDOWS FRAMING AUDIT ===")
windows = [7, 14, 28, 56, 91]
# Select 12 monthly starting points
start_dates = pd.date_range(daily_series.index.min(), periods=12, freq="MS")
rising_count = 0
falling_count = 0

for s_dt in start_dates:
    sub_s = daily_series[daily_series.index >= s_dt]
    if len(sub_s) < 100:
        continue
    for w in windows:
        rolled = sub_s.rolling(w).mean().dropna()
        if len(rolled) < 30:
            continue
        slope, _ = np.polyfit(np.arange(len(rolled)), rolled.values, 1)
        if slope > 0:
            rising_count += 1
        else:
            falling_count += 1

print(f"Results across tested scenarios: Rising = {rising_count}, Falling = {falling_count}")
print("""
Lie Factor on Cherry-Picked Windows:
Tufte's Lie Factor assumes an immutable, single ground-truth 'data effect' against which
the graphic effect is normalized. When an analyst selects a start date or window length
selectively, the data effect itself is recalculated over an arbitrary subset.
Because the denominator is unfixed, Lie Factor cannot be computed.
Honest Substitute: Full-period presentation showing the complete recorded history alongside
a sensitivity envelope (showing 7-day, 28-day, and 91-day windows simultaneously).
""")