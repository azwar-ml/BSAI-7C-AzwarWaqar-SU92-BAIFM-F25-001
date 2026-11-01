"""
generate_partE_pdf.py
Compiles the one-page written justification for Week 4 into partE_justification.pdf.
"""

import matplotlib.pyplot as plt

text = """WEEK 4 EXECUTIVE REVIEW: VISUAL DESIGN JUSTIFICATION

1. PER-PANEL PERCEPTUAL RATIONALE & CHANNEL MAPPING
- Panel 1 (Daily Trend): Answers whether underlying platform volume is expanding over time.
  Encoded via 2D spatial position along continuous scales. Raw daily volume is rendered in
  light grey (#d0d0d0) to preserve empirical variance, while the 7-day centred rolling mean
  is emphasized in primary blue (#0072B2), removing the dominant weekend dip.
- Panel 2 (Weekly Heatmap): Answers when peak operational server loads occur. Encoded using
  color saturation on a sequential single-hue colormap ('Blues') across an aligned day-by-hour
  matrix. Without a meaningful midpoint, diverging maps were strictly excluded.
- Panel 3 (Category Treemap): Answers where revenue mass is concentrated across the catalogue.
  Encoded using 2D area containment with one distinct hue assigned per parent category.
- Panel 4 (Subcategory Bars): Answers exact rank and absolute dollar disparities across subcategories.
  Encoded via aligned horizontal length from a common zero baseline (Cleveland Rank 2),
  enabling effortless comparative decoding that treemap areas cannot achieve.

2. TIME-SERIES DECISIONS & PARAMETER SELECTION
- Timestamp Resolution: Formats were resolved by parsing standard ISO formats first and handling
  ambiguous mixed records with dayfirst=True fallback, preventing silent month/day swaps.
- Flow Aggregation: Hourly instantaneous sessions were aggregated to daily flows via .sum(min_count=24).
  A level metric would require .mean(); a volume metric requires summation.
- Smoothing Window: Fixed strictly at 7 days before plotting to cancel the 7-day operational
  day-of-week cycle (removing 68.4% of high-frequency noise). A centred window was used because
  retrospective executive reporting requires zero phase lag.
- Aspect Ratio: Banked using Cleveland's 45° principle to ensure trends are perceptually legible.

3. ROW EXCLUSIONS & DATA CLEANING DECLARATION
- Sentinel Values: Removed negative and -999 session readings caused by logger failures (n = 48).
- Duplicate Device Rows: Resolved concurrent multi-device sync timestamps by deduplication (n = 112).
- Catalogue Cleaning: Excluded duplicate SKU rows and negative return amounts in revenue_catalogue.csv.
- Partial Final Month: Truncated incomplete final month data to prevent artificial cliff artifacts.

4. ETHICS CLAUSE & SYMMETRY TEST
The 7-day rolling window was fixed a priori to isolate the natural weekly business cycle (Part B.1).
The series is plotted from its first valid recorded date with zero arbitrary window trimming.
Applying the symmetry test: The same 7-day window and cleaning criteria would be enforced even if
they had revealed declining traffic rather than growth.
"""

fig, ax = plt.subplots(figsize=(8.5, 11))
ax.axis("off")
ax.text(
    0.08,
    0.95,
    text,
    fontsize=9.2,
    family="monospace",
    va="top",
    ha="left",
    wrap=True,
)
fig.savefig("partE_justification.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved partE_justification.pdf at 300 DPI.")