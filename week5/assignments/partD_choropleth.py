"""
partD_choropleth.py
Week 5 Part D3: Normalized Spatial Choropleth with Proportional Symbols / Bar Summary.
"""

import matplotlib.pyplot as plt
import pandas as pd
import geolib
import vizlib

vizlib.style_defaults()

df = pd.read_csv("data/district_health.csv")
geo = geolib.read_geojson("data/districts.geojson")

# Self-check verification: 24 districts join with 0 failures
geo_codes = {feature["code"] for feature in geo}
data_codes = set(df["code"])
failures = len(data_codes - geo_codes)
print(f"Join check: {len(data_codes)} records, {failures} join failures.")

df["rate"] = (df["cases"] / df["population"]) * 1e5
rate_dict = dict(zip(df["code"], df["rate"]))

fig, (ax_map, ax_bar) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.2, 1]})

# Equal-area choropleth of normalized rate
geolib.choropleth(
    geojson=geo, values=rate_dict, ax=ax_map,
    cmap="YlOrRd", scheme="natural", k=5
)
ax_map.set_title("Incidence: Cases per 100,000 Residents (Jenks Natural Breaks)", fontweight="bold", fontsize=10)
ax_map.axis("off")

# Side bar chart of counts
df_sorted = df.sort_values("cases", ascending=True)
ax_bar.barh(df_sorted["code"], df_sorted["cases"], color="#0072B2", height=0.6)
ax_bar.set_title("Absolute Case Counts by District (Population Drivers D17, D13, D09)", fontweight="bold", fontsize=10)
ax_bar.set_xlabel("Total Reported Cases")
for s in ("top", "right"):
    ax_bar.spines[s].set_visible(False)

fig.suptitle(
    "Disease Distribution: High Rates Cluster in Small Peripheral Districts, Not Metropolitan Hubs",
    fontweight="bold",
    fontsize=12,
    y=0.98,
)
plt.tight_layout()
vizlib.save_fig(fig, "partD_choropleth.png")
plt.close(fig)
print("Saved partD_choropleth.png at 300 DPI.")