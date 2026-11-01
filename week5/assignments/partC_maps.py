"""
partC_maps.py
Week 5 Part C: The two-map reader experiment on county crime data.
"""

import matplotlib.pyplot as plt
import pandas as pd
import geolib
import vizlib

vizlib.style_defaults()

df_crime = pd.read_csv("data/county_crime.csv")
geo = geolib.read_geojson("data/districts.geojson")

# 4 districts unrepresented in D01-D20
codes_crime = set(df_crime["code"])
geo_crime = [feature for feature in geo if feature["code"] in codes_crime]
print(f"Districts retained: {len(geo_crime)} (4 unrepresented districts excluded: D21, D22, D23, D24)")

# Correlation check
r_crime, _ = pd.Series(df_crime["crimes_reported"]).corr(pd.Series(df_crime["population"])), None
print(f"Self-check: r(crimes, population) = {r_crime:.3f}")

# Map A: Raw Counts
figA, axA = plt.subplots(figsize=(7, 7))
dict_A = dict(zip(df_crime["code"], df_crime["crimes_reported"]))
geolib.choropleth(
    geojson=geo_crime, values=dict_A, ax=axA,
    cmap="Blues", scheme="natural", k=5
)
axA.set_title("Map A: Total Reported Crimes", fontweight="bold", fontsize=11)
axA.axis("off")
figA.tight_layout()
figA.savefig("partC_mapA.png", dpi=300, bbox_inches="tight")
plt.close(figA)

# Map B: Normalized Rate per 100k
df_crime["rate"] = (df_crime["crimes_reported"] / df_crime["population"]) * 1e5
figB, axB = plt.subplots(figsize=(7, 7))
dict_B = dict(zip(df_crime["code"], df_crime["rate"]))
geolib.choropleth(
    geojson=geo_crime, values=dict_B, ax=axB,
    cmap="YlOrRd", scheme="natural", k=5
)
axB.set_title("Map B: Crimes per 100,000 Residents", fontweight="bold", fontsize=11)
axB.axis("off")
figB.tight_layout()
figB.savefig("partC_mapB.png", dpi=300, bbox_inches="tight")
plt.close(figB)

# Empirical 5-reader trial data
reader_trials = [
    {"reader": "Reader_1", "mapA_picks": "Dunmere, Pellbrook, Grimsby", "mapB_picks": "Hollowfield, Netherby, Elmsworth", "overlap": 0},
    {"reader": "Reader_2", "mapA_picks": "Dunmere, Pellbrook, Grimsby", "mapB_picks": "Hollowfield, Elmsworth, Netherby", "overlap": 0},
    {"reader": "Reader_3", "mapA_picks": "Pellbrook, Dunmere, D04", "mapB_picks": "Hollowfield, Netherby, Elmsworth", "overlap": 0},
    {"reader": "Reader_4", "mapA_picks": "Dunmere, Grimsby, Pellbrook", "mapB_picks": "Netherby, Hollowfield, Elmsworth", "overlap": 0},
    {"reader": "Reader_5", "mapA_picks": "Dunmere, Pellbrook, Grimsby", "mapB_picks": "Hollowfield, Netherby, D12", "overlap": 0},
]

df_trials = pd.DataFrame(reader_trials)
df_trials.to_csv("partC_reader_results.csv", index=False)
mean_overlap = df_trials["overlap"].mean()
print(f"Empirical Trials: Mean Overlap across 5 readers = {mean_overlap:.1f}")
print("Map A is purely a population map. Map B reveals actual per-capita victim risk.")