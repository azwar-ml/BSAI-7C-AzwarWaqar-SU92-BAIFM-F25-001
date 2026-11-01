"""
generate_partE_pdf.py
Week 5 Part E: One-page PDF justification for spatial and multivariate decisions.
"""

import matplotlib.pyplot as plt

text = """WEEK 5 MULTIVARIATE & GEOSPATIAL VISUALIZATION: JUSTIFICATION

1. CHANNEL BUDGET ALLOCATION (SLIDE 8 PERCEPTUAL RANKING)
- Spatial Position (X/Y): Allocated to GDP per capita (log scale) and Healthcare Access Index.
  Position on an aligned scale is Cleveland's most accurate visual channel, essential for detecting
  non-linear saturation curves.
- Mark Area (Bubble Size): Allocated to Total Population (pop_m). Area is mapped strictly linearly
  (s = k * pop_m) so perceived circle surface area matches population magnitude, avoiding the r^2 trap.
- Hue (Color Identity): Allocated to Region (categorical). Six distinct Okabe-Ito hues encode geographic
  identity without falsely implying numerical magnitude.

2. PARALLEL COORDINATES DECISIONS (SLIDES 22 & 24)
- Scaling Method: Percentile rank scaling was selected over min-max. Min-max collapsed 58/60 nations below
  0.10 due to severe right-skewed GDP outliers, whereas rank scaling restores structural variance across all axes.
- Axis Ordering: Constructed via greedy correlation chaining: Healthcare Access -> Cases per 100k ->
  Urban Pct -> Median Age -> GDP per Capita. Placing Healthcare Access adjacent to Cases per 100k exposes
  an immediate X-shaped inverse crossing (r = -0.799) that pairwise scatter matrices obscure.
- Clutter Control: All 60 countries rendered in faint grey (#d0d0d0, alpha=0.45); the focal region is
  brushed in bold primary orange (#D55E00) to highlight structural divergence.

3. SPATIAL DECISIONS & CLASSIFICATION (SLIDES 33-50)
- Denominator: Strictly normalized to cases per 100,000 residents. A raw-count map is merely a population
  density map (r = 0.986 with population).
- Classification Scheme: Jenks natural breaks (k=5) selected to minimize intra-class variance and maximize
  inter-class disparity without manufacturing false gradients (unlike quantiles).
- Projection: Equal-area projection utilized. Choropleths encode numerical values via colored area; using
  conformal or unprojected coordinates introduces Mercator area distortion that magnifies northern units.

4. COUNTERFACTUAL ETHICS DECLARATION (SLIDE 48)
If raw case counts had been mapped, the three most populous urban centers (D17, D13, D09) would have been
falsely labeled as the disease epicenters. In reality, their infection rate per capita is among the lowest
in the federation. Publishing raw counts would have misallocated medical quarantine resources to cities
while abandoning peripheral districts (D19, D20, D21) where the per-capita incidence is nearly 4x higher.
"""

fig, ax = plt.subplots(figsize=(8.5, 11))
ax.axis("off")
ax.text(0.08, 0.95, text, fontsize=9.2, family="monospace", va="top", ha="left", wrap=True)
plt.tight_layout()
fig.savefig("partE_justification.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved partE_justification.pdf at 300 DPI.")