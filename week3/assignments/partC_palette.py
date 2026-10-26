"""
partC_palette.py
Week 3 Part C: Color-vision deficiency simulation and perceptual Delta-E validation.
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import vizlib

# 1. Non-safe candidate: Default unadjusted 4-color palette
unsafe_palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]  # Red, Blue, Green, Purple

# 2. Course Okabe-Ito subset (3 primary + 4th optional slot)
course_palette = vizlib.PALETTE["series"] + ["#CC79A7"]

sim_kinds = ["normal", "deutan", "protan", "tritan"]

fig, axes = plt.subplots(2, 4, figsize=(14, 5))

for col_idx, kind in enumerate(sim_kinds):
    # Unsafe palette simulation
    sim_unsafe = (
        unsafe_palette
        if kind == "normal"
        else vizlib.simulate_cvd(unsafe_palette, kind)
    )
    for i, hex_c in enumerate(sim_unsafe):
        axes[0, col_idx].add_patch(
            patches.Rectangle((i * 0.25, 0), 0.25, 1, color=hex_c)
        )
    axes[0, col_idx].set_xlim(0, 1)
    axes[0, col_idx].set_ylim(0, 1)
    axes[0, col_idx].axis("off")
    axes[0, col_idx].set_title(
        f"Unsafe: {kind.capitalize()}", fontsize=10, fontweight="bold"
    )

    # Course palette simulation
    sim_course = (
        course_palette
        if kind == "normal"
        else vizlib.simulate_cvd(course_palette, kind)
    )
    for i, hex_c in enumerate(sim_course):
        axes[1, col_idx].add_patch(
            patches.Rectangle((i * 0.25, 0), 0.25, 1, color=hex_c)
        )
    axes[1, col_idx].set_xlim(0, 1)
    axes[1, col_idx].set_ylim(0, 1)
    axes[1, col_idx].axis("off")
    axes[1, col_idx].set_title(
        f"Course: {kind.capitalize()}", fontsize=10, fontweight="bold"
    )

fig.suptitle(
    "Palette CVD Simulation: Unsafe Palette Collapses Under Deutan/Protan Vision",
    fontweight="bold",
    y=0.98,
)
plt.tight_layout()
fig.savefig("palette_simulation.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Compute OKLab Delta-E distances
print("=== PART C: WORST PAIR DELTA-E AUDIT ===")
for name, pal in [("Unsafe Palette", unsafe_palette), ("Course Palette (4-slot)", course_palette)]:
    print(f"\nEvaluating: {name}")
    for kind in ["deutan", "protan", "tritan"]:
        sim_p = vizlib.simulate_cvd(pal, kind)
        res = vizlib.worst_pair(sim_p)
        print(f"  {kind.capitalize():7s} -> worst_pair output: {res}")

print(
    """
Rule for >3 categories:
Never add a fifth or sixth hue. Instead, rely on direct labeling, faceted small
multiples (one panel per category), or secondary encodings like line styles and
symbols so that category identification does not depend on color alone.
"""
)