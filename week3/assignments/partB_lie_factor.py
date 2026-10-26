"""
partB_lie_factor.py
Week 3 Part B: Computing Tufte's Lie Factor across Graphics A-E and Chart F.
"""

import pandas as pd
import vizlib

df_lf = pd.read_csv("data/case07_lie_factor.csv")

results = []
for _, row in df_lf.iterrows():
    # Pass exact parameters as defined in the CSV
    lf, effect_data, effect_graphic = vizlib.lie_factor(
        row["value_start"],
        row["value_end"],
        row["ink_start"],
        row["ink_end"],
        ink_dimension=int(row["ink_dimension"]),
    )
    verdict = (
        "Honest"
        if 0.95 <= lf <= 1.05
        else ("Overstated" if lf > 1.05 else "Understated")
    )
    results.append(
        {
            "Chart": row["chart_id"],
            "Subject": row["subject"],
            "Dim": int(row["ink_dimension"]),
            "Data Effect": round(effect_data, 3),
            "Graphic Effect": round(effect_graphic, 3),
            "Lie Factor": round(lf, 3),
            "Verdict": verdict,
        }
    )

res_df = pd.DataFrame(results)
print("=== PART B: LIE FACTOR AUDIT TABLE ===")
print(res_df.to_string(index=False))

print("\n=== THEORETICAL AUDIT ANSWERS ===")
print(
    """
1. Honest Graphic:
   - Identify the graphic where 0.95 <= Lie Factor <= 1.05 from the table above.
   - The remaining graphics distort the underlying data either through baseline
     truncation (exaggerating length) or through 2D area scaling of 1D metrics.

2. Ink Dimension 2 Trap:
   - When a 1D scalar data value is encoded by scaling a 2D symbol, pictogram, or
     circle in BOTH width and height (e.g., Chart B), visual ink expands as area:
         Area_final / Area_initial = (s1 / s0)^2
   - If ink_dimension is left at 1, the audit measures only linear height/width change,
     hiding the quadratic area distortion perceived by the human visual system.

3. Understated Lie Factor (< 1.0):
   - A Lie Factor below 0.95 dampens and compresses real data changes.
   - Incumbents, public relations bodies, or corporate executives use understated
     charts to downplay budget deficits, mask inflation, or minimize public health
     risk surges.

4. Chart F (Dual-Axis Chart from Part A, Case 7):
   - Lie Factor strictly assumes a single affine mapping: y_pixel = m * v + c.
   - A dual-axis chart introduces two completely decoupled, independent scale
     transformations:
         y_pixel_rev = m1 * v_rev + c1
         y_pixel_cost = m2 * v_cost + c2
   - Because m1, m2, c1, and c2 can be arbitrarily chosen to force curves to intersect
     or invert at will, visual distances have no fixed geometric ratio to data units.
   - Therefore, Lie Factor is mathematically undefined for dual-axis charts.
   - What the data actually shows: ShaheenGo's revenue exceeded costs in every
     single month (100% profitable); the crossing was a pure visual artifact of
     mismatched axis ranges.
"""
)