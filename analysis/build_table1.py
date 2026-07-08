"""
Stage 1 -- Table 1, Project 21 (Spatial NCD Burden Ghana).
Table 1a: national NCD-burden/determinants panel descriptive statistics (2000-2022).
Table 1b: nutrition double-burden (wasting/stunting) descriptive statistics (1988-2022, "Total" stratum).
Table 1c: district structural NCD-risk vulnerability determinants (261 districts) descriptive statistics.
Table 1d: district determinant correlation matrix (collinearity check ahead of Stage 3 index build).
"""
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "outputs" / "data"
TAB = Path(__file__).resolve().parent.parent / "outputs" / "tables"
TAB.mkdir(parents=True, exist_ok=True)


def describe(df, cols, label):
    rows = []
    for c in cols:
        s = df[c].dropna()
        rows.append({
            "variable": c, "n": len(s), "n_years_or_districts_total": len(df),
            "missing_pct": round(100 * (1 - len(s) / len(df)), 1),
            "mean": round(s.mean(), 3) if len(s) else None,
            "sd": round(s.std(), 3) if len(s) > 1 else None,
            "min": round(s.min(), 3) if len(s) else None,
            "max": round(s.max(), 3) if len(s) else None,
        })
    out = pd.DataFrame(rows)
    print(f"\n{label}\n", out.to_string(index=False))
    return out

# Table 1a -- national panel
panel = pd.read_csv(OUT / "master_national_panel.csv")
t1a = describe(panel, [c for c in panel.columns if c != "year"], "TABLE 1a -- National NCD panel (2000-2022)")
t1a.to_csv(TAB / "table1a_national_panel_descriptives.csv", index=False)

# Table 1b -- nutrition double burden
nutr = pd.read_csv(OUT / "master_nutrition_double_burden.csv")
t1b = describe(nutr, [c for c in nutr.columns if c != "year"], "TABLE 1b -- Nutrition double burden (1988-2022, 'Total' stratum)")
t1b.to_csv(TAB / "table1b_nutrition_descriptives.csv", index=False)

# Table 1c -- district determinants
dist = pd.read_csv(OUT / "master_district_vulnerability.csv")
DIST_COLS = ["poverty_incidence", "poverty_intensity", "literacy_rate", "uninsured_rate",
             "unemployment_rate", "elderly_share_65plus_pct", "dependency_ratio", "total_population"]
t1c = describe(dist, DIST_COLS, "TABLE 1c -- District structural NCD-risk vulnerability determinants (N=261)")
t1c.to_csv(TAB / "table1c_district_descriptives.csv", index=False)

# Table 1d -- district determinant correlation matrix (collinearity check ahead of Stage 3 index build)
corr = dist[DIST_COLS].corr(numeric_only=True).round(3)
print("\nTABLE 1d -- District determinant correlation matrix\n", corr.to_string())
corr.to_csv(TAB / "table1d_district_determinant_correlation.csv")

print("\nN districts check:", dist["district"].nunique(), "(expect 261)")
