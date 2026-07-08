"""
Stage 5 -- Mann-Kendall trend tests on the national NCD panel, closing a gap
identified during manuscript Results drafting: Mann-Kendall was referenced
throughout Stages 0-4 as the method that would be applied to trend-eligible
series, but was never actually executed. Applied only to series classified
'trend_and_ml_eligible' in Stage 1's variable-tier metadata (excludes
uhc_service_coverage_index [n=7, descriptive-only] and tobacco/air-pollution
per their own sparse-series caveats already logged).
"""
import pandas as pd
import pymannkendall as mk
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "outputs" / "data"
TAB = Path(__file__).resolve().parent.parent / "outputs" / "tables"

panel = pd.read_csv(OUT / "master_national_panel.csv")
tier = pd.read_csv(OUT / "national_panel_variable_tier_metadata.csv")
eligible = tier[tier["role"] == "trend_and_ml_eligible"]["variable"].tolist()
print("Series eligible for Mann-Kendall (trend_and_ml_eligible per Stage 1 tier metadata):")
print(eligible)

results = []
for col in eligible:
    series = panel[["year", col]].dropna()
    if len(series) < 4:
        continue
    result = mk.original_test(series[col].values)
    results.append({
        "variable": col,
        "n_obs": len(series),
        "year_range": f"{int(series['year'].min())}-{int(series['year'].max())}",
        "trend": result.trend,
        "mk_statistic_S": result.s,
        "tau": round(result.Tau, 4),
        "p_value": round(result.p, 5),
        "sens_slope_per_year": round(result.slope, 5),
        "first_value": round(series[col].iloc[0], 3),
        "last_value": round(series[col].iloc[-1], 3),
    })
    print(f"{col}: trend={result.trend}, S={result.s}, tau={result.Tau:.4f}, p={result.p:.5f}, "
          f"Sen's slope={result.slope:.5f}/yr, {series[col].iloc[0]:.2f} ({int(series['year'].iloc[0])}) "
          f"-> {series[col].iloc[-1]:.2f} ({int(series['year'].iloc[-1])})")

# Also run on the two Track 1 headline NCD outcomes (already in eligible list but
# highlighted separately since they are the primary manuscript outcomes)
results_df = pd.DataFrame(results)
results_df.to_csv(TAB / "table5a_mann_kendall_trends.csv", index=False)
print(f"\nSaved: outputs/tables/table5a_mann_kendall_trends.csv ({len(results_df)} series tested)")

# Nutrition double-burden series -- Stage 4 council ruling: descriptive-only,
# NO Mann-Kendall (68.6% missing, 11 irregular survey points) -- confirmed excluded
print("\nWasting/stunting: EXCLUDED from Mann-Kendall per Stage 4 council ruling (descriptive-only, 68.6% missing).")
