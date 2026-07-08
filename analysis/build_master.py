"""
Stage 1 -- Master CSV builds, Project 21 (Spatial NCD Burden Ghana).
Builds three master tables per the council-approved dual-track design
(docs/stage0_council_verdict.md):
  1. outputs/data/master_national_panel.csv          -- 2000-2022 national NCD-burden/determinants panel
  2. outputs/data/master_nutrition_double_burden.csv  -- 1988-2022 wasting/stunting survey-based series (own trajectory)
  3. outputs/data/master_district_vulnerability.csv   -- 261-district structural NCD-risk vulnerability inputs

Mandate 2 (council, Stage 0): per-indicator disaggregation applied. Cancer mortality (n=1,
2019 only) and UHC Service Coverage Index (n=7, 2000-2021) are demoted to descriptive/
contextual-only columns -- never used for Mann-Kendall/LASSO/RF/XGBoost. Tobacco truncated
to observed years only (raw file carries WHO 2025/2030 projection years -- excluded).
Air pollution's single-source status (global_strategy file only, not replicated in WHS)
is disclosed via the panel metadata step below.

Mandate 3 (council, Stage 0): RF/XGBoost+SHAP target locked = year-over-year change in
the 30-70y multi-NCD mortality probability ("ncd_3070_probability_pct"), predicted from
obesity, cholesterol, alcohol, tobacco, physical-inactivity, OOP, and air-pollution
covariates. Non-circular: target is a distinct WHO indicator family from every feature,
no algebraic relationship. Small-N (n<=22 after differencing) mandates leave-one-out CV
only, no held-out test split, SHAP interpreted descriptively not confirmatorily (Stage 4).

Mandate 7 (council, Stage 0): wasting/stunting stratum-selection rule -- "Total" dimension
selected as the primary unstratified national series (66 rows, most complete coverage);
"Total (All ages)" used as fallback only where "Total" is absent for a given year. This
is disclosed explicitly, not a silent average across incompatible wealth/education groupings.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def load_gho(fname):
    df = pd.read_csv(RAW / fname, skiprows=[1])
    df["YEAR (DISPLAY)"] = pd.to_numeric(df["YEAR (DISPLAY)"], errors="coerce")
    df["STARTYEAR"] = pd.to_numeric(df["STARTYEAR"], errors="coerce")
    df["Numeric"] = pd.to_numeric(df["Numeric"], errors="coerce")
    return df


def series_both_sexes(df, indicator, year_col="STARTYEAR"):
    """Extract a national annual series preferring 'Both sexes' dimension; falls back
    to the undimensioned row if no sex dimension exists for this indicator."""
    sub = df[df["GHO (DISPLAY)"] == indicator]
    both = sub[sub["DIMENSION (NAME)"] == "Both sexes"]
    src = both if not both.empty else sub
    return src[[year_col, "Numeric"]].dropna().groupby(year_col)["Numeric"].mean()


# ---------------------------------------------------------------------------
# 1. NATIONAL NCD-BURDEN / DETERMINANTS PANEL -- primary window 2000-2022
# ---------------------------------------------------------------------------
PANEL_YEARS = range(2000, 2023)
panel = pd.DataFrame({"year": PANEL_YEARS}).set_index("year")

ncd = load_gho("noncommunicable_diseases_indicators_gha.csv")
gstrat = load_gho("global_strategy_indicators_gha.csv")
alc = load_gho("global_information_system_on_alcohol_and_health_indicators_gha.csv")
fin = load_gho("health_financing_indicators_gha.csv")

CORE_SERIES = {
    # source_df, indicator, column name
    "ncd_premature_death_proportion_pct": (ncd, "Premature deaths due to noncommunicable diseases (NCD) as a proportion of all NCD deaths"),
    "ncd_3070_probability_pct": (ncd, "Probability (%) of dying between age 30 and exact age 70 from any of cardiovascular disease, cancer, diabetes, or chronic respiratory disease"),
    "overweight_adult_pct_agestd": (ncd, "Prevalence of overweight among adults, BMI &GreaterEqual; 25 (age-standardized estimate) (%)"),
    "obesity_adult_pct_agestd": (ncd, "Prevalence of obesity among adults, BMI &GreaterEqual; 30 (age-standardized estimate) (%)"),
    "diabetes_prevalence_pct_agestd": (ncd, "Prevalence of diabetes, age-standardized"),
    "non_hdl_cholesterol_mean_agestd": (ncd, "Mean Non-HDL cholesterol, age-standardized"),
    "tobacco_use_pct_agestd": (ncd, "Estimate of current tobacco use prevalence (%) (age-standardized rate)"),
    "insufficient_physical_activity_pct_agestd": (ncd, "Prevalence of insufficient physical activity among adults aged 18+ years (age-standardized estimate) (%)"),
    "alcohol_percapita_litres": (alc, "Alcohol, total per capita (15+) consumption (in litres of pure alcohol) (SDG Indicator 3.5.2), three-year average"),
    "air_pollution_death_rate_agestd": (gstrat, "Ambient and household air pollution attributable death rate (per 100 000 population, age-standardized)"),
    "uhc_service_coverage_index": (gstrat, "UHC Service Coverage Index (SDG 3.8.1)"),  # descriptive-only, n=7
    "oop_pct_che": (fin, "Out-of-pocket expenditure as percentage of current health expenditure (CHE) (%)"),
    "oop_per_capita_usd": (fin, "Out-of-pocket expenditure (OOP) per capita in US$"),
}

for col, (src_df, indicator) in CORE_SERIES.items():
    sub = series_both_sexes(src_df, indicator)
    panel[col] = sub.reindex(panel.index)

# Cancer mortality -- single 2019 cross-section (n=6 rows, 1 year). NEVER reindexed onto
# the annual panel as if it were a trend; stored separately as a scalar descriptive fact.
cancer_2019 = series_both_sexes(alc, "Cancer, age-standardized death rates (15+), per 100,000 population")
cancer_mortality_2019_value = float(cancer_2019.iloc[0]) if not cancer_2019.empty else np.nan

# Truncate tobacco to OBSERVED years only (raw file carries 2025/2030 WHO projections)
panel.loc[panel.index > 2022, "tobacco_use_pct_agestd"] = np.nan  # no-op given PANEL_YEARS bound, kept for clarity/audit

panel = panel.reset_index()
panel.to_csv(OUT / "master_national_panel.csv", index=False)
print("National panel (2000-2022):", panel.shape)
print(panel.isna().mean().round(2).sort_values(ascending=False))
print("\nCancer age-standardized death rate, 2019 (single cross-section, NOT in panel):", cancer_mortality_2019_value)

# ---------------------------------------------------------------------------
# 1b. Panel data-quality / forecast-tier metadata (Mandate 2 disaggregation lock)
# ---------------------------------------------------------------------------
DESCRIPTIVE_ONLY_COLS = {"uhc_service_coverage_index"}  # n=7, cannot support Mann-Kendall/LASSO/RF
SINGLE_SOURCE_COLS = {"air_pollution_death_rate_agestd"}  # not replicated in WHS, unlike every other checked indicator

miss = panel.drop(columns=["year"]).isna().mean()
tier = pd.DataFrame({"variable": miss.index, "missing_pct_2000_2022": miss.values.round(3)})
tier["role"] = np.where(
    tier["variable"].isin(DESCRIPTIVE_ONLY_COLS), "descriptive_only_excluded_from_trend_and_ml",
    np.where(tier["missing_pct_2000_2022"] >= 0.60, "sparse_flag_for_stage4_review", "trend_and_ml_eligible")
)
tier["single_sourced_unreplicated_in_whs"] = tier["variable"].isin(SINGLE_SOURCE_COLS)
tier.to_csv(OUT / "national_panel_variable_tier_metadata.csv", index=False)
print("\nVariable tier metadata:\n", tier.to_string(index=False))

# ---------------------------------------------------------------------------
# 1c. RF/XGBoost+SHAP target lock (Mandate 3 -- council Stage 0 resolution,
#     CORRECTED at Stage 1 council gate 2026-07-04, see docs/stage1_council_verdict.md
#     Claim 1). Original 8-feature spec yielded complete-case n=2 (verified) --
#     the Spatial & ML Auditor's "blocking omission" objection was correct and
#     decisive. Tobacco (65.2% missing) and air pollution (56.5% missing) are
#     DROPPED from the feature set; they remain in the descriptive national
#     panel and are still reported/disclosed, just not fed into the ML model.
#     Complete-case n with the reduced 6-feature set = 18 (verified).
# ---------------------------------------------------------------------------
RF_XGB_TARGET = "ncd_3070_probability_pct_yoy_change"
RF_XGB_FEATURES = [
    "overweight_adult_pct_agestd", "obesity_adult_pct_agestd", "non_hdl_cholesterol_mean_agestd",
    "alcohol_percapita_litres", "insufficient_physical_activity_pct_agestd", "oop_pct_che",
]  # tobacco_use_pct_agestd and air_pollution_death_rate_agestd DROPPED -- Stage 1 council correction
target_series = panel.set_index("year")["ncd_3070_probability_pct"].diff().rename(RF_XGB_TARGET)
ml_frame = panel.set_index("year")[RF_XGB_FEATURES].join(target_series)
ml_frame_complete_n = ml_frame.dropna().shape[0]
ml_frame.to_csv(OUT / "national_ml_modelling_frame.csv")
print(f"\nRF/XGBoost+SHAP target locked: {RF_XGB_TARGET}")
print(f"Features (reduced, post Stage-1-council correction): {RF_XGB_FEATURES}")
print(f"Complete-case n (all features + target simultaneously non-missing): {ml_frame_complete_n} "
      f"(verified -- original 8-feature spec gave complete-case n=2, rejected)")
print("Small-N mandate: leave-one-out CV only, no held-out test split; report per-fold R2/MAE "
      "variance explicitly at Stage 4; SHAP interpreted descriptively, not confirmatorily; a "
      "permutation-null baseline (shuffle target, refit, compare SHAP magnitude distributions) "
      "is mandatory before any SHAP ranking is presented as a substantive finding.")

# ---------------------------------------------------------------------------
# 2. NUTRITION DOUBLE-BURDEN -- wasting/stunting, 1988-2022, own trajectory
#    (Mandate 7: "Total" dimension selected explicitly, NOT silently averaged
#    across incompatible wealth-decile/tercile/quintile/education strata)
# ---------------------------------------------------------------------------
nutr = load_gho("nutrition_indicators_gha.csv")

def total_stratum_series(df, indicator):
    sub = df[df["GHO (DISPLAY)"] == indicator]
    total = sub[sub["DIMENSION (NAME)"] == "Total"]
    if total.empty:
        total = sub[sub["DIMENSION (NAME)"] == "Total (All ages)"]
    return total[["STARTYEAR", "Numeric"]].dropna().groupby("STARTYEAR")["Numeric"].mean()

nutr_years = range(1988, 2023)
nutrition_panel = pd.DataFrame({"year": nutr_years}).set_index("year")
NUTR_SERIES = {
    "wasting_pct_under5_total": "Wasting prevalence among children under 5 years of age (% weight-for-height <-2 SD), survey-based estimates",
    "stunting_pct_under5_total": "Stunting prevalence among children under 5 years of age (% height-for-age <-2 SD), survey-based estimates",
}
for col, indicator in NUTR_SERIES.items():
    sub = total_stratum_series(nutr, indicator)
    nutrition_panel[col] = sub.reindex(nutrition_panel.index)

nutrition_panel = nutrition_panel.reset_index()
nutrition_panel.to_csv(OUT / "master_nutrition_double_burden.csv", index=False)
print("\nNutrition double-burden panel (1988-2022, 'Total' stratum):", nutrition_panel.shape)
print(nutrition_panel.dropna(how="all", subset=["wasting_pct_under5_total", "stunting_pct_under5_total"]))

# ---------------------------------------------------------------------------
# 3. DISTRICT STRUCTURAL NCD-RISK VULNERABILITY INDEX INPUTS (261 MMDAs)
#    -- structural determinants only; NOT disease data (council Claim 1/2)
# ---------------------------------------------------------------------------
ms = pd.read_excel(RAW / "Master_Sheet.xlsx")
ms = ms.rename(columns={
    "Metropolitan, Municipal, and District Assemblies (MMDA's)": "district",
    "Region": "region",
    "Class": "class_type",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Employed Population": "employed_pop",
    "Unemployed Population": "unemployed_pop",
    "Incidence of Poverty": "poverty_incidence",
    "Intensity of Poverty": "poverty_intensity",
    "Illiterate Population": "illiterate_pop",
    "Uninsured Population": "uninsured_pop",
    "Male Population 0-14": "male_0_14",
    "Female Population 0-14": "female_0_14",
    "Male Population 15-64": "male_15_64",
    "Female Population 15-64": "female_15_64",
    "Male Population 65+": "male_65plus",
    "Female Population 65+": "female_65plus",
    "Total Population": "total_population",
})
ms["literacy_rate"] = 1 - (ms["illiterate_pop"] / ms["total_population"])
ms["uninsured_rate"] = ms["uninsured_pop"] / ms["total_population"]
ms["unemployment_rate"] = ms["unemployed_pop"] / (ms["employed_pop"] + ms["unemployed_pop"])
# NCD-relevant content upgrade (Synthesis Lead, Stage 0): elderly (65+) population share --
# directly relevant NCD-risk covariate not emphasized in prior WASH/NTD/TB/Hepatitis indices.
ms["elderly_65plus_pop"] = ms["male_65plus"] + ms["female_65plus"]
ms["elderly_share_65plus_pct"] = 100 * ms["elderly_65plus_pop"] / ms["total_population"]
ms["working_age_pop"] = ms["male_15_64"] + ms["female_15_64"]
ms["dependency_pop"] = (ms["male_0_14"] + ms["female_0_14"] + ms["elderly_65plus_pop"])
ms["dependency_ratio"] = ms["dependency_pop"] / ms["working_age_pop"]

ms.to_csv(OUT / "master_district_vulnerability.csv", index=False)
print("\nDistrict structural NCD-risk vulnerability master:", ms.shape)
print("Missing check:", ms[["poverty_incidence", "poverty_intensity", "literacy_rate", "uninsured_rate",
                              "elderly_share_65plus_pct", "latitude", "longitude"]].isna().sum().to_dict())
print("N districts:", ms["district"].nunique(), "(expect 261 per standing ghana-261-districts rule)")
