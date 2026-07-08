"""
Stage 3 -- variable_provenance.csv data dictionary, Project 21.
Closes the Stage 1 Master CSV deliverable requirement (Q1_DELIVERABLE_STANDARDS.md
Section 4: "Data dictionary (variable * definition * source * survey/year * units *
range * transform) = variable_provenance.csv").
"""
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "outputs" / "data"
REL = Path(__file__).resolve().parent.parent / "outputs" / "release"
REL.mkdir(parents=True, exist_ok=True)

rows = [
    # --- master_national_panel.csv ---
    dict(variable="year", file="master_national_panel.csv", definition="Calendar year", source="n/a", survey_year="2000-2022", units="year", transform="none"),
    dict(variable="ncd_premature_death_proportion_pct", file="master_national_panel.csv", definition="Premature deaths due to NCDs as % of all NCD deaths", source="WHO GHO (noncommunicable_diseases_indicators_gha.csv)", survey_year="2000-2021", units="%", transform="Both-sexes series extracted"),
    dict(variable="ncd_3070_probability_pct", file="master_national_panel.csv", definition="Probability (%) of dying age 30-70 from CVD/cancer/diabetes/CRD (SDG 3.4 indicator)", source="WHO GHO (noncommunicable_diseases_indicators_gha.csv)", survey_year="2000-2021", units="%", transform="Both-sexes series extracted"),
    dict(variable="overweight_adult_pct_agestd", file="master_national_panel.csv", definition="Adult overweight prevalence (BMI>=25), age-standardized", source="WHO GHO", survey_year="1990-2022", units="%", transform="Both-sexes, age-standardized"),
    dict(variable="obesity_adult_pct_agestd", file="master_national_panel.csv", definition="Adult obesity prevalence (BMI>=30), age-standardized", source="WHO GHO", survey_year="1990-2022", units="%", transform="Both-sexes, age-standardized"),
    dict(variable="diabetes_prevalence_pct_agestd", file="master_national_panel.csv", definition="Diabetes prevalence, age-standardized", source="WHO GHO", survey_year="1990-2022", units="%", transform="Both-sexes, age-standardized"),
    dict(variable="non_hdl_cholesterol_mean_agestd", file="master_national_panel.csv", definition="Mean non-HDL cholesterol, age-standardized", source="WHO GHO", survey_year="1980-2018 (within 2000-2022 panel window)", units="mmol/L", transform="Both-sexes, age-standardized"),
    dict(variable="tobacco_use_pct_agestd", file="master_national_panel.csv", definition="Current tobacco use prevalence, age-standardized", source="WHO GHO", survey_year="2000-2022 (observed years only; 2025/2030 WHO projections excluded)", units="%", transform="Truncated to observed years"),
    dict(variable="insufficient_physical_activity_pct_agestd", file="master_national_panel.csv", definition="Insufficient physical activity prevalence, adults 18+, age-standardized", source="WHO GHO", survey_year="2000-2022", units="%", transform="Both-sexes, age-standardized"),
    dict(variable="alcohol_percapita_litres", file="master_national_panel.csv", definition="Total alcohol per-capita consumption (SDG 3.5.2), 3-yr average", source="WHO GISAH", survey_year="2000-2022", units="litres pure alcohol", transform="3-year rolling average as reported"),
    dict(variable="air_pollution_death_rate_agestd", file="master_national_panel.csv", definition="Ambient+household air pollution attributable death rate, age-standardized", source="WHO Global Strategy indicators (single-sourced, not replicated in WHS)", survey_year="2010-2019", units="per 100,000", transform="Both-sexes, age-standardized"),
    dict(variable="uhc_service_coverage_index", file="master_national_panel.csv", definition="UHC Service Coverage Index (SDG 3.8.1) -- DESCRIPTIVE ONLY, excluded from trend/ML (n=7)", source="WHO Global Strategy indicators", survey_year="2000-2021 (7 obs)", units="index 0-100", transform="none"),
    dict(variable="oop_pct_che", file="master_national_panel.csv", definition="Out-of-pocket expenditure as % of current health expenditure", source="WHO GHO (health_financing_indicators_gha.csv)", survey_year="2000-2022", units="%", transform="none"),
    dict(variable="oop_per_capita_usd", file="master_national_panel.csv", definition="Out-of-pocket expenditure per capita", source="WHO GHO (health_financing_indicators_gha.csv)", survey_year="2000-2022", units="US$", transform="none"),
    # --- master_nutrition_double_burden.csv ---
    dict(variable="wasting_pct_under5_total", file="master_nutrition_double_burden.csv", definition="Wasting prevalence, under-5 (weight-for-height <-2SD), 'Total' stratum", source="WHO GHO (nutrition_indicators_gha.csv)", survey_year="1988-2022 (11 survey rounds)", units="%", transform="'Total' dimension selected explicitly; 'Total (All ages)' fallback"),
    dict(variable="stunting_pct_under5_total", file="master_nutrition_double_burden.csv", definition="Stunting prevalence, under-5 (height-for-age <-2SD), 'Total' stratum", source="WHO GHO (nutrition_indicators_gha.csv)", survey_year="1988-2022 (11 survey rounds)", units="%", transform="'Total' dimension selected explicitly; 'Total (All ages)' fallback"),
    # --- master_district_vulnerability.csv ---
    dict(variable="district", file="master_district_vulnerability.csv", definition="MMDA (Metropolitan/Municipal/District Assembly) name", source="Ghana Statistical Service 2021 PHC (Master_Sheet.xlsx)", survey_year="2021 census", units="text", transform="none"),
    dict(variable="poverty_incidence", file="master_district_vulnerability.csv", definition="District poverty incidence rate", source="GSS 2021 PHC", survey_year="2021", units="%", transform="none"),
    dict(variable="poverty_intensity", file="master_district_vulnerability.csv", definition="District poverty intensity (depth)", source="GSS 2021 PHC", survey_year="2021", units="%", transform="none"),
    dict(variable="literacy_rate", file="master_district_vulnerability.csv", definition="District literacy rate", source="GSS 2021 PHC", survey_year="2021", units="proportion 0-1", transform="1 - (illiterate_pop/total_population)"),
    dict(variable="uninsured_rate", file="master_district_vulnerability.csv", definition="District uninsured population rate", source="GSS 2021 PHC", survey_year="2021", units="proportion 0-1", transform="uninsured_pop/total_population"),
    dict(variable="unemployment_rate", file="master_district_vulnerability.csv", definition="District unemployment rate", source="GSS 2021 PHC", survey_year="2021", units="proportion 0-1", transform="unemployed_pop/(employed_pop+unemployed_pop)"),
    dict(variable="elderly_share_65plus_pct", file="master_district_vulnerability.csv", definition="Population aged 65+ as % of total -- NCD-risk covariate, deliberately EXCLUDED from the composite vulnerability index (near-zero correlation with poverty/literacy, Stage 1 finding); used as the held-out benchmark for Stage 3's index-vs-poverty sensitivity comparison", source="GSS 2021 PHC", survey_year="2021", units="%", transform="100*(male_65plus+female_65plus)/total_population"),
    dict(variable="dependency_ratio", file="master_district_vulnerability.csv", definition="Ratio of dependents (0-14 + 65+) to working-age (15-64) population", source="GSS 2021 PHC", survey_year="2021", units="ratio", transform="dependency_pop/working_age_pop"),
    dict(variable="vulnerability_index_pc1", file="master_district_vulnerability.csv", definition="Composite district structural NCD-risk vulnerability index (first principal component of poverty_incidence, poverty_intensity, illiteracy_rate, uninsured_rate, unemployment_rate, dependency_ratio; z-scored inputs; oriented higher=more vulnerable; PC1 explains 56.5% of variance; Cronbach's alpha=0.801)", source="Derived, this project (analysis/build_vulnerability_index.py)", survey_year="2021 (single cross-section)", units="standardized index score, unbounded", transform="PCA on 6 z-scored components; see outputs/tables/table3a_vulnerability_index_component_loadings.csv"),
    dict(variable="vulnerability_index_pc1_rank", file="master_district_vulnerability.csv", definition="District rank by vulnerability_index_pc1 (1=most vulnerable)", source="Derived, this project", survey_year="2021", units="rank 1-261", transform="descending rank of vulnerability_index_pc1"),
    dict(variable="insurance_access_vulnerability_pc2", file="master_district_vulnerability.csv", definition="Second, independently-named vulnerability dimension (PC2 of the same 6-component PCA) capturing insurance-access gaps -- reported separately per Stage 3 council correction, since uninsured_rate loads near-zero on PC1 (0.043) but strongly on PC2 (0.658) and would be functionally inert if folded into a single collapsed index", source="Derived, this project (analysis/build_vulnerability_index.py)", survey_year="2021 (single cross-section)", units="standardized index score, unbounded", transform="PC2 of the same PCA as vulnerability_index_pc1, oriented so higher uninsured_rate loads positively"),
    dict(variable="insurance_access_vulnerability_pc2_rank", file="master_district_vulnerability.csv", definition="District rank by insurance_access_vulnerability_pc2 (1=most insurance-access-vulnerable)", source="Derived, this project", survey_year="2021", units="rank 1-261", transform="descending rank of insurance_access_vulnerability_pc2"),
]

df = pd.DataFrame(rows)
df.to_csv(REL / "variable_provenance.csv", index=False)
print(f"variable_provenance.csv written: {len(df)} variables documented")
print(df["file"].value_counts())
