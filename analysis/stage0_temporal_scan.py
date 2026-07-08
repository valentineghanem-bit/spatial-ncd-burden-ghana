import pandas as pd

RAW = "../data/raw/"

files = {
    "NCD": "noncommunicable_diseases_indicators_gha.csv",
    "GlobalStrategy": "global_strategy_indicators_gha.csv",
    "Alcohol": "global_information_system_on_alcohol_and_health_indicators_gha.csv",
    "GHE_LifeExp": "global_health_estimates_life_expectancy_and_leading_causes_of_death_and_disability_indicators_gh.csv",
    "Nutrition": "nutrition_indicators_gha.csv",
    "HealthFinancing": "health_financing_indicators_gha.csv",
    "HealthSystems": "health_systems_indicators_gha.csv",
    "WHS": "world_health_statistics_indicators_gha.csv",
}

target_indicators = [
    "Premature deaths due to noncommunicable diseases (NCD) as a proportion of all NCD deaths",
    "Probability (%) of dying between age 30 and exact age 70 from any of cardiovascular disease, cancer, diabetes, or chronic respiratory disease",
    "Prevalence of overweight among adults, BMI &GreaterEqual; 25 (age-standardized estimate) (%)",
    "Prevalence of obesity among adults, BMI &GreaterEqual; 30 (age-standardized estimate) (%)",
    "Mean Non-HDL cholesterol, age-standardized",
    "Alcohol, total per capita (15+) consumption (in litres of pure alcohol) (SDG Indicator 3.5.2), three-year average",
    "Ambient and household air pollution attributable death rate (per 100 000 population, age-standardized)",
    "UHC Service Coverage Index (SDG 3.8.1)",
    "Out-of-pocket expenditure as percentage of current health expenditure (CHE) (%)",
    "Out-of-pocket expenditure (OOP) per capita in US$",
    "Wasting prevalence among children under 5 years of age (% weight-for-height <-2 SD), survey-based estimates",
    "Stunting prevalence among children under 5 years of age (% height-for-age <-2 SD), survey-based estimates",
    "Estimate of current tobacco use prevalence (%) (age-standardized rate)",
    "Prevalence of insufficient physical activity among adults aged 18+ years (age-standardized estimate) (%)",
    "Prevalence of diabetes, age-standardized",
    "Cancer, age-standardized death rates (15+), per 100,000 population",
]

dfs = {}
for label, fn in files.items():
    df = pd.read_csv(RAW + fn, low_memory=False, skiprows=[1])
    dfs[label] = df

print("For each target indicator: which file(s) contain it, year range, n rows, dimension breakdown")
for ind in target_indicators:
    print("\n" + "=" * 100)
    print("INDICATOR:", ind)
    for label, df in dfs.items():
        if "GHO (DISPLAY)" not in df.columns:
            continue
        sub = df[df["GHO (DISPLAY)"] == ind]
        if len(sub) == 0:
            continue
        years = pd.to_numeric(sub["STARTYEAR"], errors="coerce")
        dims = sub["DIMENSION (NAME)"].dropna().unique()
        print(f"  [{label}] n={len(sub)} years={years.min()}-{years.max()} dims={list(dims)[:8]}")
