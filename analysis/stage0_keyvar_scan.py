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

keywords = [
    "premature", "30-70", "30 to 70", "probability of dying",
    "overweight", "obesity", "obese",
    "cholesterol", "non-hdl", "non hdl",
    "alcohol",
    "cardiovascular", "cancer",
    "air pollution",
    "out-of-pocket", "out of pocket", "oop",
    "universal health coverage", "uhc",
    "wasting", "stunting", "stunted", "wasted",
    "raised blood pressure", "hypertension", "blood glucose", "diabetes",
    "tobacco", "physical activity", "salt", "sodium",
]

dfs = {}
for label, fn in files.items():
    df = pd.read_csv(RAW + fn, low_memory=False, skiprows=[1])
    dfs[label] = df
    print(f"{label}: {fn} -> {df.shape} (post-HXL-drop)")

print("\n" + "=" * 100)
print("KEYWORD SCAN across GHO (DISPLAY) indicator names")
print("=" * 100)

for label, df in dfs.items():
    if "GHO (DISPLAY)" not in df.columns:
        continue
    names = df["GHO (DISPLAY)"].dropna().unique()
    hits = []
    for kw in keywords:
        matched = [n for n in names if kw.lower() in str(n).lower()]
        if matched:
            hits.append((kw, matched))
    if hits:
        print(f"\n--- {label} ({len(names)} distinct indicators) ---")
        for kw, matched in hits:
            for m in set(matched):
                print(f"  [{kw}] {m}")
