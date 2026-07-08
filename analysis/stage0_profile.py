import pandas as pd
import json

RAW = "../data/raw/"

files_ghogh = [
    "noncommunicable_diseases_indicators_gha.csv",
    "global_strategy_indicators_gha.csv",
    "global_information_system_on_alcohol_and_health_indicators_gha.csv",
    "global_health_estimates_life_expectancy_and_leading_causes_of_death_and_disability_indicators_gh.csv",
    "nutrition_indicators_gha.csv",
    "health_financing_indicators_gha.csv",
    "health_systems_indicators_gha.csv",
    "world_health_statistics_indicators_gha.csv",
]

for f in files_ghogh:
    print("=" * 100)
    print(f)
    df = pd.read_csv(RAW + f, low_memory=False)
    print("shape (incl HXL row):", df.shape)
    print("columns:", list(df.columns))
    print("first 3 rows:")
    print(df.head(3).to_string())
    # try dropping HXL row
    if df.shape[0] > 1:
        row1 = df.iloc[0].astype(str).str.contains("#").sum()
        print("row0 hash-tag-cell count:", row1, "/", df.shape[1])

print("=" * 100)
print("Master_Sheet.xlsx")
xl = pd.ExcelFile(RAW + "Master_Sheet.xlsx")
print("sheets:", xl.sheet_names)
for sn in xl.sheet_names:
    dfm = xl.parse(sn)
    print(f"--sheet {sn} shape:", dfm.shape)
    print("columns:", list(dfm.columns))
    print(dfm.head(3).to_string())

print("=" * 100)
print("Ghana_New_260_District.geojson")
with open(RAW + "Ghana_New_260_District.geojson", encoding="utf-8") as fh:
    gj = json.load(fh)
print("type:", gj.get("type"))
feats = gj.get("features", [])
print("n features:", len(feats))
if feats:
    print("sample properties:", feats[0]["properties"])
