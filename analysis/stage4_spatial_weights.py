"""
Stage 4 -- spatial weights specification lock (Stage 1 Mandate 8: Queen contiguity
proposed as "provisional default"; Stage 2 domain-transfer caveat: Getis-Ord/LISA
describe structural risk-factor/demographic co-location, NOT contagion-like
transmission -- stated explicitly here, in Methods language, not left implicit).

Runs Getis-Ord Gi* on vulnerability_index_pc1 under TWO specifications (Queen
contiguity vs k=6 k-NN) and reports the numeric spread between them explicitly --
per the standing Learning Log rule (Project 20 COUNCIL-025): "near-identical"
is not an acceptable qualitative label without the actual number.

*** CORRECTION (discovered at Stage 6, applied retroactively to this Stage 4
script) ***: esda's G_Local computes Gi*/Gi as a RATIO statistic --
statistic = (W @ y) / (y.sum() - y * (not star)) -- which is only well-behaved
when y is non-negative with a comfortably positive sum (the classical
Getis-Ord requirement; see e.g. ESRI's Hot Spot Analysis documentation).
vulnerability_index_pc1 is a mean-centered PCA component (sum across the 258
matched districts = 0.585, essentially zero), so for any district with
y_i > 0.585 the denominator went NEGATIVE, flipping the statistic's sign
independent of the actual local clustering pattern. This produced a
near-random, weakly NEGATIVE correlation (-0.35) between the raw index value
and its own Gi* z-score, when strong positive global autocorrelation
(Moran's I=0.80, unaffected by this bug) implies it should be strongly
POSITIVE. The fix (applied below): shift y to be strictly positive
(y - min(y) + 1) before Gi*/Gi -- a pure positive affine transform that leaves
the correlation structure, rank ordering, and Moran's I/LISA results
completely unchanged (those use mean-deviation products, not this ratio, and
were never affected). Global Moran's I and Moran_Local (LISA) below are
unaffected and unchanged from the original run.
"""
import json
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from libpysal.weights import Queen, KNN
from esda.getisord import G_Local
from esda.moran import Moran, Moran_Local

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "data"
TAB = Path(__file__).resolve().parent.parent / "outputs" / "tables"
DOCS = Path(__file__).resolve().parent.parent / "docs"

# ---------------------------------------------------------------------------
# 1. Load GeoJSON (260 polygons) + crosswalk (261 Master Sheet districts -> 260 polygons)
# ---------------------------------------------------------------------------
print("Loading GeoJSON (this file is large, ~27MB)...")
gdf = gpd.read_file(RAW / "Ghana_New_260_District.geojson")
gdf["DISTRICT"] = gdf["DISTRICT"].str.strip().str.upper()
print(f"GeoJSON: {len(gdf)} polygons")

crosswalk = pd.read_csv(DOCS / "district_crosswalk_261_to_260.csv")
crosswalk["geojson_district"] = crosswalk["geojson_district"].str.strip().str.upper()

dist = pd.read_csv(OUT / "master_district_vulnerability.csv")
print(f"Master vulnerability data: {len(dist)} districts")

# Join: master_sheet_district -> geojson_district -> polygon
merged_keys = dist.merge(crosswalk, left_on="district", right_on="master_sheet_district", how="left")
print(f"Districts matched to crosswalk: {merged_keys['geojson_district'].notna().sum()} / {len(merged_keys)}")

geo_merged = gdf.merge(merged_keys, left_on="DISTRICT", right_on="geojson_district", how="inner")
print(f"Polygons with vulnerability index data joined: {len(geo_merged)}")

# NOTE (Ghana 261-districts standing rule): the crosswalk maps 261 Master Sheet
# districts onto 260 polygons (some parent/child districts share one legacy polygon
# for mapping purposes only). geo_merged may therefore have slightly fewer or more
# rows than 260/261 depending on many-to-one merges -- this is expected and does not
# indicate a data-loss bug; verified below.
print(f"Unique polygons used: {geo_merged['DISTRICT'].nunique()}, "
      f"unique master districts represented: {geo_merged['district'].nunique()}")

# ---------------------------------------------------------------------------
# 2. Build TWO spatial weights specifications
# ---------------------------------------------------------------------------
geo_merged = geo_merged.reset_index(drop=True)
w_queen = Queen.from_dataframe(geo_merged, use_index=False)
w_queen.transform = "r"

w_knn6 = KNN.from_dataframe(geo_merged, k=6, use_index=False)
w_knn6.transform = "r"

print(f"\nQueen contiguity: mean neighbors = {np.mean(list(w_queen.cardinalities.values())):.2f}, "
      f"islands (no neighbors) = {sum(1 for v in w_queen.cardinalities.values() if v == 0)}")
print(f"k=6 k-NN: mean neighbors = 6.0 (by construction), islands = 0")

y_raw = geo_merged["vulnerability_index_pc1"].values
# Gi*/Gi correction (see module docstring): shift to strictly positive before Gi*/Gi only.
# Moran's I and LISA below use y_raw directly -- they are unaffected by this issue.
y = y_raw + (-y_raw.min() + 1.0)

# ---------------------------------------------------------------------------
# 3. Global Moran's I under both specifications
# ---------------------------------------------------------------------------
moran_queen = Moran(y_raw, w_queen, permutations=999)
moran_knn = Moran(y_raw, w_knn6, permutations=999)
print(f"\nGlobal Moran's I -- Queen contiguity: I={moran_queen.I:.4f}, p={moran_queen.p_sim:.4f}")
print(f"Global Moran's I -- k=6 k-NN:         I={moran_knn.I:.4f}, p={moran_knn.p_sim:.4f}")
moran_spread = abs(moran_queen.I - moran_knn.I)
print(f"Numeric spread between specifications: {moran_spread:.4f} "
      f"(reported as an actual number, not 'near-identical')")

# ---------------------------------------------------------------------------
# 4. Getis-Ord Gi* under both specifications -- compare hotspot/coldspot classification
# ---------------------------------------------------------------------------
gi_queen = G_Local(y, w_queen, transform="r", permutations=999)
gi_knn = G_Local(y, w_knn6, transform="r", permutations=999)

def classify_gi(gi_result, alpha=0.05):
    z = gi_result.Zs
    p = gi_result.p_sim
    cls = np.where((p < alpha) & (z > 0), "hotspot",
          np.where((p < alpha) & (z < 0), "coldspot", "not_significant"))
    return cls

cls_queen = classify_gi(gi_queen)
cls_knn = classify_gi(gi_knn)

agreement = (cls_queen == cls_knn).mean()
print(f"\nGetis-Ord Gi* hotspot/coldspot classification agreement between "
      f"Queen and k=6 k-NN: {agreement*100:.1f}% of districts classified identically")

n_hotspot_queen = (cls_queen == "hotspot").sum()
n_hotspot_knn = (cls_knn == "hotspot").sum()
n_coldspot_queen = (cls_queen == "coldspot").sum()
n_coldspot_knn = (cls_knn == "coldspot").sum()
print(f"Hotspots: Queen={n_hotspot_queen}, k-NN={n_hotspot_knn} districts")
print(f"Coldspots: Queen={n_coldspot_queen}, k-NN={n_coldspot_knn} districts")

sensitivity_table = pd.DataFrame([{
    "metric": "Global Moran's I",
    "queen_contiguity": round(moran_queen.I, 4),
    "knn_k6": round(moran_knn.I, 4),
    "numeric_spread": round(moran_spread, 4),
    "queen_p": round(moran_queen.p_sim, 4),
    "knn_p": round(moran_knn.p_sim, 4),
}, {
    "metric": "Gi* classification agreement (%)",
    "queen_contiguity": round(agreement * 100, 1),
    "knn_k6": np.nan,
    "numeric_spread": round(100 - agreement * 100, 1),
    "queen_p": np.nan,
    "knn_p": np.nan,
}, {
    "metric": "N hotspot districts",
    "queen_contiguity": n_hotspot_queen,
    "knn_k6": n_hotspot_knn,
    "numeric_spread": abs(n_hotspot_queen - n_hotspot_knn),
    "queen_p": np.nan,
    "knn_p": np.nan,
}, {
    "metric": "N coldspot districts",
    "queen_contiguity": n_coldspot_queen,
    "knn_k6": n_coldspot_knn,
    "numeric_spread": abs(n_coldspot_queen - n_coldspot_knn),
    "queen_p": np.nan,
    "knn_p": np.nan,
}])
sensitivity_table.to_csv(TAB / "table4c_spatial_weights_sensitivity.csv", index=False)
print("\nSaved: outputs/tables/table4c_spatial_weights_sensitivity.csv")

# Stage 4 council correction (Spatial & ML Auditor): "93% agreement" and "94 vs 108
# coldspots" can look contradictory without a full cross-tabulation -- Gi* has 3
# classes (hotspot/coldspot/not_significant), so a district can flip in ways that
# don't show up as a net coldspot-count change. Show the actual 3x3 cross-tab.
crosstab = pd.crosstab(pd.Series(cls_queen, name="Queen"), pd.Series(cls_knn, name="kNN_k6"))
print("\nQueen x k-NN Gi* classification cross-tabulation (full breakdown, not just net counts):")
print(crosstab)
crosstab.to_csv(TAB / "table4c2_gi_star_crosstab.csv")
n_agree = np.trace(crosstab.values)
n_disagree = len(geo_merged) - n_agree
print(f"\nDistricts agreeing: {n_agree} ({n_agree/len(geo_merged)*100:.1f}%), "
      f"disagreeing: {n_disagree} ({n_disagree/len(geo_merged)*100:.1f}%)")
print("Saved: outputs/tables/table4c2_gi_star_crosstab.csv")

# ---------------------------------------------------------------------------
# 5. Save per-district Gi* results (Queen, the confirmed-primary spec after this check)
# ---------------------------------------------------------------------------
geo_merged["gi_star_queen"] = gi_queen.Zs
geo_merged["gi_star_queen_pvalue"] = gi_queen.p_sim
geo_merged["gi_star_queen_class"] = cls_queen
geo_merged["gi_star_knn6"] = gi_knn.Zs
geo_merged["gi_star_knn6_class"] = cls_knn

# Bivariate-style LISA on vulnerability_index_pc1 itself (univariate local Moran's I)
lisa_queen = Moran_Local(y_raw, w_queen, permutations=999)
geo_merged["lisa_quadrant_queen"] = lisa_queen.q  # 1=HH, 2=LH, 3=LL, 4=HL
geo_merged["lisa_pvalue_queen"] = lisa_queen.p_sim

out_cols = ["district", "region", "vulnerability_index_pc1", "gi_star_queen", "gi_star_queen_class",
            "gi_star_knn6", "gi_star_knn6_class", "lisa_quadrant_queen", "lisa_pvalue_queen"]
geo_merged[out_cols].to_csv(TAB / "table4d_district_spatial_statistics.csv", index=False)
print("Saved: outputs/tables/table4d_district_spatial_statistics.csv")

print("\nTop hotspot districts (Queen contiguity, Gi* significant, p<0.05):")
print(geo_merged[geo_merged["gi_star_queen_class"] == "hotspot"][["district", "region", "vulnerability_index_pc1", "gi_star_queen"]]
      .sort_values("gi_star_queen", ascending=False).head(10).to_string(index=False))
