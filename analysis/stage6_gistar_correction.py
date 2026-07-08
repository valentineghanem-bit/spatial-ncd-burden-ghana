"""
Stage 6 -- CRITICAL CORRECTION to Stage 4's Getis-Ord Gi* analysis, discovered
while building the Stage 6 spatial hotspot map figure.

Root cause: esda's G_Local computes Gi*/Gi as a RATIO statistic --
    G = (W @ y) / (y.sum() - y * (not star))
(see esda/getisord.py line 506). This ratio is only well-behaved when y is
non-negative with a comfortably positive sum, which is the classical Getis-Ord
requirement (see e.g. ESRI's Hot Spot Analysis documentation: "For the
Getis-Ord Gi* statistic to work properly, at least some of your values must be
greater than zero... If you have both positive and negative values, add a
constant to make all values positive").

vulnerability_index_pc1 is a mean-centered PCA component (Stage 3), so its
global sum across the 258 spatially-matched districts is 0.585 -- essentially
zero. For any district with y_i > 0.585, the denominator (y.sum() - y_i) goes
NEGATIVE, flipping the sign of the ratio independent of the actual local
clustering pattern. This produced a near-random-looking, weakly NEGATIVE
correlation (-0.35) between the raw index value and its own Gi* z-score, when
strong positive spatial autocorrelation (Moran's I=0.80) implies this
correlation should be strongly POSITIVE. Verified directly: the previously
"coldspot"-labelled class had the HIGHEST mean vulnerability score (0.597,
n=95) and contained the five most individually-vulnerable rural districts
(Nabdam, Yunyoo Nasuan, Wa West, Central Gonja, East Mamprusi), while the
"hotspot"-labelled class (the four urban centres) had a near-zero mean (0.032)
-- backwards from the intended interpretation.

Global Moran's I and Moran_Local (LISA) are UNAFFECTED: their formula uses
sums of products of mean-deviations (a covariance-like structure), which is
well-defined for any signed variable and does not require this shift. Only
the Gi*/Gi statistic needs correcting.

Fix: shift vulnerability_index_pc1 to be strictly positive (y_shifted = y -
min(y) + 1) before Gi* computation. This preserves the index's relative
spatial structure and rank ordering exactly (a positive affine shift changes
neither the correlation structure nor the relative distances between
districts) while giving G_Local a numerically well-behaved, comfortably
positive-summing input.
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from libpysal.weights import Queen, KNN
from esda.getisord import G_Local

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "data"
TAB = Path(__file__).resolve().parent.parent / "outputs" / "tables"
DOCS = Path(__file__).resolve().parent.parent / "docs"

print("Loading GeoJSON (~27MB)...")
gdf = gpd.read_file(RAW / "Ghana_New_260_District.geojson")
gdf["DISTRICT"] = gdf["DISTRICT"].str.strip().str.upper()

crosswalk = pd.read_csv(DOCS / "district_crosswalk_261_to_260.csv")
crosswalk["geojson_district"] = crosswalk["geojson_district"].str.strip().str.upper()

dist = pd.read_csv(OUT / "master_district_vulnerability.csv")
merged_keys = dist.merge(crosswalk, left_on="district", right_on="master_sheet_district", how="left")
geo_merged = gdf.merge(merged_keys, left_on="DISTRICT", right_on="geojson_district", how="inner")
geo_merged = geo_merged.reset_index(drop=True)
print(f"Districts joined: {len(geo_merged)}")

w_queen = Queen.from_dataframe(geo_merged, use_index=False)
w_queen.transform = "r"
w_knn6 = KNN.from_dataframe(geo_merged, k=6, use_index=False)
w_knn6.transform = "r"

y_raw = geo_merged["vulnerability_index_pc1"].values
print(f"Raw y: sum={y_raw.sum():.4f}, min={y_raw.min():.4f}, max={y_raw.max():.4f} -- "
      f"sum near zero, denominator sign-flip confirmed as root cause")

# THE FIX: shift to strictly positive before Gi*/Gi (Moran's I/LISA unaffected, not re-run here)
SHIFT = -y_raw.min() + 1.0
y_shifted = y_raw + SHIFT
print(f"Shifted y (y - min + 1): sum={y_shifted.sum():.4f}, min={y_shifted.min():.4f}, max={y_shifted.max():.4f}")
print(f"Correlation(y_raw, y_shifted) = {np.corrcoef(y_raw, y_shifted)[0,1]:.6f} (must be 1.0 -- confirms pure affine shift, no structure change)")

gi_queen = G_Local(y_shifted, w_queen, transform="r", permutations=999, seed=42)
gi_knn = G_Local(y_shifted, w_knn6, transform="r", permutations=999, seed=42)

print(f"\nCorrected correlation(y_raw, gi_star_queen Z): {np.corrcoef(y_raw, gi_queen.Zs)[0,1]:.4f} (should now be strongly positive)")


def classify_gi(gi_result, alpha=0.05):
    z, p = gi_result.Zs, gi_result.p_sim
    return np.where((p < alpha) & (z > 0), "hotspot",
           np.where((p < alpha) & (z < 0), "coldspot", "not_significant"))


cls_queen = classify_gi(gi_queen)
cls_knn = classify_gi(gi_knn)
agreement = (cls_queen == cls_knn).mean()
n_hot_q, n_hot_k = (cls_queen == "hotspot").sum(), (cls_knn == "hotspot").sum()
n_cold_q, n_cold_k = (cls_queen == "coldspot").sum(), (cls_knn == "coldspot").sum()

print(f"\nCORRECTED Gi* results (Queen contiguity):")
print(f"  Hotspots: {n_hot_q}, Coldspots: {n_cold_q}, Not significant: {(cls_queen=='not_significant').sum()}")
print(f"CORRECTED Gi* results (k=6 k-NN):")
print(f"  Hotspots: {n_hot_k}, Coldspots: {n_cold_k}, Not significant: {(cls_knn=='not_significant').sum()}")
print(f"Classification agreement Queen vs k-NN: {agreement*100:.1f}%")

crosstab = pd.crosstab(pd.Series(cls_queen, name="Queen"), pd.Series(cls_knn, name="kNN_k6"))
print("\nCorrected cross-tabulation:")
print(crosstab)
crosstab.to_csv(TAB / "table4c2_gi_star_crosstab_CORRECTED.csv")

sensitivity_table = pd.DataFrame([{
    "metric": "Gi* classification agreement (%)", "queen_contiguity": round(agreement*100, 1),
    "knn_k6": np.nan, "numeric_spread": round(100-agreement*100, 1), "queen_p": np.nan, "knn_p": np.nan,
}, {
    "metric": "N hotspot districts", "queen_contiguity": int(n_hot_q), "knn_k6": int(n_hot_k),
    "numeric_spread": abs(int(n_hot_q)-int(n_hot_k)), "queen_p": np.nan, "knn_p": np.nan,
}, {
    "metric": "N coldspot districts", "queen_contiguity": int(n_cold_q), "knn_k6": int(n_cold_k),
    "numeric_spread": abs(int(n_cold_q)-int(n_cold_k)), "queen_p": np.nan, "knn_p": np.nan,
}])
sensitivity_table.to_csv(TAB / "table4c_spatial_weights_sensitivity_CORRECTED.csv", index=False)

geo_merged["gi_star_queen_CORRECTED"] = gi_queen.Zs
geo_merged["gi_star_queen_pvalue_CORRECTED"] = gi_queen.p_sim
geo_merged["gi_star_queen_class_CORRECTED"] = cls_queen
geo_merged["gi_star_knn6_CORRECTED"] = gi_knn.Zs
geo_merged["gi_star_knn6_class_CORRECTED"] = cls_knn

out_cols = ["district", "region", "vulnerability_index_pc1", "gi_star_queen_CORRECTED",
            "gi_star_queen_class_CORRECTED", "gi_star_knn6_CORRECTED", "gi_star_knn6_class_CORRECTED"]
geo_merged[out_cols].to_csv(TAB / "table4d_district_spatial_statistics_CORRECTED.csv", index=False)
print("\nSaved: table4c_spatial_weights_sensitivity_CORRECTED.csv, table4c2_gi_star_crosstab_CORRECTED.csv, "
      "table4d_district_spatial_statistics_CORRECTED.csv")

print("\nTop hotspot districts (corrected, Queen contiguity):")
hot = geo_merged[geo_merged["gi_star_queen_class_CORRECTED"] == "hotspot"][
    ["district", "region", "vulnerability_index_pc1", "gi_star_queen_CORRECTED"]
].sort_values("gi_star_queen_CORRECTED", ascending=False)
print(hot.to_string(index=False))

print("\nTop coldspot districts (corrected, Queen contiguity):")
cold = geo_merged[geo_merged["gi_star_queen_class_CORRECTED"] == "coldspot"][
    ["district", "region", "vulnerability_index_pc1", "gi_star_queen_CORRECTED"]
].sort_values("gi_star_queen_CORRECTED").head(10)
print(cold.to_string(index=False))
