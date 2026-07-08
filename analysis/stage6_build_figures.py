"""
Stage 6 -- Interpretation: figure generation for Project 21. Colourblind-safe
Okabe-Ito-derived palette throughout (per the global anti-slop mandate); bars
start at zero; direct-labelled; titles state the finding, not the data type.
300 DPI PNG (SVG/vector export deferred to Stage 10 production).
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "data"
TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10, "figure.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})

# Okabe-Ito colourblind-safe palette
BLUE, ORANGE, VERMILION, TEAL, GREY = "#0072B2", "#E69F00", "#D55E00", "#009E73", "#7F7F7F"

# ---------------------------------------------------------------------------
# Figure 1: declining mortality vs rising risk factors (the paper's headline tension)
# ---------------------------------------------------------------------------
nat = pd.read_csv(OUT / "master_national_panel.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

axes[0].plot(nat["year"], nat["ncd_premature_death_proportion_pct"], marker="o", color=VERMILION, linewidth=2)
axes[0].set_title("Premature NCD mortality is declining\n(tau=-0.73, p<0.0001)", fontsize=10)
axes[0].set_xlabel("Year"); axes[0].set_ylabel("% of NCD deaths before age 70")
axes[0].set_ylim(58, 70)

for col, color, label in [
    ("overweight_adult_pct_agestd", BLUE, "Overweight"),
    ("obesity_adult_pct_agestd", TEAL, "Obesity"),
    ("insufficient_physical_activity_pct_agestd", ORANGE, "Insufficient activity"),
]:
    axes[1].plot(nat["year"], nat[col], marker="o", color=color, linewidth=2, label=label, markersize=3)
axes[1].set_title("Adiposity and inactivity are rising\nmonotonically (tau=1.00 each, p<0.0001)", fontsize=10)
axes[1].set_xlabel("Year"); axes[1].set_ylabel("% of adult population")
axes[1].legend(fontsize=8, frameon=False, loc="upper left")

for ax, letter in zip(axes, "AB"):
    ax.text(-0.12, 1.12, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

plt.tight_layout()
plt.savefig(FIG / "figure1_mortality_vs_riskfactors.png", dpi=300)
plt.close()
print("Saved figure1_mortality_vs_riskfactors.png")

# ---------------------------------------------------------------------------
# Figure 2: vulnerability index component loadings (PC1), uninsured_rate outlier
# ---------------------------------------------------------------------------
loadings = pd.read_csv(TAB / "table3a_vulnerability_index_component_loadings.csv").rename(columns={"Unnamed: 0": "component"})
loadings = loadings.sort_values("PC1")
colors = [VERMILION if c == "uninsured_rate" else BLUE for c in loadings["component"]]
labels = {
    "poverty_incidence": "Poverty incidence", "poverty_intensity": "Poverty intensity",
    "illiteracy_rate": "Illiteracy rate", "uninsured_rate": "Uninsured rate\n(moved to PC2)",
    "unemployment_rate": "Unemployment rate", "dependency_ratio": "Dependency ratio",
}
fig, ax = plt.subplots(figsize=(7, 4.2))
bars = ax.barh([labels[c] for c in loadings["component"]], loadings["PC1"], color=colors)
ax.set_xlabel("PC1 loading")
ax.set_title("Five of six structural determinants load on PC1;\nuninsured rate does not and is reported as a separate dimension")
ax.axvline(0, color="black", linewidth=0.8)
for b, v in zip(bars, loadings["PC1"]):
    ax.text(v + (0.01 if v >= 0 else -0.01), b.get_y() + b.get_height()/2, f"{v:.3f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "figure2_vulnerability_index_loadings.png", dpi=300)
plt.close()
print("Saved figure2_vulnerability_index_loadings.png")

# ---------------------------------------------------------------------------
# Figure 3: spatial Gi* hotspot/coldspot map (queen contiguity)
# ---------------------------------------------------------------------------
try:
    import geopandas as gpd
    gdf = gpd.read_file(ROOT / "data" / "raw" / "Ghana_New_260_District.geojson")
    crosswalk = pd.read_csv(ROOT / "docs" / "district_crosswalk_261_to_260.csv")
    spatial = pd.read_csv(TAB / "table4d_district_spatial_statistics.csv")

    gdf["DISTRICT_upper"] = gdf["DISTRICT"].str.upper().str.strip()
    crosswalk["geojson_district_upper"] = crosswalk["geojson_district"].str.upper().str.strip()
    merged = gdf.merge(crosswalk[["master_sheet_district", "geojson_district_upper"]],
                        left_on="DISTRICT_upper", right_on="geojson_district_upper", how="left")
    merged = merged.merge(spatial[["district", "gi_star_queen_class"]],
                           left_on="master_sheet_district", right_on="district", how="left")

    class_colors = {"hotspot": VERMILION, "coldspot": BLUE, "not_significant": "#E5E5E5"}
    merged["plot_color"] = merged["gi_star_queen_class"].map(class_colors).fillna("#FFFFFF")

    fig, ax = plt.subplots(figsize=(6.5, 7))
    merged.plot(ax=ax, color=merged["plot_color"], edgecolor="#999999", linewidth=0.3)
    ax.set_title("A large contiguous hotspot spans Ghana's three northern regions\n(Getis-Ord Gi*, queen contiguity, p<0.05)", fontsize=10)
    ax.axis("off")
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=VERMILION, edgecolor="#999999", label=f"Hotspot (n={(merged['gi_star_queen_class']=='hotspot').sum()})"),
        Patch(facecolor=BLUE, edgecolor="#999999", label=f"Coldspot (n={(merged['gi_star_queen_class']=='coldspot').sum()})"),
        Patch(facecolor="#E5E5E5", edgecolor="#999999", label="Not significant"),
        Patch(facecolor="#FFFFFF", edgecolor="#999999", label="No polygon match / not testable"),
    ]
    ax.legend(handles=legend_elems, loc="lower left", fontsize=8, frameon=False)

    # North arrow (top-right, clear of the map body)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax_x = xlim[0] + 0.94 * (xlim[1] - xlim[0])
    ax_y0 = ylim[0] + 0.90 * (ylim[1] - ylim[0])
    ax_y1 = ylim[0] + 0.97 * (ylim[1] - ylim[0])
    ax.annotate("N", xy=(ax_x, ax_y1), xytext=(ax_x, ax_y0),
                arrowprops=dict(facecolor="black", edgecolor="black", width=3, headwidth=10, headlength=8),
                ha="center", va="center", fontsize=10, fontweight="bold")

    # Scale bar: 1 decimal degree of longitude at ~8N approx 110 km (cos(8deg) x 111km)
    # Placed bottom-right (legend occupies bottom-left, north arrow occupies top-right)
    bar_deg = 0.5
    bar_km = bar_deg * 111.0 * np.cos(np.radians(8.0))
    bx1 = xlim[0] + 0.96 * (xlim[1] - xlim[0])
    bx0 = bx1 - bar_deg
    by = ylim[0] + 0.04 * (ylim[1] - ylim[0])
    ax.plot([bx0, bx1], [by, by], color="black", linewidth=2, solid_capstyle="butt")
    ax.text(bx0 + bar_deg / 2, by + 0.015 * (ylim[1] - ylim[0]), f"~{bar_km:.0f} km",
            ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(FIG / "figure3_spatial_hotspot_map.png", dpi=300)
    plt.close()
    print("Saved figure3_spatial_hotspot_map.png "
          f"({merged['gi_star_queen_class'].notna().sum()} districts matched and classified)")
except Exception as e:
    print(f"Figure 3 (spatial map) FAILED: {e}")

# ---------------------------------------------------------------------------
# Figure 4: SHAP feature importance (mandatory per Tenet 13 for tree-based models)
# ---------------------------------------------------------------------------
shap_tab = pd.read_csv(TAB / "table4f_shap_feature_ranking.csv").sort_values("mean_abs_shap")
feature_labels = {
    "insufficient_physical_activity_pct_agestd": "Insufficient physical activity",
    "overweight_adult_pct_agestd": "Overweight prevalence",
    "obesity_adult_pct_agestd": "Obesity prevalence",
    "oop_pct_che": "Out-of-pocket expenditure (% CHE)",
    "alcohol_percapita_litres": "Alcohol per-capita consumption",
    "non_hdl_cholesterol_mean_agestd": "Non-HDL cholesterol",
}
fig, ax = plt.subplots(figsize=(8.5, 4.4))
colors = [TEAL if v > 0.03 else GREY for v in shap_tab["mean_abs_shap"]]
ax.barh([feature_labels.get(f, f) for f in shap_tab["feature"]], shap_tab["mean_abs_shap"], color=colors)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Physical inactivity and adiposity dominate feature importance\n(~6-fold margin over financing/behavioural covariates)", fontsize=10)
plt.tight_layout()
plt.savefig(FIG / "figure4_shap_importance.png", dpi=300)
plt.close()
print("Saved figure4_shap_importance.png")

# ---------------------------------------------------------------------------
# Figure 5: LOOCV bootstrap stability check -- honest, non-reassuring finding
# ---------------------------------------------------------------------------
# Reads the raw per-resample values cached by stage5_loocv_bootstrap_stability.py
# (that script was re-run once with raw-value persistence added; no need to
# repeat the ~25-minute computation here).
boot_vals = pd.read_csv(TAB / "table6a_loocv_r2_bootstrap_raw_values.csv")["bootstrap_r2"].values
print(f"Figure 5: loaded {len(boot_vals)} cached bootstrap R2 values")

fig, ax = plt.subplots(figsize=(7, 4.6))
counts, _bins, _patches = ax.hist(boot_vals, bins=25, color=GREY, edgecolor="white")
ax.axvline(0.2311, color=VERMILION, linewidth=2, label="Original LOOCV R² = 0.231 (actual 18 districts)")
ax.axvline(np.median(boot_vals), color=BLUE, linewidth=2, linestyle="--",
           label=f"Bootstrap median = {np.median(boot_vals):.2f} (inflated by duplicate-resample leakage)")
ax.set_xlabel("LOOCV R²"); ax.set_ylabel("Bootstrap resamples")
ax.set_title("The bootstrap distribution sits well above the original estimate,\nconfirming rather than resolving instability at n=18", fontsize=10)
# Headroom above the tallest bar keeps the legend clear of the histogram at any x-position
ax.set_ylim(0, counts.max() * 1.32)
ax.legend(fontsize=8, frameon=True, framealpha=0.95, edgecolor="none", loc="upper right")
plt.tight_layout()
plt.savefig(FIG / "figure5_loocv_bootstrap_distribution.png", dpi=300)
plt.close()
print("Saved figure5_loocv_bootstrap_distribution.png")

print("\nAll figures saved to outputs/figures/")
