"""
Stage 4 -- carried-forward diagnostics from Stage 3 council (Spatial & ML Auditor's
framed question, which specified these two checks but never returned a response due
to a session limit). Both are executed here, not silently dropped.

1. KMO (Kaiser-Meyer-Olkin) sampling adequacy + Bartlett's test of sphericity --
   standard pre-PCA diagnostics confirming the correlation matrix is suitable for
   factor extraction at all (per Allik 2019 / Sharma 2023, this project's own cited
   methodology templates).
2. Bootstrap stability of the PC1 loadings themselves (5,000 resamples of the 261
   districts, re-run PCA each time, report the range/SD of each component's loading).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

OUT = Path(__file__).resolve().parent.parent / "outputs" / "data"
TAB = Path(__file__).resolve().parent.parent / "outputs" / "tables"

dist = pd.read_csv(OUT / "master_district_vulnerability.csv")
comp_cols = ["poverty_incidence", "poverty_intensity", "illiteracy_rate",
             "uninsured_rate", "unemployment_rate", "dependency_ratio"]
Z = (dist[comp_cols] - dist[comp_cols].mean()) / dist[comp_cols].std(ddof=0)

# ---------------------------------------------------------------------------
# 1. KMO + Bartlett
# ---------------------------------------------------------------------------
kmo_per_item, kmo_overall = calculate_kmo(Z)
chi_square, p_value = calculate_bartlett_sphericity(Z)

print("=" * 70)
print("KMO SAMPLING ADEQUACY + BARTLETT'S TEST OF SPHERICITY")
print("=" * 70)
print(f"Overall KMO: {kmo_overall:.3f}")
print("Per-item KMO:")
for col, k in zip(comp_cols, kmo_per_item):
    print(f"  {col}: {k:.3f}")
print(f"\nBartlett's test: chi-square={chi_square:.1f}, p={p_value:.2e}")

kmo_interpretation = (
    "marvelous" if kmo_overall >= 0.9 else
    "meritorious" if kmo_overall >= 0.8 else
    "middling" if kmo_overall >= 0.7 else
    "mediocre" if kmo_overall >= 0.6 else
    "miserable" if kmo_overall >= 0.5 else
    "unacceptable"
)
print(f"\nKaiser's interpretation of overall KMO={kmo_overall:.3f}: '{kmo_interpretation}'"
      f" (conventional cutoffs: >=0.6 acceptable, >=0.8 good)")
bartlett_verdict = "REJECTS the null of an identity correlation matrix (p<0.001) -- factor extraction is justified" if p_value < 0.001 else "does NOT clearly reject the null -- factor extraction may not be justified"
print(f"Bartlett verdict: {bartlett_verdict}")

diag = pd.DataFrame({"component": comp_cols, "kmo_per_item": kmo_per_item})
diag_summary = pd.DataFrame([{
    "kmo_overall": round(kmo_overall, 3),
    "kmo_interpretation": kmo_interpretation,
    "bartlett_chi_square": round(chi_square, 1),
    "bartlett_p_value": p_value,
    "bartlett_verdict": bartlett_verdict,
}])
diag.to_csv(TAB / "table4a_kmo_per_item.csv", index=False)
diag_summary.to_csv(TAB / "table4a_kmo_bartlett_summary.csv", index=False)
print("\nSaved: outputs/tables/table4a_kmo_per_item.csv, table4a_kmo_bartlett_summary.csv")

# ---------------------------------------------------------------------------
# 2. Bootstrap stability of PC1 loadings (5,000 resamples)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BOOTSTRAP STABILITY OF PC1 LOADINGS (5,000 resamples)")
print("=" * 70)

rng = np.random.default_rng(42)
n = len(dist)
boot_loadings = {c: [] for c in comp_cols}

# Reference orientation: original PC1 loading sign per component (poverty_incidence positive)
pca_ref = PCA(n_components=len(comp_cols))
pca_ref.fit(Z.values)
ref_loadings = pca_ref.components_[0]
if ref_loadings[comp_cols.index("poverty_incidence")] < 0:
    ref_loadings = -ref_loadings

for _ in range(5000):
    samp_idx = rng.choice(n, size=n, replace=True)
    Z_samp = Z.values[samp_idx]
    pca_b = PCA(n_components=len(comp_cols))
    pca_b.fit(Z_samp)
    load_b = pca_b.components_[0]
    # Orient sign to match reference (PCA sign is arbitrary per resample)
    if np.dot(load_b, ref_loadings) < 0:
        load_b = -load_b
    for i, c in enumerate(comp_cols):
        boot_loadings[c].append(load_b[i])

stability = []
for c in comp_cols:
    arr = np.array(boot_loadings[c])
    ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
    stability.append({
        "component": c,
        "original_loading": round(ref_loadings[comp_cols.index(c)], 3),
        "bootstrap_mean": round(arr.mean(), 3),
        "bootstrap_sd": round(arr.std(), 3),
        "ci_95_low": round(ci_low, 3),
        "ci_95_high": round(ci_high, 3),
    })
    print(f"{c}: original={ref_loadings[comp_cols.index(c)]:.3f}, "
          f"bootstrap mean={arr.mean():.3f} (SD={arr.std():.3f}), "
          f"95% CI=({ci_low:.3f}, {ci_high:.3f})")

stability_df = pd.DataFrame(stability)
stability_df.to_csv(TAB / "table4b_pc1_loading_bootstrap_stability.csv", index=False)
print("\nSaved: outputs/tables/table4b_pc1_loading_bootstrap_stability.csv")

# Flag any component whose CI crosses zero (unstable sign/contribution) or whose
# original loading falls outside its own bootstrap 95% CI (surprising instability)
print("\nStability flags:")
for row in stability:
    crosses_zero = row["ci_95_low"] <= 0 <= row["ci_95_high"]
    flag = "UNSTABLE (CI crosses zero)" if crosses_zero else "stable"
    print(f"  {row['component']}: {flag}")
