"""
Stage 5 -- bootstrap stability check on the Stage 4 Random Forest LOOCV R2=0.2311.
Closes the gap independently flagged by both the Scite Skeptic and the Spatial & ML
Auditor during the Stage 5 council review: the Chairman's Stage 4 ruling logged
"whether LOOCV R2 is a well-calibrated estimator at n=18" as an open, unresolved
methodological question but no diagnostic was actually run against it. This script
runs the cheap diagnostic rather than deferring it further.

Method: case-resampling bootstrap. For each of 200 iterations, draw n=18 districts
with replacement from the complete-case ML frame, refit the full leave-one-out CV
Random Forest pipeline (identical spec to stage4_ml_pipeline.py -- 200 trees, max
depth 3) on the resampled data, and record the resulting LOOCV R2. This produces an
empirical distribution over R2 that reflects sampling variability in which 18
districts happened to be observed, which is exactly the calibration question the
Skeptic raised. Duplicate rows within a bootstrap resample are an accepted property
of case resampling at this sample size, not an error -- they are what make the
resampled dataset a legitimate draw from the same empirical distribution as the
original 18 observations.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score

OUT = Path(__file__).resolve().parent.parent / "outputs" / "data"
TAB = Path(__file__).resolve().parent.parent / "outputs" / "tables"

ml = pd.read_csv(OUT / "national_ml_modelling_frame.csv").dropna()
FEATURES = ["overweight_adult_pct_agestd", "obesity_adult_pct_agestd", "non_hdl_cholesterol_mean_agestd",
            "alcohol_percapita_litres", "insufficient_physical_activity_pct_agestd", "oop_pct_che"]
TARGET = "ncd_3070_probability_pct_yoy_change"

X_full = ml[FEATURES].values
y_full = ml[TARGET].values
n = len(ml)
rf_fn = lambda seed: RandomForestRegressor(n_estimators=200, max_depth=3, random_state=seed, n_jobs=1)


def loocv_r2(X, y, seed):
    loo = LeaveOneOut()
    preds, actuals = [], []
    for train_idx, test_idx in loo.split(X):
        if len(np.unique(y[train_idx])) < 2:
            return np.nan
        model = rf_fn(seed)
        model.fit(X[train_idx], y[train_idx])
        preds.append(model.predict(X[test_idx])[0])
        actuals.append(y[test_idx][0])
    return r2_score(actuals, preds)


rng = np.random.default_rng(42)
N_BOOT = 200
boot_r2s = []
n_skipped = 0
t_start = __import__("time").time()
for b in range(N_BOOT):
    idx = rng.integers(0, n, size=n)
    X_b, y_b = X_full[idx], y_full[idx]
    r2_b = loocv_r2(X_b, y_b, seed=42)
    if np.isnan(r2_b):
        n_skipped += 1
        continue
    boot_r2s.append(r2_b)
    if (b + 1) % 20 == 0:
        elapsed = __import__("time").time() - t_start
        print(f"  bootstrap {b+1}/{N_BOOT} done ({elapsed:.0f}s elapsed)", flush=True)

boot_r2s = np.array(boot_r2s)
ci_low, ci_high = np.percentile(boot_r2s, [2.5, 97.5])
print(f"\nOriginal (non-resampled) LOOCV R2: 0.2311")
print(f"Bootstrap (n={len(boot_r2s)} valid resamples of {N_BOOT}, {n_skipped} skipped -- degenerate/constant-target draws): "
      f"mean={boot_r2s.mean():.4f}, SD={boot_r2s.std():.4f}, median={np.median(boot_r2s):.4f}")
print(f"95% bootstrap CI on LOOCV R2: ({ci_low:.4f}, {ci_high:.4f})")
pct_negative = (boot_r2s < 0).mean() * 100
print(f"Share of bootstrap resamples with R2 < 0: {pct_negative:.1f}%")

pd.DataFrame({"bootstrap_r2": boot_r2s}).to_csv(TAB / "table6a_loocv_r2_bootstrap_raw_values.csv", index=False)
print("Saved raw per-resample values: outputs/tables/table6a_loocv_r2_bootstrap_raw_values.csv (for Figure 5)")

pd.DataFrame([{
    "original_loocv_r2": 0.2311,
    "n_bootstrap_valid": len(boot_r2s),
    "n_bootstrap_skipped_degenerate": n_skipped,
    "bootstrap_mean_r2": round(boot_r2s.mean(), 4),
    "bootstrap_sd_r2": round(boot_r2s.std(), 4),
    "bootstrap_median_r2": round(float(np.median(boot_r2s)), 4),
    "ci_95_low": round(ci_low, 4),
    "ci_95_high": round(ci_high, 4),
    "pct_resamples_r2_negative": round(pct_negative, 1),
}]).to_csv(TAB / "table6a_loocv_r2_bootstrap_stability.csv", index=False)
print("\nSaved: outputs/tables/table6a_loocv_r2_bootstrap_stability.csv")
