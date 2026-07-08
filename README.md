# Noncommunicable Disease Burden and Determinants in Ghana: National Epidemiological Trends and District-Level Structural Risk-Vulnerability Mapping

[![CI](https://github.com/valentineghanem-bit/spatial-ncd-burden-ghana/actions/workflows/ci.yml/badge.svg)](https://github.com/valentineghanem-bit/spatial-ncd-burden-ghana/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0002--8332--0220-green.svg)](https://orcid.org/0009-0002-8332-0220)

**Author:** Valentine Golden Ghanem | Ghana COCOBOD Cocoa Clinic, Accra, Ghana
**ORCID:** [0009-0002-8332-0220](https://orcid.org/0009-0002-8332-0220)
**Affiliation:** Ghana COCOBOD Cocoa Clinic, Accra, Ghana · University of Cape Coast, Ghana
**Reporting standard:** STROBE · RECORD-Spatial · TRIPOD-AI
**Date:** 2026
**Status:** Manuscript in preparation

## 1. Abstract

Noncommunicable diseases (NCDs) account for the majority of premature mortality risk in sub-Saharan Africa, yet district-level NCD surveillance data remain unavailable across most of the region, including Ghana. This study bounds its analysis to what national and district-level evidence actually support, rather than extrapolating estimates the data cannot provide. Two independent data layers were analysed, never statistically joined. A national panel (WHO Global Health Observatory and GISAH, 1980–2022) tracked NCD trends via Mann-Kendall tests and modelled year-over-year change in the 30–70y multi-NCD mortality probability using Random Forest and XGBoost. A district structural vulnerability index (261 districts, 2021 census) was built via principal component analysis on six determinants, then tested for spatial clustering via Getis-Ord Gi* and Moran's I. Premature NCD mortality declined over 2000–2021 (tau=−0.73, p<0.0001) while obesity, diabetes prevalence, and physical inactivity rose monotonically (tau=1.00, p<0.0001 each). Getis-Ord Gi* identified a contiguous 54-district hotspot cluster spanning Ghana's three northern regions. The predictive model's apparent signal (R²=0.231) did not survive an autocorrelation-preserving null (p≈0.18) and is reported as inconclusive.

## 2. Research Question & Aims

1. What are Ghana's national NCD-burden and risk-factor trends over 2000–2022, and does a machine-learning model carry genuine predictive signal for year-over-year change in premature NCD mortality risk at the sample size the national data actually permits?
2. Where in Ghana is district-level structural NCD-risk vulnerability spatially concentrated, and is the insurance-access dimension of that vulnerability statistically separable from the poverty/illiteracy/dependency cluster?

These two questions are answered as two independent, never-statistically-joined analytical tracks, reflecting the genuine absence of any district-level NCD outcome data for Ghana.

## 3. Methods Summary

| Method | Tool | Purpose |
|---|---|---|
| Mann-Kendall trend test + Sen's slope | `pymannkendall` | National NCD-burden/risk-factor trend direction and magnitude, 2000–2022 |
| Hamed-Rao (1998) autocorrelation-corrected variance | custom (`scipy`) | Robustness check against Mann-Kendall's serial-independence assumption |
| Benjamini-Hochberg FDR correction | `scipy.stats` | Multiple-comparison correction across 11 national trend tests |
| Random Forest / XGBoost, leave-one-out CV | `scikit-learn`, `xgboost` | National year-over-year multi-NCD mortality change prediction (n=18) |
| Permutation null (full-randomisation + autocorrelation-preserving circular-shift) | custom | Significance testing against an appropriate null given target autocorrelation |
| SHAP | `shap` | Feature-importance interpretation, full-data + LOOCV-subset stability |
| Principal component analysis, KMO/Bartlett, bootstrap loading stability | `scikit-learn`, `factor_analyzer` | District structural vulnerability index construction and validation |
| Getis-Ord Gi*, global Moran's I, LISA | `esda`, `libpysal` | Spatial clustering of district structural vulnerability, queen contiguity + k=6 k-NN sensitivity |

## 4. Data Sources

| Source | Variables | Year | Access |
|---|---|---|---|
| WHO Global Health Observatory | NCD mortality, risk-factor prevalence, health expenditure | 1980–2022 | [who.int/data/gho](https://www.who.int/data/gho) |
| Global Information System on Alcohol and Health (GISAH) | Alcohol per-capita consumption | 2000–2019 | WHO GISAH |
| World Health Statistics compendium | Cross-check source (superset), never merged as independent observation | 1980–2022 | WHO |
| WHO Joint Malnutrition Estimates | Under-5 wasting, stunting | 1988–2022 | WHO/UNICEF/World Bank |
| Ghana Statistical Service 2021 Population and Housing Census | District-level structural determinants (poverty, literacy, insurance, employment, dependency) | 2021 | Ghana Statistical Service |
| District boundary GeoJSON | 261 Metropolitan/Municipal/District Assemblies (258 boundary-matched) | 2021 | Derived from Ghana Statistical Service census geography |

## 5. Key Findings

| Metric | Value |
|---|---|
| Premature NCD mortality trend, 2000–2021 | tau=−0.73, p<0.0001 (declining) |
| Adult overweight/obesity/inactivity trend, 2000–2022 | tau=1.00, p<0.0001 each (rising monotonically) |
| District vulnerability index reliability | Cronbach's alpha=0.801; KMO=0.662 |
| Getis-Ord Gi* hotspot cluster | 54 districts, three northern regions, p<0.05 |
| Getis-Ord Gi* coldspot cluster | 46 districts, Greater Accra |
| Global Moran's I | 0.797–0.803 across weights specifications, p=0.001 |
| National ML model, LOOCV R² | 0.231; not significant vs. autocorrelation-preserving null (p≈0.18) — reported inconclusive |
| Districts analysed (structural vulnerability layer) | 261 (258 spatially matched) |

## 6. Repository Structure

```
spatial-ncd-burden-ghana/
├── README.md  CITATION.cff  LICENSE  requirements.txt  .gitignore  .gitattributes
├── .github/workflows/ci.yml
├── data/{raw,processed}/
├── analysis/                  # Python analysis pipeline (Stage 0-6 scripts)
├── outputs/
│   ├── data/                  # Working master CSVs
│   ├── release/               # Master CSV deliverable (3 CSVs + variable_provenance.csv)
│   ├── figures/                # 5 manuscript figures, 300 DPI
│   └── tables/                 # Canonical analysis output tables
├── docs/                       # Stage verdict and audit documents
├── dashboard/NCD_Spatial_Burden_Ghana_Dashboard.html
├── poster/NCD_Spatial_Burden_Ghana_Poster.html
└── qa/
```

## 7. Reproducibility

### 7.1 Requirements
Python 3.12, packages listed in `requirements.txt`.

### 7.2 Clone & install
```bash
git clone https://github.com/valentineghanem-bit/spatial-ncd-burden-ghana.git
cd spatial-ncd-burden-ghana
pip install -r requirements.txt
```

### 7.3 Run the analytical pipeline
```bash
python analysis/build_master.py
python analysis/build_vulnerability_index.py
python analysis/build_variable_provenance.py
python analysis/stage4_spatial_weights.py
python analysis/stage4_ml_pipeline.py
python analysis/stage5_mann_kendall_trends.py
python analysis/stage6_build_figures.py
```

### 7.4 Run the test suite
No automated pytest suite is bundled with this repository; CI performs a syntax and dependency-consistency check on every script (`.github/workflows/ci.yml`). Analytical correctness is verified via the manuscript's own QA gates (`docs/`), not a unit-test suite.

### 7.5 Launch the interactive Dash application
Not applicable — this repository ships a self-contained, offline HTML dashboard (§7.6) rather than a live Dash server.

### 7.6 Open the static HTML dashboard
Open `dashboard/NCD_Spatial_Burden_Ghana_Dashboard.html` directly in a browser — no server required.

## 8. Outputs

| Output | Description |
|---|---|
| `outputs/release/` | Master CSV deliverable: 3 analytic datasets + `variable_provenance.csv` data dictionary (28 variables) |
| `outputs/figures/` | 5 manuscript figures (300 DPI): national trends, PCA loadings, spatial hotspot map, SHAP importance, LOOCV bootstrap distribution |
| `outputs/tables/` | Canonical analysis output tables underlying every manuscript table |
| `dashboard/` | Interactive HI-EI dashboard (district choropleth, LISA cluster map, ranking, scatter, heatmap, treemap) |
| `poster/` | A0 conference poster |

## 8a. Downloadable Artefacts (HTML)

| Artefact | View on GitHub | Live preview | Direct download |
|---|---|---|---|
| Interactive dashboard | [NCD_Spatial_Burden_Ghana_Dashboard.html](https://github.com/valentineghanem-bit/spatial-ncd-burden-ghana/blob/main/dashboard/NCD_Spatial_Burden_Ghana_Dashboard.html) | [Preview](https://htmlpreview.github.io/?https://github.com/valentineghanem-bit/spatial-ncd-burden-ghana/blob/main/dashboard/NCD_Spatial_Burden_Ghana_Dashboard.html) | [Download](https://raw.githubusercontent.com/valentineghanem-bit/spatial-ncd-burden-ghana/main/dashboard/NCD_Spatial_Burden_Ghana_Dashboard.html) |
| Conference poster | [NCD_Spatial_Burden_Ghana_Poster.html](https://github.com/valentineghanem-bit/spatial-ncd-burden-ghana/blob/main/poster/NCD_Spatial_Burden_Ghana_Poster.html) | [Preview](https://htmlpreview.github.io/?https://github.com/valentineghanem-bit/spatial-ncd-burden-ghana/blob/main/poster/NCD_Spatial_Burden_Ghana_Poster.html) | [Download](https://raw.githubusercontent.com/valentineghanem-bit/spatial-ncd-burden-ghana/main/poster/NCD_Spatial_Burden_Ghana_Poster.html) |

## 9. Reporting Standard

This study follows the STROBE Statement (cross-sectional variant, for the district-level structural vulnerability component), the RECORD-Spatial extension (for the routinely-collected spatial health data component), and TRIPOD-AI (for the national machine-learning prediction component). Full item-by-item checklists are included as Supplementary Materials in the manuscript (not committed to this repository per Tenet 20 — see §10).

## 10. Ethical Statement

This study is a secondary analysis of publicly available, de-identified, aggregate data (WHO Global Health Observatory / GISAH / World Health Statistics national indicators; Ghana Statistical Service 2021 Population and Housing Census district-level data). No individual-level data were accessed at any stage. As secondary analysis of public, de-identified, aggregate data, this study is anticipated to qualify for Ghana Health Service Ethics Review Board exemption; a formal exemption statement will accompany manuscript submission.

## 11. Citation

If you use this software or dataset, please cite:

> Ghanem, V.G. (2026). *Noncommunicable Disease Burden and Determinants in Ghana: National Epidemiological Trends and District-Level Structural Risk-Vulnerability Mapping* [Software]. GitHub. https://github.com/valentineghanem-bit/spatial-ncd-burden-ghana

```bibtex
@software{ghanem2026spatialncd,
  author = {Ghanem, Valentine Golden},
  title = {Noncommunicable Disease Burden and Determinants in Ghana: National Epidemiological Trends and District-Level Structural Risk-Vulnerability Mapping},
  year = {2026},
  url = {https://github.com/valentineghanem-bit/spatial-ncd-burden-ghana}
}
```

See also `CITATION.cff` for machine-readable citation metadata.

## 12. License

Code: [MIT License](LICENSE). Outputs and figures: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## 13. Author & Contact

**Valentine Golden Ghanem**
Ghana COCOBOD Cocoa Clinic, Accra, Ghana · University of Cape Coast, Ghana
ORCID: [0009-0002-8332-0220](https://orcid.org/0009-0002-8332-0220)
Email: valentineghanem@gmail.com

## 14. Acknowledgements

Not applicable.
