"""
Build the 259-polygon regions JSON consumed by the HI-EI dashboard/poster generator
(_system/bespoke/bespoke_gen.js). Uses the vetted 261-to-GeoJSON crosswalk
(docs/district_crosswalk_261_to_260.csv) rather than naive name matching, which only
resolved 174/261 districts and left the majority of the choropleth blank.

Two corrections applied on top of the crosswalk, both verified directly against the
GeoJSON's actual feature list (259 features):
  - Sagnarigu Municipal: crosswalk marks this a structural_gap absorbed into Tamale
    Metropolitan, but the GeoJSON has its own distinct "SAGNERIGU" polygon (a spelling
    variant). Mapped directly rather than merged.
  - Ayawaso Central Municipal: crosswalk claims an exact match that does not exist in
    this GeoJSON (only West/North/East do). Merged into the adjacent AYAWASO EAST
    MUNICIPAL polygon, population-weighted, for map display only -- the master CSV and
    manuscript keep it as its own distinct row.

Output: writes to the path passed as argv[1], or prints to stdout if omitted.
"""
import sys
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GAP_PARENT_POLYGON = {
    "Awutu Senya West": "AWUTU SENYA",
    "Guan": "KRACHI EAST MUNICIPAL",
    "Sagnarigu Municipal": "SAGNERIGU",
    "Ayawaso Central Municipal": "AYAWASO EAST MUNICIPAL",
}


def lisa_class(row):
    if pd.isna(row["lisa_pvalue_queen"]) or row["lisa_pvalue_queen"] >= 0.05:
        return "NS"
    q = int(row["lisa_quadrant_queen"])
    return {1: "HH", 2: "LH", 3: "LL", 4: "HL"}.get(q, "NS")


def build(geojson_path: str):
    master = pd.read_csv(ROOT / "outputs/data/master_district_vulnerability.csv")
    spatial = pd.read_csv(ROOT / "outputs/tables/table4d_district_spatial_statistics.csv")
    crosswalk = pd.read_csv(ROOT / "docs/district_crosswalk_261_to_260.csv")

    spatial["lisa"] = spatial.apply(lisa_class, axis=1)
    lisa_by_district = dict(zip(spatial.district, spatial.lisa))
    cw_map = dict(zip(crosswalk.master_sheet_district, crosswalk.geojson_district))

    def geojson_name(d):
        if d in GAP_PARENT_POLYGON:
            return GAP_PARENT_POLYGON[d]
        gj = cw_map.get(d)
        return None if pd.isna(gj) else gj

    master["geojson_name"] = master["district"].apply(geojson_name)
    missing = master[master["geojson_name"].isna()]
    if len(missing):
        raise SystemExit(f"Unmapped districts remain: {list(missing.district)}")

    out = []
    for name, g in master.groupby("geojson_name"):
        if len(g) == 1:
            r = g.iloc[0]
            v, x = float(r.vulnerability_index_pc1), r.elderly_share_65plus_pct
            lisa = lisa_by_district.get(r.district, "NS")
        else:
            w = g.total_population.values
            v = float((g.vulnerability_index_pc1.values * w).sum() / w.sum())
            x = float((g.elderly_share_65plus_pct.values * w).sum() / w.sum())
            lisa = next((lisa_by_district[d] for d in g.district if d in lisa_by_district), "NS")
        out.append({
            "name": name, "short": name.title(), "v": round(v, 4), "lisa": lisa,
            "x": None if pd.isna(x) else round(float(x), 3),
        })

    geo = json.load(open(geojson_path, encoding="utf-8"))
    geo_names = {f["properties"]["name"] for f in geo["features"]}
    my_names = {o["name"] for o in out}
    unmatched_geo = geo_names - my_names
    if unmatched_geo:
        raise SystemExit(f"GeoJSON polygons with no data after build: {unmatched_geo}")

    return out


if __name__ == "__main__":
    geojson_path = sys.argv[1] if len(sys.argv) > 1 else None
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    if not geojson_path:
        raise SystemExit("Usage: build_dashboard_regions.py <geojson_path> [out_json_path]")
    regions = build(geojson_path)
    text = json.dumps(regions, ensure_ascii=False)
    if out_path:
        Path(out_path).write_text(text, encoding="utf-8")
        print(f"Wrote {len(regions)} regions to {out_path}")
    else:
        print(text)
