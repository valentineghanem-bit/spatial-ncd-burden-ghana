"""
Build the 261-row regions JSON consumed by the HI-EI dashboard/poster generator
(_system/bespoke/bespoke_gen.js). Uses the vetted 261-to-GeoJSON crosswalk
(docs/district_crosswalk_261_to_260.csv) rather than naive name matching, which only
resolved 174/261 districts and left the majority of the choropleth blank.

Every one of the 261 master-sheet districts is kept as its own distinct entry (matching
the true census frame -- see [[ghana-261-districts]]), never merged or averaged down to
the GeoJSON's own smaller feature count. Two corrections applied on top of the crosswalk
for the `name` (polygon) each district points to, both verified directly against the
GeoJSON's actual feature list (259 features):
  - Sagnarigu Municipal: crosswalk marks this a structural_gap absorbed into Tamale
    Metropolitan, but the GeoJSON has its own distinct "SAGNERIGU" polygon (a spelling
    variant). Mapped directly rather than merged.
  - Ayawaso Central Municipal: crosswalk claims an exact match that does not exist in
    this GeoJSON (only West/North/East do). Points at the adjacent AYAWASO EAST
    MUNICIPAL polygon -- disclosed as a rendering fallback (this GeoJSON is missing that
    one polygon), never a change to the underlying district's own data.

Because the GeoJSON has only 259 features for 261 real districts (Awutu Senya
West/Guan/Ayawaso Central all point at a sibling's polygon, per the notes above), exactly
2 polygons end up carrying two districts' worth of choropleth colour on the map -- an
honest, disclosed characteristic of the boundary file, not something papered over by
averaging the two districts' values into one. The regions array itself always has 261
entries, matching the master CSV row count exactly.

Output: writes to the path passed as argv[1], or prints to stdout if omitted.
"""
import sys
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Districts whose crosswalk-listed GeoJSON name is wrong or missing; each is pointed at
# the nearest sibling polygon for MAP RENDERING ONLY. The district's own row, and its own
# value, are never dropped or merged with that sibling's.
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
    for r in master.itertuples():
        out.append({
            "name": r.geojson_name,
            "short": r.district,
            "v": round(float(r.vulnerability_index_pc1), 4),
            "lisa": lisa_by_district.get(r.district, "NS"),
            "x": None if pd.isna(r.elderly_share_65plus_pct) else round(float(r.elderly_share_65plus_pct), 3),
        })

    if len(out) != 261:
        raise SystemExit(f"Expected 261 districts, got {len(out)}")

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
