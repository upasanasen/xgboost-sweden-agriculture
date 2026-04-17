"""
Fetch real data for the Swedish crop yield project.

Run this script *on a machine with open internet access* to replace the
synthesized reproduction data in `data/synthetic/` with authoritative
sources. The downstream notebook (`notebooks/xgboost_pipeline.ipynb`)
looks at `data/real/` first and falls back to `data/synthetic/`.

Sources
-------
1. SCB (Statistics Sweden) — crop yield per hectare by county.
   PxWeb API endpoint (as of 2025):
       https://api.scb.se/OV0104/v1/doris/en/ssd/START/JO/JO0601/JO0601B/SkordarGrodaL
   Public, no auth. Supports JSON POST with variable selection.

2. ClimateEU (Wang & Hamann, v4.63) — agro-climatic features.
   Download the Windows tool here:
       https://sites.ualberta.ca/~ahamann/data/climateeu.html
   Feed in a CSV of county centroids (Latitude, Longitude, Elevation) to
   export MAT, MWMT, monthly Tave/Tmax/Tmin/PPT, seasonal vars, etc. for
   the normal period 1961-1990, historical (1901-2009), and the
   Ensemble15 RCP 8.5 projection for the 2050s (2041-2070).

3. Monthly sunshine — Wikipedia "List of cities by sunshine duration"
   for Gothenburg and Stockholm, averaged.

Usage
-----
    python scripts/fetch_real_data.py --scb
    # then run ClimateEU on the exported county centroid CSV and drop the
    # two output CSVs into data/real/

The SCB step writes:
    data/real/Harvests_by_county_year1965_2019_Hectare_harvest_kg_per_hectare.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

try:
    import requests
except ImportError:
    requests = None

REAL = Path(__file__).resolve().parents[1] / "data" / "real"
REAL.mkdir(parents=True, exist_ok=True)

SCB_ENDPOINT = (
    "https://api.scb.se/OV0104/v1/doris/en/ssd/START/JO/JO0601/"
    "JO0601B/SkordarGrodaL"
)

# Crop codes in SCB table JO0601B, SkordarGrodaL (Swedish-English aliases)
SCB_CROPS = {
    "01": "autumn_wheat", "02": "spring_wheat", "03": "rye",
    "04": "autumn_barley", "05": "spring_barley", "06": "oats",
    "07": "mixed_grain", "08": "peas", "09": "field_beans",
    "10": "autumn_rapeseed", "11": "spring_rapeseed", "12": "hostryps",
    "13": "Flax", "14": "potato", "15": "sugar_beet",
}

COUNTY_CODES = [
    "01", "03", "04", "05", "06", "07", "08", "09", "10", "12",
    "13", "14", "17", "18", "19", "20", "21", "22", "23", "24", "25",
]  # 21 modern Swedish län


def fetch_scb_yields(years: Iterable[int] = range(1965, 2020)) -> None:
    """Pull county x crop x year yield from SCB PxWeb and save as CSV.

    Build request body once and paginate by chunks of years if necessary;
    the PxWeb API caps responses to ~10_000 cells per call.
    """
    if requests is None:
        sys.exit("Install requests: pip install requests")
    body = {
        "query": [
            {"code": "Region", "selection": {"filter": "item", "values": COUNTY_CODES}},
            {"code": "Groda", "selection": {"filter": "item", "values": list(SCB_CROPS)}},
            {"code": "Tid", "selection": {"filter": "item", "values": [str(y) for y in years]}},
        ],
        "response": {"format": "json"},
    }
    print(f"POST {SCB_ENDPOINT}")
    resp = requests.post(SCB_ENDPOINT, json=body, timeout=60)
    resp.raise_for_status()
    payload = resp.json()

    import pandas as pd
    rows = []
    for entry in payload["data"]:
        region, crop_code, year = entry["key"]
        value = entry["values"][0]
        if value in ("..", None, ""):
            continue
        rows.append({
            "County": region,
            "Crop": SCB_CROPS.get(crop_code, crop_code),
            "Year": int(year),
            "yield (kg/ha)": float(value),
        })
    df = pd.DataFrame(rows)
    out = REAL / "Harvests_by_county_year1965_2019_Hectare_harvest_kg_per_hectare.csv"
    df.to_csv(out, index=False)
    print(f"  -> {len(df)} rows -> {out}")


def write_centroid_csv() -> None:
    """Write a CSV of county centroids for ClimateEU to ingest."""
    from scripts.generate_synthetic_data import COUNTIES  # reuse table

    import pandas as pd
    df = pd.DataFrame(COUNTIES, columns=["County", "Latitude", "Longitude", "Elevation"])
    df.insert(0, "ID1", df.index + 1)
    df.insert(1, "ID2", df["County"])
    out = REAL / "sweden_county_centroids_for_climateeu.csv"
    df.to_csv(out, index=False)
    print(f"Feed this to ClimateEU: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scb", action="store_true", help="Fetch SCB yields")
    parser.add_argument("--centroids", action="store_true",
                        help="Write county centroid CSV for ClimateEU")
    args = parser.parse_args()
    if args.scb:
        fetch_scb_yields()
    if args.centroids:
        write_centroid_csv()
    if not (args.scb or args.centroids):
        parser.print_help()


if __name__ == "__main__":
    main()
