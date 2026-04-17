"""
Synthetic data generator for the Swedish crop yield XGBoost project.

Reproduces the feature schema and data shape described in the thesis:
"Analysis of climate change impacts on Swedish agriculture in 2050 using
machine learning" (U. Sen, 2021).

This generates realistic — but synthesized — agro-climatic features and
crop yields for 21 Swedish counties across 1965-2009, plus a 2050
projection under RCP 8.5. Feature distributions and county/crop-level
baselines are anchored to the thesis tables and figures so that a model
trained on this data produces plausible behavior.

REAL-DATA EQUIVALENTS (see scripts/fetch_real_data.py):
- input_se_1901-2009MSYT.csv  -> ClimateEU v4.63 historical export
- input_se_Ensemble15_rcp85_2050sMSY.csv -> ClimateEU RCP 8.5 projection
- Harvests_by_county_year*.csv -> SCB PxWeb JO0601 table
- Monthly_sunshine_sweden.csv -> Wikipedia List_of_cities_by_sunshine_duration
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
OUT = Path(__file__).resolve().parents[1] / "data" / "synthetic"
OUT.mkdir(parents=True, exist_ok=True)

# --- Swedish counties with approximate centroid lat/lon/elevation ----------
# Source: Sweden administrative geography (21 modern län)
COUNTIES = [
    # name, lat, lon, elev_m
    ("Stockholms",       59.33, 18.07,  20),
    ("Uppsala",          59.86, 17.64,  30),
    ("Sodermanlands",    59.03, 16.46,  40),
    ("Ostergotland",     58.40, 15.62,  50),
    ("Jonkoping",        57.78, 14.16, 180),
    ("Kronoberg",        56.88, 14.81, 170),
    ("Kalmar",           56.66, 16.36,  25),
    ("Gotland",          57.47, 18.48,  25),
    ("Blekinge",         56.25, 15.28,  30),
    ("Skane",            55.99, 13.60,  60),
    ("Halland",          56.67, 12.86,  40),
    ("Vastra Gotaland",  58.22, 12.50,  75),
    ("Varmlands",        59.40, 13.50, 120),
    ("Orebro",           59.27, 15.21,  70),
    ("Vastmanland",      59.61, 16.55,  40),
    ("Dalarna",          60.67, 15.00, 220),
    ("Gavleborg",        61.30, 16.25,  80),
    ("Vasternorrland",   63.18, 17.10, 110),
    ("Jamtland",         63.18, 14.64, 350),
    ("Vasterbotten",     64.45, 19.30, 180),
    ("Norrbotten",       66.43, 20.61, 100),
]

CROPS = [
    "autumn_wheat", "spring_wheat", "rye", "autumn_barley", "spring_grain",
    "oats", "spring_rye_wheat", "mixed_grain", "peas", "field_beans",
    "maize", "potato", "sugar_beet", "autumn_rapeseed", "spring_rapeseed",
    "hostryps", "spring_turnip", "Flax",
]

# Thesis Table 1 — 2009 reference mean yields (kg/ha)
# Missing (0 in thesis) filled with a biologically reasonable value so the
# synthetic signal is non-degenerate.
YIELD_2009 = {
    "Flax": 1840.0, "autumn_barley": 5350.0, "autumn_rapeseed": 3085.7,
    "autumn_wheat": 5003.2, "field_beans": 3425.0, "hostryps": 2200.0,
    "maize": 6550.0, "mixed_grain": 3353.3, "oats": 4008.8, "peas": 2964.4,
    "potato": 25777.5, "rye": 5213.8, "spring_grain": 3940.0,
    "spring_rapeseed": 1907.5, "spring_rye_wheat": 3400.0,
    "spring_turnip": 1320.0, "spring_wheat": 4483.3, "sugar_beet": 56875.0,
}

# Relative climate sensitivity (how much yield drops with +2C / shifted precip)
# Anchored to thesis "relative decrease" column.
CLIMATE_SENSITIVITY = {
    "Flax": 0.41, "autumn_barley": 0.22, "autumn_rapeseed": 0.12,
    "autumn_wheat": -0.03, "field_beans": 0.33, "hostryps": 0.05,
    "maize": 0.27, "mixed_grain": 0.08, "oats": 0.17, "peas": 0.13,
    "potato": 0.00, "rye": 0.14, "spring_grain": 0.10,
    "spring_rapeseed": 0.15, "spring_rye_wheat": 0.04,
    "spring_turnip": 0.01, "spring_wheat": 0.18, "sugar_beet": 0.32,
}

# County productivity modifier (Skane is the agricultural heartland)
# Derived qualitatively from figures 6-20.
COUNTY_MOD = {
    "Skane": 1.25, "Halland": 1.10, "Skaraborgs": 1.05,
    "Ostergotland": 1.08, "Uppsala": 1.02, "Vastra Gotaland": 1.00,
    "Stockholms": 0.95, "Sodermanlands": 1.00, "Jonkoping": 0.90,
    "Kronoberg": 0.85, "Kalmar": 0.95, "Gotland": 1.00, "Blekinge": 0.95,
    "Varmlands": 0.85, "Orebro": 0.95, "Vastmanland": 0.95, "Dalarna": 0.80,
    "Gavleborg": 0.78, "Vasternorrland": 0.72, "Jamtland": 0.65,
    "Vasterbotten": 0.70, "Norrbotten": 0.65,
}

YEARS_HIST = list(range(1901, 2010))   # climate historical
YEARS_YIELD = list(range(1965, 2020))  # yield observations
YEAR_2050 = 2050


def climate_for(lat: float, elev_m: float, year: int) -> dict:
    """Generate plausible agro-climatic features for a location/year.

    Temperature decreases ~0.7 C per degree latitude and ~0.6 C per 100 m
    elevation. A mild linear warming trend of 0.015 C/year is applied.
    Precipitation is higher in the west/south and lower in the north.
    """
    base_mat = 8.5 - 0.70 * (lat - 55.0) - 0.006 * elev_m + 0.015 * (year - 1970)
    seasonal_range = 18 + 0.25 * (lat - 55.0)  # bigger seasonal swing up north
    mwmt = base_mat + seasonal_range / 2 + RNG.normal(0, 0.5)
    mcmt = base_mat - seasonal_range / 2 + RNG.normal(0, 0.7)
    mat  = base_mat + RNG.normal(0, 0.4)
    td   = mwmt - mcmt
    map_ = 620 - 4.0 * (lat - 55.0) + RNG.normal(0, 60)
    map_ = max(350, map_)
    msp  = 0.45 * map_ + RNG.normal(0, 20)

    ah_m = (mat + 10) / (map_ / 1000)
    sh_m = mwmt / (msp / 1000)

    dd0  = max(0, -25 * mcmt + RNG.normal(0, 50))
    dd5  = max(0, 1800 - 150 * (lat - 55.0) + 18 * (year - 1970) + RNG.normal(0, 80))
    dd18 = max(0, 3200 - 80 * (mat - 5) + RNG.normal(0, 100))
    dd_over18 = max(0, 30 + 8 * (mwmt - 16) + RNG.normal(0, 10))
    nffd = max(80, 210 - 6 * (lat - 55.0) + RNG.normal(0, 8))
    ffp  = max(60, nffd - 30 + RNG.normal(0, 5))
    bffp = 100 + 3 * (lat - 55.0) + RNG.normal(0, 4)   # Julian start
    effp = bffp + ffp
    pas  = max(0, 40 + 18 * (lat - 55.0) + RNG.normal(0, 20))
    emt  = mcmt - 10 - RNG.uniform(0, 6)
    eref = 800 - 10 * (lat - 55.0) + RNG.normal(0, 40)
    cmd  = max(0, eref - msp * 0.8 + RNG.normal(0, 20))

    # Seasonal means
    season_shift = {"wt": -seasonal_range / 2 - 2, "sp": 2, "sm": seasonal_range / 2, "at": 3}
    seasonal = {}
    for tag, shift in season_shift.items():
        seasonal[f"TAV_{tag}"]  = mat + shift + RNG.normal(0, 0.5)
        seasonal[f"TMAX_{tag}"] = seasonal[f"TAV_{tag}"] + 5 + RNG.normal(0, 0.5)
        seasonal[f"TMIN_{tag}"] = seasonal[f"TAV_{tag}"] - 5 + RNG.normal(0, 0.5)
        seasonal[f"PPT_{tag}"]  = map_ / 4 + RNG.normal(0, 15)

    # Monthly (12 months each for Tave/Tmax/Tmin/PPT)
    months = np.arange(1, 13)
    month_tave = mat + (seasonal_range / 2) * np.sin(2 * np.pi * (months - 4) / 12)
    month_tave = month_tave + RNG.normal(0, 0.4, 12)
    month_tmax = month_tave + 5 + RNG.normal(0, 0.3, 12)
    month_tmin = month_tave - 5 + RNG.normal(0, 0.3, 12)
    month_ppt  = (map_ / 12) * (1 + 0.3 * np.sin(2 * np.pi * (months - 8) / 12))
    month_ppt  = np.maximum(5, month_ppt + RNG.normal(0, 5, 12))

    row = {
        "MAT": mat, "MWMT": mwmt, "MCMT": mcmt, "TD": td, "MAP": map_, "MSP": msp,
        "AH:M": ah_m, "SH:M": sh_m,
        "DD<0": dd0, "DD>5": dd5, "DD<18": dd18, "DD>18": dd_over18,
        "NFFD": nffd, "FFP": ffp, "bFFP": bffp, "eFFP": effp,
        "PAS": pas, "EMT": emt, "Eref": eref, "CMD": cmd,
    }
    row.update(seasonal)
    for i, m in enumerate(months):
        row[f"Tave{m:02d}"] = month_tave[i]
        row[f"Tmax{m:02d}"] = month_tmax[i]
        row[f"Tmin{m:02d}"] = month_tmin[i]
        row[f"PPT{m:02d}"]  = month_ppt[i]
    return row


def rcp85_shift(base: dict) -> dict:
    """Apply RCP 8.5 shifts to a base climate row: +2 C mean, +30-50% winter
    precipitation, -30 mm summer precipitation, growing season extended."""
    out = dict(base)
    dT = 2.0 + RNG.normal(0, 0.3)
    for key in list(out.keys()):
        if key.startswith(("MAT", "MWMT", "MCMT", "TAV_", "TMAX_", "TMIN_", "Tave", "Tmax", "Tmin", "EMT")):
            out[key] = out[key] + dT
    # Precip shifts
    out["MAP"] = out["MAP"] * RNG.uniform(1.05, 1.12)
    out["MSP"] = out["MSP"] * RNG.uniform(0.80, 0.95)
    out["PPT_wt"] = out["PPT_wt"] * RNG.uniform(1.30, 1.50)
    out["PPT_sp"] = out["PPT_sp"] * RNG.uniform(1.05, 1.15)
    out["PPT_sm"] = out["PPT_sm"] * RNG.uniform(0.75, 0.90)
    out["PPT_at"] = out["PPT_at"] * RNG.uniform(1.00, 1.10)
    for m in range(1, 13):
        mult = 1.25 if m in (12, 1, 2) else (0.85 if m in (6, 7, 8) else 1.05)
        out[f"PPT{m:02d}"] = out[f"PPT{m:02d}"] * mult * RNG.uniform(0.95, 1.05)
    # Derived updates
    out["TD"] = out["MWMT"] - out["MCMT"]
    out["AH:M"] = (out["MAT"] + 10) / (out["MAP"] / 1000)
    out["SH:M"] = out["MWMT"] / (out["MSP"] / 1000)
    out["DD>5"]  = out["DD>5"] * 1.18
    out["DD>18"] = out["DD>18"] * 1.40
    out["DD<0"]  = out["DD<0"] * 0.85
    out["NFFD"] = out["NFFD"] + 10
    out["FFP"]  = out["FFP"] + 12
    out["CMD"]  = out["CMD"] * 1.15
    return out


def yield_for(crop: str, county: str, year: int, climate: dict) -> float:
    """Produce a synthetic yield with signal from year/county/climate."""
    base = YIELD_2009[crop]
    county_mult = COUNTY_MOD.get(county, 1.0)
    # Long-run agronomy trend ~ +0.6%/year before climate pressure
    trend = 1 + 0.006 * (year - 2009)
    # Climate pressure: above ideal MAT (7 C for most Swedish crops) yields drop
    ideal_mat = 6.5
    stress = max(0.0, climate["MAT"] - ideal_mat)
    sens = CLIMATE_SENSITIVITY[crop]
    climate_factor = 1 - sens * (stress / 2.0) * 0.5
    # Water stress
    water_factor = 1 - 0.08 * max(0, (climate["CMD"] - 400) / 400)
    # Small crop-county interactions: Skane boosts sugar beet, maize
    if county == "Skane" and crop in ("sugar_beet", "maize"):
        county_mult *= 1.05
    if county in ("Norrbotten", "Vasterbotten", "Jamtland") and crop in ("maize", "sugar_beet"):
        county_mult *= 0.5  # harder to grow up north
    noise = RNG.normal(1.0, 0.08)
    y = base * county_mult * trend * climate_factor * water_factor * noise
    return float(max(0.0, y))


# --- Build frames ----------------------------------------------------------
print("Generating historical climate (1901-2009) for 21 counties...")
climate_rows = []
for year in YEARS_HIST:
    for county, lat, lon, elev in COUNTIES:
        row = {"Year": year, "County": county, "Country": "Sweden",
               "Latitude": lat, "Longitude": lon, "Elevation": elev}
        row.update(climate_for(lat, elev, year))
        climate_rows.append(row)
climate_hist = pd.DataFrame(climate_rows)
climate_hist.to_csv(OUT / "input_se_1901-2009MSYT.csv", index=False)
print(f"  -> {len(climate_hist)} rows, {len(climate_hist.columns)} cols")

print("Generating 2050 projection (RCP 8.5)...")
proj_rows = []
for county, lat, lon, elev in COUNTIES:
    base = climate_for(lat, elev, 2005)  # use 2005 baseline, then apply shifts
    shifted = rcp85_shift(base)
    row = {"Year": 2050, "County": county, "Country": "Sweden",
           "Latitude": lat, "Longitude": lon, "Elevation": elev}
    row.update(shifted)
    proj_rows.append(row)
climate_2050 = pd.DataFrame(proj_rows)
climate_2050.to_csv(OUT / "input_se_Ensemble15_rcp85_2050sMSY.csv", index=False)
print(f"  -> {len(climate_2050)} rows, {len(climate_2050.columns)} cols")

print("Generating observed crop yields (1965-2019)...")
yield_rows = []
hist_index = climate_hist.set_index(["Year", "County"])
for year in YEARS_YIELD:
    for county, lat, lon, elev in COUNTIES:
        # For the overlap with climate (1965-2009), use real climate row;
        # beyond 2009, extrapolate by repeating 2009 climate + trend.
        src_year = min(year, 2009)
        clim = hist_index.loc[(src_year, county)].to_dict()
        for crop in CROPS:
            # Not every crop is observed in every county/year; drop ~15% at random
            if RNG.random() < 0.15:
                continue
            y = yield_for(crop, county, year, clim)
            yield_rows.append({"County": county, "Crop": crop, "Year": year,
                               "yield (kg/ha)": round(y, 1)})
obs_yield = pd.DataFrame(yield_rows)
obs_yield.to_csv(OUT / "Harvests_by_county_year1965_2019_Hectare_harvest_kg_per_hectare.csv", index=False)
print(f"  -> {len(obs_yield)} rows")

print("Generating monthly sunshine (Gothenburg/Stockholm avg)...")
# From Wikipedia-style monthly sunshine hours per day, averaged for Sweden
sunshine = {
    "Month": [f"{i:02d}" for i in range(1, 13)],
    "Avg":   [45, 80, 145, 200, 285, 295, 285, 240, 175, 110, 55, 35],
}
pd.DataFrame(sunshine).to_csv(OUT / "Monthly_sunshine_sweden.csv", index=False)
print("Done.")
print(f"Files in {OUT}:")
for p in sorted(OUT.iterdir()):
    print(f"  {p.name}  ({p.stat().st_size / 1024:.1f} KB)")
