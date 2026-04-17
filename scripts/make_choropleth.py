"""Build an interactive Sweden map of 2050 crop-yield change (RCP 8.5).

Reads `outputs/predictions_2050_by_county.csv` (produced by pipeline.py's
predict stage) and the county centroid table; writes:

    outputs/sweden_yield_2050_map.html   - standalone interactive HTML
    outputs/figures/sweden_yield_2050_map.png - static image (if kaleido)

We use Plotly Scattergeo with county centroids as bubbles (offline-safe,
no external GeoJSON fetch needed). Includes a crop selector dropdown.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]

# Swedish län centroids (matches generate_synthetic_data.COUNTIES)
COUNTIES = [
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

OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)


def main() -> None:
    pred = pd.read_csv(OUT / "predictions_2050_by_county.csv")
    centroids = pd.DataFrame(COUNTIES,
                             columns=["County", "Latitude", "Longitude", "Elevation"])

    df = pred.merge(centroids, on="County", how="left")
    crops = sorted(df["Crop"].unique())

    # "All crops (average)" view
    avg = (df.groupby("County")
             .agg(pct_change=("pct_change", "mean"),
                  Latitude=("Latitude", "first"),
                  Longitude=("Longitude", "first"))
             .reset_index())
    avg["Crop"] = "All crops (avg)"
    avg["yield (kg/ha)"] = df.groupby("County")["yield (kg/ha)"].mean().values
    avg["yield_2009"] = df.groupby("County")["yield_2009"].mean().values
    df_all = pd.concat([avg, df], ignore_index=True)
    crops = ["All crops (avg)"] + crops

    traces = []
    for crop in crops:
        d = df_all[df_all["Crop"] == crop].dropna(subset=["pct_change"])
        hovertext = []
        for _, row in d.iterrows():
            y09 = row["yield_2009"] if "yield_2009" in row else float("nan")
            hovertext.append(
                f"<b>{row['County']}</b><br>"
                f"Crop: {row['Crop']}<br>"
                f"2009 yield: {y09:,.0f} kg/ha<br>"
                f"2050 yield: {row['yield (kg/ha)']:,.0f} kg/ha<br>"
                f"<b>Change: {row['pct_change']:+.1f}%</b>"
            )
        traces.append(go.Scattergeo(
            lon=d["Longitude"], lat=d["Latitude"],
            text=hovertext, hoverinfo="text",
            marker=dict(
                # tighter scaling + cmin/cmax tuned to actual range (|Δ| ≤ ~10%)
                size=(d["pct_change"].abs() * 1.8 + 8).clip(lower=6, upper=34),
                color=d["pct_change"],
                colorscale="RdBu",
                cmin=-10, cmax=10,
                reversescale=False,   # RdBu native: red=low/loss, blue=high/gain
                line=dict(color="black", width=0.5),
                colorbar=dict(title="% change<br>vs 2009",
                              thickness=14, len=0.7),
            ),
            name=crop, visible=(crop == "All crops (avg)"),
            showlegend=False,
        ))

    fig = go.Figure(data=traces)

    buttons = []
    for i, crop in enumerate(crops):
        vis = [j == i for j in range(len(crops))]
        buttons.append(dict(label=crop, method="update",
                            args=[{"visible": vis}]))

    fig.update_layout(
        # no layout title — the annotation below is the single header
        title=None,
        geo=dict(
            scope="europe",
            projection_type="mercator",
            center=dict(lat=62.5, lon=16.5),
            lataxis=dict(range=[54, 70]),
            lonaxis=dict(range=[10, 25]),
            showland=True, landcolor="#f4f1de",
            showocean=True, oceancolor="#cfe3f0",
            showlakes=True, lakecolor="#cfe3f0",
            countrycolor="#777", showcountries=True,
            bgcolor="#ffffff",
        ),
        paper_bgcolor="white",
        font=dict(family="system-ui, -apple-system, sans-serif", size=12),
        width=900, height=780,
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            x=0.02, xanchor="left",
            y=1.00, yanchor="bottom",
            bgcolor="white", bordercolor="#888",
        )],
        margin=dict(l=20, r=20, t=130, b=30),
    )

    annotations = [
        dict(text="<b>Sweden: 2050 projected crop yield change under RCP 8.5</b>",
             xref="paper", yref="paper", x=0.5, y=1.16,
             xanchor="center", showarrow=False, font=dict(size=16)),
        dict(text="Bubble size = magnitude of change &nbsp;|&nbsp; "
                  "Color: <span style='color:#b2182b'><b>red = loss</b></span>, "
                  "<span style='color:#2166ac'><b>blue = gain</b></span>",
             xref="paper", yref="paper", x=0.5, y=1.11,
             xanchor="center", showarrow=False, font=dict(size=11, color="#555")),
        dict(text="Source: Swedish crop-yield XGBoost model &nbsp;|&nbsp; "
                  "County centroids &nbsp;|&nbsp; 2050s = 2041-2070 normal",
             xref="paper", yref="paper", x=0.98, y=-0.03,
             xanchor="right", yanchor="top", showarrow=False,
             font=dict(size=9, color="#888")),
    ]
    fig.update_layout(annotations=annotations)

    html_path = OUT / "sweden_yield_2050_map.html"
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    print(f"-> {html_path}")

    # Static matplotlib fallback (no Chrome needed)
    import matplotlib.pyplot as plt
    import numpy as np

    avg_df = df_all[df_all["Crop"] == "All crops (avg)"].dropna(subset=["pct_change"])
    mfig, mx = plt.subplots(figsize=(7, 10))
    mx.set_xlim(10, 25); mx.set_ylim(54, 70)
    mx.set_aspect(1.8)
    mx.set_facecolor("#e7f0f8")
    # A very rough Sweden outline just to orient the viewer
    sweden_outline = [
        (12.1, 55.35), (12.4, 56.3), (11.3, 58.4), (11.6, 60.2),
        (12.3, 61.0), (12.8, 63.0), (14.2, 64.5), (14.5, 66.0),
        (17.5, 68.2), (20.2, 69.0), (23.8, 68.3), (24.2, 66.5),
        (23.8, 65.8), (22.0, 65.7), (20.3, 63.5), (19.0, 62.5),
        (18.1, 60.6), (17.0, 58.8), (16.6, 57.0), (14.9, 55.7),
        (13.0, 55.3), (12.1, 55.35),
    ]
    xs, ys = zip(*sweden_outline)
    mx.fill(xs, ys, facecolor="#f4f1de", edgecolor="#555", linewidth=1.2, zorder=1)

    sizes = (avg_df["pct_change"].abs() * 14 + 50).clip(upper=500)
    sc = mx.scatter(avg_df["Longitude"], avg_df["Latitude"],
                    s=sizes, c=avg_df["pct_change"],
                    cmap="RdBu", vmin=-10, vmax=10,   # red=loss, blue=gain
                    edgecolor="black", linewidth=0.6, zorder=2)
    for _, r in avg_df.iterrows():
        mx.annotate(r["County"], (r["Longitude"], r["Latitude"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=7, color="#222")
    plt.colorbar(sc, ax=mx, shrink=0.6, label="% yield change vs 2009")
    mx.set_title("Sweden 2050 average crop yield change (RCP 8.5)",
                 fontsize=13, pad=10)
    mx.set_xlabel("Longitude"); mx.set_ylabel("Latitude")
    mfig.savefig(FIG / "sweden_yield_2050_map.png", dpi=140, bbox_inches="tight")
    plt.close(mfig)
    print(f"-> {FIG / 'sweden_yield_2050_map.png'}")

    # Per-crop static grid for the PDF case study
    key_crops = ["autumn_wheat", "spring_wheat", "autumn_barley",
                 "potato", "sugar_beet", "Flax"]
    key_crops = [c for c in key_crops if c in df_all["Crop"].unique()]
    ncols = 3
    nrows = int(np.ceil(len(key_crops) / ncols))
    gfig, gax = plt.subplots(nrows, ncols, figsize=(14, 5*nrows))
    gax = gax.flatten() if hasattr(gax, "flatten") else [gax]
    for ax, crop in zip(gax, key_crops):
        d = df_all[df_all["Crop"] == crop].dropna(subset=["pct_change"])
        ax.set_xlim(10, 25); ax.set_ylim(54, 70); ax.set_aspect(1.8)
        ax.set_facecolor("#e7f0f8")
        ax.fill(xs, ys, facecolor="#f4f1de", edgecolor="#555", linewidth=0.8)
        ax.scatter(d["Longitude"], d["Latitude"],
                   s=(d["pct_change"].abs() * 10 + 40).clip(upper=300),
                   c=d["pct_change"], cmap="RdBu", vmin=-15, vmax=15,   # red=loss, blue=gain
                   edgecolor="black", linewidth=0.5)
        ax.set_title(crop, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in gax[len(key_crops):]:
        ax.axis("off")
    gfig.suptitle("Sweden 2050 yield change by crop (red = loss, blue = gain)",
                  fontsize=13, y=1.01)
    gfig.savefig(FIG / "sweden_yield_2050_map_grid.png", dpi=140, bbox_inches="tight")
    plt.close(gfig)
    print(f"-> {FIG / 'sweden_yield_2050_map_grid.png'}")


if __name__ == "__main__":
    main()
