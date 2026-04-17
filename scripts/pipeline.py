"""
End-to-end ML pipeline for the Swedish crop yield project.

Runs as staged checkpoints so each step saves intermediate artefacts and
can be resumed if interrupted. Stages:
  1. prepare   - load + merge + one-hot encode + save X/y
  2. baselines - LinearRegression + RandomForest (small, n_jobs=1)
  3. xgb       - XGBoost default + RandomizedSearchCV-tuned, save model
  4. shap      - SHAP feature importance on 500-row sample
  5. predict   - 2050 RCP 8.5 per-county-per-crop yield projection
  6. report    - model_comparison.csv + yield_change_summary.csv + figs

Run from the project root:
    python scripts/pipeline.py            # run all remaining stages
    python scripts/pipeline.py --force    # re-run everything
    python scripts/pipeline.py --stage shap   # only a specific stage
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "2")

ROOT = Path(__file__).resolve().parents[1]
DATA_REAL = ROOT / "data" / "real"
DATA_SYN = ROOT / "data" / "synthetic"
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)
CKPT = OUT / "checkpoints"; CKPT.mkdir(exist_ok=True)


# ----------------------------- Helpers --------------------------------------

def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def choose_dir() -> Path:
    if (DATA_REAL / "input_se_1901-2009MSYT.csv").exists():
        _log(f">> using REAL data in {DATA_REAL}")
        return DATA_REAL
    _log(f">> using SYNTHETIC reproduction data in {DATA_SYN}")
    return DATA_SYN


def load_sunshine(path: Path) -> pd.DataFrame:
    ss = pd.read_csv(path)
    ss = ss[["Avg"]].T
    ss = ss.reset_index(drop=True)
    ss.columns = [f"SS{i+1:02d}" for i in range(12)]
    return ss


def add_sunshine(df_input: pd.DataFrame, ss: pd.DataFrame) -> pd.DataFrame:
    ss_rep = pd.concat([ss] * df_input.shape[0], ignore_index=True)
    return pd.concat([df_input.reset_index(drop=True), ss_rep], axis=1)


# ----------------------------- Stage 1 --------------------------------------

def stage_prepare(force: bool = False) -> dict:
    ckpt = CKPT / "prepare.npz"
    cols_path = MODELS / "feature_columns.json"
    merged_path = CKPT / "merged.pkl"

    if ckpt.exists() and cols_path.exists() and merged_path.exists() and not force:
        _log("[prepare] cached; skipping")
        data = np.load(ckpt, allow_pickle=True)
        feature_cols = json.load(open(cols_path))
        return {"X": data["X"], "y": data["y"],
                "feature_cols": feature_cols,
                "merged_path": merged_path}

    src = choose_dir()
    _log("[prepare] loading CSVs...")
    df_climate = pd.read_csv(src / "input_se_1901-2009MSYT.csv")
    df_yield   = pd.read_csv(src / "Harvests_by_county_year1965_2019_Hectare_harvest_kg_per_hectare.csv")
    df_2050    = pd.read_csv(src / "input_se_Ensemble15_rcp85_2050sMSY.csv")
    df_sun     = load_sunshine(src / "Monthly_sunshine_sweden.csv")

    df_climate = add_sunshine(df_climate, df_sun)
    df_2050    = add_sunshine(df_2050, df_sun)
    _log(f"[prepare] climate {df_climate.shape}, yield {df_yield.shape}, "
         f"proj2050 {df_2050.shape}")

    merged = df_yield.merge(df_climate, on=["Year", "County"], how="inner")
    merged = merged.drop(columns=["Country"], errors="ignore")
    merged = merged.dropna(subset=["yield (kg/ha)"])
    merged = pd.get_dummies(merged, columns=["Crop"], prefix="Crop")

    feature_cols = [c for c in merged.columns
                    if c not in ("Year", "County", "yield (kg/ha)")]
    X = merged[feature_cols].astype(float).values
    y = merged["yield (kg/ha)"].astype(float).values
    _log(f"[prepare] training matrix X={X.shape}, y={y.shape}, "
         f"features={len(feature_cols)}")

    merged.to_pickle(merged_path)
    df_2050.to_pickle(CKPT / "proj2050.pkl")
    np.savez_compressed(ckpt, X=X, y=y)
    json.dump(feature_cols, open(cols_path, "w"))
    _log(f"[prepare] saved {ckpt.name}")

    # Quick EDA figures
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140,
                         "figure.autolayout": True})
    sns.set_theme(style="whitegrid")

    eda = merged.groupby("Year")["yield (kg/ha)"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(eda["Year"], eda["yield (kg/ha)"], linewidth=2, color="#2a9d8f")
    ax.set_title("Average observed crop yield across Sweden, 1965-2019")
    ax.set_xlabel("Year"); ax.set_ylabel("Yield (kg/ha)")
    fig.savefig(FIG / "eda_yield_trend.png", bbox_inches="tight")
    plt.close(fig)

    key_cols = [c for c in ["MAT", "MWMT", "MCMT", "MAP", "MSP",
                            "DD>5", "NFFD", "CMD", "yield (kg/ha)"]
                if c in merged.columns]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(merged[key_cols].corr(), annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation: key climate features vs yield")
    fig.savefig(FIG / "eda_correlation.png", bbox_inches="tight")
    plt.close(fig)

    return {"X": X, "y": y, "feature_cols": feature_cols,
            "merged_path": merged_path}


# ----------------------------- Stage 2 --------------------------------------

def stage_baselines(state: dict, force: bool = False) -> dict:
    """Fit LinearRegression + RandomForest with manual CV (low memory)."""
    ckpt = CKPT / "baselines.json"
    if ckpt.exists() and not force:
        _log("[baselines] cached; skipping")
        return {"results": json.load(open(ckpt))}

    import gc
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    from sklearn.model_selection import KFold, train_test_split

    X, y = state["X"], state["y"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=7)
    np.savez_compressed(CKPT / "split.npz",
                        X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te)
    _log(f"[baselines] split: train={X_tr.shape}, test={X_te.shape}")

    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    results = []

    def manual_cv_mae(factory, name: str) -> tuple[float, float]:
        maes = []
        for i, (tr_idx, va_idx) in enumerate(cv.split(X_tr)):
            m = factory()
            m.fit(X_tr[tr_idx], y_tr[tr_idx])
            pred = m.predict(X_tr[va_idx])
            maes.append(mean_absolute_error(y_tr[va_idx], pred))
            _log(f"  {name} fold {i+1}/5  MAE={maes[-1]:.1f}")
            del m
            gc.collect()
        return float(np.mean(maes)), float(np.std(maes))

    _log("[baselines] fitting Linear Regression (5-fold manual CV)...")
    mae_mean, mae_std = manual_cv_mae(LinearRegression, "LR")
    lr = LinearRegression().fit(X_tr, y_tr)
    y_pred = lr.predict(X_te)
    results.append({
        "model": "Linear Regression",
        "cv_mae_mean": mae_mean, "cv_mae_std": mae_std,
        "test_mae":   float(mean_absolute_error(y_te, y_pred)),
        "test_rmse":  float(np.sqrt(mean_squared_error(y_te, y_pred))),
        "test_r2":    float(r2_score(y_te, y_pred)),
    })
    _log(f"  LR    MAE={mae_mean:.1f} +/-{mae_std:.1f}")
    del lr; gc.collect()

    _log("[baselines] fitting Random Forest (n_estimators=40, depth=8)...")
    rf_factory = lambda: RandomForestRegressor(
        n_estimators=40, max_depth=8, min_samples_leaf=5,
        random_state=7, n_jobs=1)
    mae_mean, mae_std = manual_cv_mae(rf_factory, "RF")
    rf = rf_factory().fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    results.append({
        "model": "Random Forest",
        "cv_mae_mean": mae_mean, "cv_mae_std": mae_std,
        "test_mae":   float(mean_absolute_error(y_te, y_pred)),
        "test_rmse":  float(np.sqrt(mean_squared_error(y_te, y_pred))),
        "test_r2":    float(r2_score(y_te, y_pred)),
    })
    _log(f"  RF    MAE={mae_mean:.1f} +/-{mae_std:.1f}")

    json.dump(results, open(ckpt, "w"), indent=2)
    _log(f"[baselines] saved {ckpt.name}")
    return {"results": results}


# ----------------------------- Stage 3 --------------------------------------

def stage_xgb(state: dict, force: bool = False) -> dict:
    """XGBoost default + holdout-tuned XGBoost.

    To keep memory low we use a single 80/20 holdout for tuning (fast,
    deterministic) and then report 5-fold CV on the tuned model.
    """
    ckpt = CKPT / "xgb.json"
    model_file = MODELS / "trained_xgb_tuned.pkl"
    if ckpt.exists() and model_file.exists() and not force:
        _log("[xgb] cached; skipping")
        return {"results": json.load(open(ckpt)),
                "xgb_tuned_path": model_file}

    import gc
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    from sklearn.model_selection import KFold, train_test_split
    from xgboost import XGBRegressor

    split = np.load(CKPT / "split.npz")
    X_tr, X_te = split["X_tr"], split["X_te"]
    y_tr, y_te = split["y_tr"], split["y_te"]
    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    results = []

    def cv_mae(factory, name: str, folds=5) -> tuple[float, float]:
        _cv = KFold(n_splits=folds, shuffle=True, random_state=1)
        maes = []
        for i, (tr_idx, va_idx) in enumerate(_cv.split(X_tr)):
            m = factory()
            m.fit(X_tr[tr_idx], y_tr[tr_idx])
            maes.append(mean_absolute_error(y_tr[va_idx], m.predict(X_tr[va_idx])))
            _log(f"  {name} fold {i+1}/{folds}  MAE={maes[-1]:.1f}")
            del m; gc.collect()
        return float(np.mean(maes)), float(np.std(maes))

    _log("[xgb] XGBoost defaults - 5-fold CV")
    mae_mean, mae_std = cv_mae(
        lambda: XGBRegressor(random_state=7, n_jobs=2, tree_method="hist"),
        "XGBd")
    xgb_default = XGBRegressor(random_state=7, n_jobs=2, tree_method="hist")
    xgb_default.fit(X_tr, y_tr)
    y_pred = xgb_default.predict(X_te)
    results.append({
        "model": "XGBoost (default)",
        "cv_mae_mean": mae_mean, "cv_mae_std": mae_std,
        "test_mae":   float(mean_absolute_error(y_te, y_pred)),
        "test_rmse":  float(np.sqrt(mean_squared_error(y_te, y_pred))),
        "test_r2":    float(r2_score(y_te, y_pred)),
    })
    _log(f"  XGB default MAE={mae_mean:.1f} +/-{mae_std:.1f}")
    del xgb_default; gc.collect()

    _log("[xgb] tuning on 80/20 holdout (6 candidates)")
    X_tune, X_val, y_tune, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=11)

    rng = np.random.default_rng(42)
    param_dist = {
        "max_depth":        [4, 6, 8],
        "learning_rate":    [0.03, 0.05, 0.08, 0.1],
        "n_estimators":     [200, 300, 500],
        "min_child_weight": [1, 3, 5],
        "subsample":        [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
    }
    candidates, seen = [], set()
    while len(candidates) < 6:
        p = {k: (v[int(rng.integers(len(v)))]) for k, v in param_dist.items()}
        key = tuple(sorted(p.items()))
        if key in seen:
            continue
        seen.add(key); candidates.append(p)

    best = (float("inf"), None)
    for ci, params in enumerate(candidates):
        t0 = time.time()
        m = XGBRegressor(**params, random_state=7, n_jobs=2, tree_method="hist")
        m.fit(X_tune, y_tune)
        mae = float(mean_absolute_error(y_val, m.predict(X_val)))
        _log(f"  cand {ci+1}/6 MAE={mae:.1f} t={time.time()-t0:.1f}s {params}")
        if mae < best[0]:
            best = (mae, params)
        del m; gc.collect()

    best_params = best[1]
    _log(f"[xgb] best holdout params: {best_params} (MAE={best[0]:.1f})")

    _log("[xgb] 5-fold CV on tuned XGBoost...")
    mae_mean, mae_std = cv_mae(
        lambda: XGBRegressor(**best_params, random_state=7, n_jobs=2,
                             tree_method="hist"),
        "XGBt")

    xgb_tuned = XGBRegressor(**best_params, random_state=7, n_jobs=2,
                             tree_method="hist")
    xgb_tuned.fit(X_tr, y_tr)
    y_pred = xgb_tuned.predict(X_te)
    results.append({
        "model": "XGBoost (tuned)",
        "cv_mae_mean": mae_mean, "cv_mae_std": mae_std,
        "test_mae":   float(mean_absolute_error(y_te, y_pred)),
        "test_rmse":  float(np.sqrt(mean_squared_error(y_te, y_pred))),
        "test_r2":    float(r2_score(y_te, y_pred)),
        "best_params": best_params,
    })
    _log(f"  XGB tuned   MAE={mae_mean:.1f} +/-{mae_std:.1f}")

    with open(model_file, "wb") as f:
        pickle.dump(xgb_tuned, f)
    json.dump(results, open(ckpt, "w"), indent=2, default=str)
    _log(f"[xgb] saved {model_file.name}")
    return {"results": results, "xgb_tuned_path": model_file}


# ----------------------------- Stage 4 --------------------------------------

def stage_shap(state: dict, force: bool = False) -> dict:
    """Feature importance via XGBoost's native pred_contribs (SHAP).

    Using the booster directly avoids the shap-library <-> xgboost 3.x
    base_score serialization mismatch.
    """
    ckpt = FIG / "shap_summary.png"
    if ckpt.exists() and (OUT / "top_features.csv").exists() and not force:
        _log("[shap] cached; skipping")
        return {}

    import matplotlib.pyplot as plt
    import xgboost as xgb_lib

    xgb_tuned = pickle.load(open(MODELS / "trained_xgb_tuned.pkl", "rb"))
    feature_cols = json.load(open(MODELS / "feature_columns.json"))
    split = np.load(CKPT / "split.npz")
    X_tr = split["X_tr"]

    _log("[shap] computing SHAP via booster pred_contribs on 500 rows...")
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(X_tr), size=min(500, len(X_tr)), replace=False)
    X_sample = X_tr[sample_idx]
    # XGBoost's DMatrix rejects feature names with '<', '[', ']', '>'.
    # Swap them for underscores just for the explainer's feature labels.
    safe_names = [c.replace("<", "_lt_").replace(">", "_gt_")
                   .replace("[", "").replace("]", "")
                  for c in feature_cols]
    booster = xgb_tuned.get_booster()
    booster.feature_names = safe_names
    dmat = xgb_lib.DMatrix(X_sample, feature_names=safe_names)
    contribs = booster.predict(dmat, pred_contribs=True)
    # pred_contribs returns shape (n_samples, n_features + 1); last col = bias
    shap_vals = contribs[:, :-1]
    _log(f"[shap] shap_vals shape={shap_vals.shape}")

    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_all = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
    top_all = top_all.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    top = top_all.head(15)
    top.to_csv(OUT / "top_features.csv", index=False)
    _log("[shap] Top 15 features:")
    print(top.to_string(index=False))

    # --- Bar plot (feature importance) -----------------------------------
    top20 = top_all.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(top20["feature"], top20["mean_abs_shap"], color="#2a9d8f")
    ax.set_xlabel("Mean |SHAP value|  (avg impact on model output)")
    ax.set_title("XGBoost feature importance (top 20)")
    fig.savefig(FIG / "shap_feature_importance.png", bbox_inches="tight")
    plt.close(fig)

    # --- Beeswarm-style summary (value vs SHAP, colored by feature val) --
    top20_names = top_all.head(20)["feature"].tolist()
    top20_idx = [feature_cols.index(n) for n in top20_names]
    fig, ax = plt.subplots(figsize=(9, 8))
    y_positions = np.arange(len(top20_names))
    for yi, fi in zip(y_positions, top20_idx):
        vals = X_sample[:, fi]
        # rescale feature value to [0,1] for color
        v_min, v_max = np.nanmin(vals), np.nanmax(vals)
        v_norm = (vals - v_min) / (v_max - v_min + 1e-9)
        sc = ax.scatter(shap_vals[:, fi],
                        np.full_like(shap_vals[:, fi], yi, dtype=float)
                        + (np.random.default_rng(fi).random(len(vals)) - 0.5) * 0.6,
                        c=v_norm, cmap="RdBu_r", s=14, alpha=0.75,
                        edgecolors="none")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(top20_names)
    ax.invert_yaxis()
    ax.axvline(0, color="#888", linewidth=0.8)
    ax.set_xlabel("SHAP value  (impact on predicted yield, kg/ha)")
    ax.set_title("SHAP summary - top 20 features (color = feature value low->high)")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label("Feature value (normalized)")
    fig.savefig(FIG / "shap_summary.png", bbox_inches="tight")
    plt.close(fig)

    # Residuals with tuned model
    from sklearn.metrics import mean_absolute_error
    X_te, y_te = split["X_te"], split["y_te"]
    y_pred = xgb_tuned.predict(X_te)
    resid = y_te - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y_pred, resid, alpha=0.25, color="#264653", s=12)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted yield (kg/ha)")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"Residuals vs predicted  (test MAE={mean_absolute_error(y_te, y_pred):.0f})")
    axes[1].hist(resid, bins=50, color="#2a9d8f", alpha=0.85)
    axes[1].set_xlabel("Residual (kg/ha)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual distribution")
    fig.savefig(FIG / "residuals.png", bbox_inches="tight")
    plt.close(fig)
    return {}


# ----------------------------- Stage 5 --------------------------------------

def stage_predict(state: dict, force: bool = False) -> dict:
    ckpt = OUT / "predictions_2050.csv"
    if ckpt.exists() and (OUT / "yield_change_summary.csv").exists() and not force:
        _log("[predict] cached; skipping")
        return {}

    xgb_tuned = pickle.load(open(MODELS / "trained_xgb_tuned.pkl", "rb"))
    feature_cols = json.load(open(MODELS / "feature_columns.json"))
    df_2050 = pd.read_pickle(CKPT / "proj2050.pkl")
    merged = pd.read_pickle(CKPT / "merged.pkl")

    _log("[predict] projecting 2050 yields under RCP 8.5...")
    crops = [c.split("Crop_")[1] for c in feature_cols if c.startswith("Crop_")]
    pred_rows = []
    for _, crow in df_2050.iterrows():
        base = crow.drop(["Year", "County", "Country"],
                         errors="ignore").to_dict()
        for crop in crops:
            row = dict(base)
            for c in crops:
                row[f"Crop_{c}"] = 1 if c == crop else 0
            arr = np.array([row.get(col, 0) for col in feature_cols],
                           dtype=float).reshape(1, -1)
            yhat = float(xgb_tuned.predict(arr)[0])
            pred_rows.append({"County": crow["County"], "Crop": crop,
                              "Year": 2050, "yield (kg/ha)": round(yhat, 1)})
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(ckpt, index=False)
    _log(f"[predict] saved {len(pred_df)} 2050 predictions")

    ref = merged[merged["Year"] == 2009].copy()
    crop_cols = [c for c in merged.columns if c.startswith("Crop_")]
    ref["Crop"] = ref[crop_cols].idxmax(axis=1).str.replace("Crop_", "")
    ref_means = ref.groupby("Crop")["yield (kg/ha)"].mean().reset_index()
    ref_means.columns = ["Crop", "yield_2009"]
    pred_means = pred_df.groupby("Crop")["yield (kg/ha)"].mean().reset_index()
    pred_means.columns = ["Crop", "yield_2050"]
    summary = pred_means.merge(ref_means, on="Crop", how="left")
    summary["pct_change_vs_2009"] = (
        (summary["yield_2050"] - summary["yield_2009"])
        / summary["yield_2009"] * 100
    ).round(2)
    summary = summary.sort_values("pct_change_vs_2009")
    summary.to_csv(OUT / "yield_change_summary.csv", index=False)
    _log("[predict] 2050 vs 2009 yield change by crop:")
    print(summary.to_string(index=False))

    # Per-county per-crop matrix for choropleth step
    pivot = (pred_df.merge(ref[["County", "Crop", "yield (kg/ha)"]]
                           .rename(columns={"yield (kg/ha)": "yield_2009"}),
                           on=["County", "Crop"], how="left"))
    pivot["pct_change"] = ((pivot["yield (kg/ha)"] - pivot["yield_2009"])
                           / pivot["yield_2009"] * 100).round(2)
    pivot.to_csv(OUT / "predictions_2050_by_county.csv", index=False)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d62828" if v < 0 else "#2a9d8f"
              for v in summary["pct_change_vs_2009"]]
    ax.bar(summary["Crop"], summary["pct_change_vs_2009"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("% change vs 2009")
    ax.set_title("Projected 2050 crop yield change in Sweden (RCP 8.5)")
    plt.xticks(rotation=45, ha="right")
    fig.savefig(FIG / "yield_change_2050.png", bbox_inches="tight")
    plt.close(fig)
    return {}


# ----------------------------- Stage 6 --------------------------------------

def stage_report(state: dict, force: bool = False) -> dict:
    results = []
    for ck in ("baselines.json", "xgb.json"):
        p = CKPT / ck
        if p.exists():
            results.extend(json.load(open(p)))

    if not results:
        _log("[report] nothing to report")
        return {}

    res_df = pd.DataFrame(results)[[
        "model", "cv_mae_mean", "cv_mae_std", "test_mae", "test_rmse", "test_r2"
    ]].round(2)
    res_df.to_csv(OUT / "model_comparison.csv", index=False)
    json.dump(results, open(OUT / "model_comparison.json", "w"),
              indent=2, default=str)
    print("\nModel comparison:")
    print(res_df.to_string(index=False))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#ccc", "#bbb", "#88bbdd", "#2a9d8f"][:len(res_df)]
    ax.barh(res_df["model"], res_df["cv_mae_mean"],
            xerr=res_df["cv_mae_std"], color=colors)
    ax.set_xlabel("5-fold CV MAE (kg/ha)")
    ax.set_title("Model comparison (lower is better)")
    for i, v in enumerate(res_df["cv_mae_mean"]):
        ax.text(v + 5, i, f"{v:.0f}", va="center")
    fig.savefig(FIG / "model_comparison.png", bbox_inches="tight")
    plt.close(fig)
    _log("[report] wrote model_comparison.{csv,json,png}")
    return {}


# ----------------------------- Main -----------------------------------------

STAGES = {
    "prepare":   stage_prepare,
    "baselines": stage_baselines,
    "xgb":       stage_xgb,
    "shap":      stage_shap,
    "predict":   stage_predict,
    "report":    stage_report,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=list(STAGES), default=None,
                        help="Run only this stage")
    parser.add_argument("--force", action="store_true",
                        help="Ignore checkpoints")
    args = parser.parse_args()

    state = stage_prepare(force=args.force and (args.stage in (None, "prepare")))

    if args.stage is None:
        for name in ("baselines", "xgb", "shap", "predict", "report"):
            _log(f"=== {name} ===")
            STAGES[name](state, force=args.force)
    else:
        if args.stage != "prepare":
            STAGES[args.stage](state, force=args.force)

    _log("DONE")


if __name__ == "__main__":
    main()
