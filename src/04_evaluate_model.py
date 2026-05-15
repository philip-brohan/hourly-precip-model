#!/usr/bin/env python3
"""Evaluate the trained precipitation model and create diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb


def load_test_sample(
    path: Path, max_rows: int | None, seed: int
) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        n = h5["X"].shape[0]
        if max_rows is None or max_rows >= n:
            idx = np.arange(n)
        else:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(n, size=max_rows, replace=False))

        out = {
            "X": h5["X"][idx, :].astype(np.float32),
            "y_log": h5["y_log"][idx].astype(np.float32),
            "y_raw": h5["y_raw"][idx].astype(np.float32),
            "ilat": h5["ilat"][idx].astype(np.int32),
            "ilon": h5["ilon"][idx].astype(np.int32),
            "lat": h5["lat"][idx].astype(np.float32),
            "lon": h5["lon"][idx].astype(np.float32),
            "month": h5["month"][idx].astype(np.uint8),
            "feature_names": np.array(
                json.loads(h5.attrs["feature_names_json"]), dtype=object
            ),
            "grid_lat": h5["grid_lat"][:].astype(np.float32),
            "grid_lon": h5["grid_lon"][:].astype(np.float32),
        }
    return out


def first_data_var_name(ds: xr.Dataset, preferred: str) -> str:
    if preferred in ds.data_vars:
        return preferred
    return next(iter(ds.data_vars))


def detect_coord_names(ds: xr.Dataset) -> tuple[str, str, str]:
    dims = list(ds.dims)
    time_dim = "time" if "time" in dims else dims[0]

    lat_candidates = ["lat", "latitude", "LAT", "Latitude"]
    lon_candidates = ["lon", "longitude", "LON", "Longitude"]

    lat_dim = next((x for x in lat_candidates if x in dims), None)
    lon_dim = next((x for x in lon_candidates if x in dims), None)

    remaining = [d for d in dims if d != time_dim]
    if lat_dim is None and len(remaining) >= 1:
        lat_dim = remaining[0]
    if lon_dim is None and len(remaining) >= 2:
        lon_dim = remaining[1]

    if lat_dim is None or lon_dim is None:
        raise RuntimeError(f"Could not detect lat/lon dims from {dims}")

    return time_dim, lat_dim, lon_dim


def parse_snapshot_time(spec: str) -> pd.Timestamp:
    """Accept YYYY-MM-DD:HH, YYYY-MM-DDTHH, or YYYY-MM-DD HH."""
    for fmt in ("%Y-%m-%d:%H", "%Y-%m-%dT%H", "%Y-%m-%d %H"):
        try:
            return pd.to_datetime(spec, format=fmt)
        except ValueError:
            continue
    raise ValueError(
        f"Invalid --snapshot-time '{spec}'. Use YYYY-MM-DD:HH (e.g. 1969-03-12:15)."
    )


def build_snapshot_fields(
    base_dir: Path,
    snapshot_time: pd.Timestamp,
    member: int,
    n_lags: int,
    classifier_threshold: float,
    model: xgb.Booster | None,
    classifier: xgb.Booster | None,
    regressor: xgb.Booster | None,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    year = int(snapshot_time.year)
    mem = f"mem{member:03d}"
    ydir = base_dir / f"{year}"
    f_prate = ydir / f"PRATE.{year}_{mem}.nc"
    f_tmp = ydir / f"TMP2m.{year}_{mem}.nc"
    f_prmsl = ydir / f"PRMSL.{year}_{mem}.nc"

    if not (f_prate.exists() and f_tmp.exists() and f_prmsl.exists()):
        raise FileNotFoundError(
            f"Missing snapshot inputs for {snapshot_time} member {mem} in {ydir}"
        )

    with xr.open_dataset(f_prate) as ds_prate, xr.open_dataset(f_tmp) as ds_tmp, xr.open_dataset(f_prmsl) as ds_prmsl:
        time_dim, lat_dim, lon_dim = detect_coord_names(ds_prate)

        v_prate = first_data_var_name(ds_prate, "PRATE")
        v_tmp = first_data_var_name(ds_tmp, "TMP2m")
        v_prmsl = first_data_var_name(ds_prmsl, "PRMSL")

        prate = ds_prate[v_prate].squeeze(drop=True).transpose(time_dim, lat_dim, lon_dim).values
        tmp = ds_tmp[v_tmp].squeeze(drop=True).transpose(time_dim, lat_dim, lon_dim).values
        prmsl = ds_prmsl[v_prmsl].squeeze(drop=True).transpose(time_dim, lat_dim, lon_dim).values

        times = pd.to_datetime(ds_prate[time_dim].values)
        target = pd.Timestamp(snapshot_time)
        idx = np.where(times == target)[0]
        if idx.size == 0:
            raise RuntimeError(
                f"Time {target} not found in {f_prate}. Available range {times.min()} to {times.max()}"
            )
        t = int(idx[0])
        if t < n_lags:
            raise RuntimeError(
                f"Requested time index {t} has insufficient history for n_lags={n_lags}"
            )

        lat_values = ds_prate[lat_dim].values.astype(np.float32)
        lon_values = ds_prate[lon_dim].values.astype(np.float32)

    nlat, nlon = prate.shape[1], prate.shape[2]
    ilat2d, ilon2d = np.indices((nlat, nlon))
    ilat = ilat2d.ravel()
    ilon = ilon2d.ravel()
    n_features = 2 * (n_lags + 1) + 6

    X = np.empty((ilat.shape[0], n_features), dtype=np.float32)
    for lag in range(0, n_lags + 1):
        X[:, lag] = tmp[t - lag, ilat, ilon]
        X[:, (n_lags + 1) + lag] = prmsl[t - lag, ilat, ilon]

    sin_doy = float(np.sin(2.0 * np.pi * (times[t].dayofyear / 365.25)))
    cos_doy = float(np.cos(2.0 * np.pi * (times[t].dayofyear / 365.25)))
    sin_hour = float(np.sin(2.0 * np.pi * (times[t].hour / 24.0)))
    cos_hour = float(np.cos(2.0 * np.pi * (times[t].hour / 24.0)))

    X[:, 2 * (n_lags + 1) + 0] = lat_values[ilat]
    X[:, 2 * (n_lags + 1) + 1] = lon_values[ilon]
    X[:, 2 * (n_lags + 1) + 2] = sin_doy
    X[:, 2 * (n_lags + 1) + 3] = cos_doy
    X[:, 2 * (n_lags + 1) + 4] = sin_hour
    X[:, 2 * (n_lags + 1) + 5] = cos_hour

    dgrid = xgb.DMatrix(X, feature_names=feature_names)

    if classifier is not None and regressor is not None:
        y_wet_prob = classifier.predict(dgrid)
        y_wet_pred = (y_wet_prob > classifier_threshold).astype(np.int32)
        y_pred_log_wet = regressor.predict(dgrid)
        y_pred_log = np.where(y_wet_pred == 1, y_pred_log_wet, 0.0)
    elif model is not None:
        y_pred_log = model.predict(dgrid)
    else:
        raise RuntimeError("No valid model configuration for snapshot prediction")

    y_pred = np.maximum(np.expm1(y_pred_log), 0.0).reshape(nlat, nlon)
    y_obs = np.maximum(prate[t, :, :].astype(np.float32), 0.0)
    return y_obs, y_pred, lat_values, lon_values


def save_snapshot_pair(
    y_obs: np.ndarray,
    y_pred: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    out: Path,
    snapshot_label: str,
) -> None:
    proj = ccrs.PlateCarree()

    def _edges(centres: np.ndarray) -> np.ndarray:
        d = np.diff(centres)
        left = centres[0] - d[0] / 2
        right = centres[-1] + d[-1] / 2
        return np.concatenate([[left], centres[:-1] + d / 2, [right]])

    lon_edges = _edges(grid_lon)
    lat_edges = _edges(grid_lat)
    lon2d, lat2d = np.meshgrid(lon_edges, lat_edges)

    vmax = float(np.nanpercentile(np.concatenate([y_obs.ravel(), y_pred.ravel()]), 99.5))
    vmax = max(vmax, 1e-9)

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 9), subplot_kw={"projection": proj}, constrained_layout=True
    )

    for ax, field, title in (
        (axes[0], y_obs, "Observed PRATE"),
        (axes[1], y_pred, "Model Predicted PRATE"),
    ):
        im = ax.pcolormesh(
            lon2d,
            lat2d,
            field,
            cmap="Blues",
            vmin=0.0,
            vmax=vmax,
            transform=proj,
            rasterized=True,
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="black")
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="grey")
        ax.set_global()
        ax.set_xticks(range(-180, 181, 60), crs=proj)
        ax.set_yticks(range(-90, 91, 30), crs=proj)
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                lambda v, _: f"{int(v)}\u00b0E" if v >= 0 else f"{int(-v)}\u00b0W"
            )
        )
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                lambda v, _: f"{int(v)}\u00b0N" if v >= 0 else f"{int(-v)}\u00b0S"
            )
        )
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=axes, orientation="vertical", shrink=0.9, pad=0.02)
    cbar.set_label("PRATE")
    fig.suptitle(f"Full-grid Snapshot: {snapshot_label}")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_scatter(y_true: np.ndarray, y_pred: np.ndarray, out: Path) -> None:
    """Scatter plot with broken axes skipping the empty region ~10^-11 to 10^-7."""
    zero_s = 1e-12    # sentinel added before log
    break_lo = 1e-11  # top of zero/dry cluster
    break_hi = 5e-7   # bottom of wet cluster

    vmax = max(float(np.nanpercentile(y_true, 99.9)),
               float(np.nanpercentile(y_pred, 99.9)),
               break_hi * 10)

    dry_lim = (zero_s * 0.4, break_lo * 6)
    wet_lim = (break_hi * 0.3, vmax * 3)

    xt = y_true.astype(np.float64) + zero_s
    xp = y_pred.astype(np.float64) + zero_s

    # Pre-compute a unified count range across all four hexbin panels
    hb_full = plt.hexbin(xt, xp, xscale="log", yscale="log", gridsize=50, mincnt=1)
    count_max = hb_full.get_array().max() if len(hb_full.get_array()) > 0 else 1
    plt.close()

    from matplotlib.colors import LogNorm
    norm = LogNorm(vmin=1, vmax=count_max)

    # 2×2 grid: columns = obs range (dry|wet), rows = pred range (wet|dry)
    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(2, 2,
                          width_ratios=[1, 5], height_ratios=[5, 1],
                          hspace=0.06, wspace=0.06)
    ax_dw = fig.add_subplot(gs[0, 0])  # dry obs, wet pred  (false alarms)
    ax_ww = fig.add_subplot(gs[0, 1])  # wet obs, wet pred  (hits)
    ax_dd = fig.add_subplot(gs[1, 0])  # dry obs, dry pred  (correct dry)
    ax_wd = fig.add_subplot(gs[1, 1])  # wet obs, dry pred  (misses)

    hb_kw = dict(gridsize=50, cmap="YlOrRd", mincnt=1, norm=norm)
    panels = [
        (ax_dw, dry_lim, wet_lim),
        (ax_ww, wet_lim, wet_lim),
        (ax_dd, dry_lim, dry_lim),
        (ax_wd, wet_lim, dry_lim),
    ]
    for ax, xlim, ylim in panels:
        ax.hexbin(xt, xp, xscale="log", yscale="log", **hb_kw)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # 1:1 reference line clipped to this quadrant
        lo = max(xlim[0], ylim[0])
        hi = min(xlim[1], ylim[1])
        if lo < hi:
            ref = np.logspace(np.log10(lo), np.log10(hi), 200)
            ax.plot(ref, ref, "k--", lw=1.2)

    # Hide inner tick marks / labels
    ax_dw.tick_params(bottom=False, labelbottom=False)
    ax_ww.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    ax_wd.tick_params(left=False, labelleft=False)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_dw, ax_ww, ax_dd, ax_wd],
                        fraction=0.02, pad=0.02, label="Count")

    # Axis labels
    ax_dd.set_xlabel("Observed PRATE", fontsize=10)
    ax_wd.set_xlabel("Observed PRATE", fontsize=10)
    ax_dd.set_ylabel("Predicted PRATE", fontsize=10)
    ax_dw.set_ylabel("Predicted PRATE", fontsize=10)

    # Quadrant annotations
    for ax, label in [
        (ax_ww, "hits"), (ax_dw, "false alarms"),
        (ax_wd, "misses"), (ax_dd, "correct dry"),
    ]:
        ax.text(0.03, 0.96, label, transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color="0.35")

    fig.suptitle("Observed vs Predicted (Test)", fontsize=13, y=0.995)
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()


def save_distribution(y_true: np.ndarray, y_pred: np.ndarray, out: Path) -> None:
    y_true_nz = y_true[y_true > 0.0]
    y_pred_nz = y_pred[y_pred > 0.0]

    plt.figure(figsize=(8, 6))

    if y_true_nz.size == 0 or y_pred_nz.size == 0:
        plt.text(
            0.5,
            0.5,
            "No non-zero precipitation values available",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
        plt.title("Precipitation Distribution (Non-zero only)")
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()
        return

    bins = np.logspace(
        np.log10(min(np.min(y_true_nz), np.min(y_pred_nz))),
        np.log10(max(np.max(y_true_nz), np.max(y_pred_nz), 1e-9)),
        80,
    )
    dens_obs, _ = np.histogram(y_true_nz, bins=bins, density=True)
    dens_pred, _ = np.histogram(y_pred_nz, bins=bins, density=True)
    x_centres = np.sqrt(bins[:-1] * bins[1:])

    # Cube-root transform expands low-density tails while compressing peak bins.
    plt.plot(x_centres, np.cbrt(dens_obs), lw=2.0, label="Observed")
    plt.plot(x_centres, np.cbrt(dens_pred), lw=2.0, label="Predicted")
    plt.xscale("log")
    plt.xlabel("PRATE")
    plt.ylabel("Cube-root density")
    plt.title("Precipitation Distribution (Non-zero only, cube-root density)")
    plt.legend()
    plt.grid(alpha=0.25, which="both")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def save_qq(y_true: np.ndarray, y_pred: np.ndarray, out: Path) -> None:
    """Q-Q plot with broken axes: dry cluster up to 10^-11, wet cluster from 5*10^-7."""
    zero_s = 1e-12
    break_lo = 1e-11
    break_hi = 5e-7

    p = np.linspace(0.001, 0.999, 500)
    q_true = np.quantile(y_true, p) + zero_s
    q_pred = np.quantile(y_pred, p) + zero_s

    vmax = max(float(q_true.max()), float(q_pred.max()), break_hi * 10)

    dry_lim = (zero_s * 0.4, break_lo * 6)
    wet_lim = (break_hi * 0.3, vmax * 3)

    fig = plt.figure(figsize=(9, 9))
    gs = fig.add_gridspec(2, 2,
                          width_ratios=[1, 5], height_ratios=[5, 1],
                          hspace=0.06, wspace=0.06)
    ax_dw = fig.add_subplot(gs[0, 0])
    ax_ww = fig.add_subplot(gs[0, 1])
    ax_dd = fig.add_subplot(gs[1, 0])
    ax_wd = fig.add_subplot(gs[1, 1])

    panels = [
        (ax_dw, dry_lim, wet_lim),
        (ax_ww, wet_lim, wet_lim),
        (ax_dd, dry_lim, dry_lim),
        (ax_wd, wet_lim, dry_lim),
    ]
    for ax, xlim, ylim in panels:
        ax.plot(q_true, q_pred, "o", ms=2.5, alpha=0.7, color="steelblue")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xscale("log")
        ax.set_yscale("log")
        lo = max(xlim[0], ylim[0])
        hi = min(xlim[1], ylim[1])
        if lo < hi:
            ref = np.logspace(np.log10(lo), np.log10(hi), 200)
            ax.plot(ref, ref, "k--", lw=1.2)

    ax_dw.tick_params(bottom=False, labelbottom=False)
    ax_ww.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    ax_wd.tick_params(left=False, labelleft=False)

    ax_dd.set_xlabel("Observed Quantiles", fontsize=10)
    ax_wd.set_xlabel("Observed Quantiles", fontsize=10)
    ax_dd.set_ylabel("Predicted Quantiles", fontsize=10)
    ax_dw.set_ylabel("Predicted Quantiles", fontsize=10)

    fig.suptitle("Q-Q Plot (Test)", fontsize=13, y=0.995)
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()


def save_monthly_cycle(
    month: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, out: Path
) -> None:
    df = pd.DataFrame({"month": month.astype(int), "obs": y_true, "pred": y_pred})
    grouped = df.groupby("month", as_index=True).mean().reindex(range(1, 13))

    plt.figure(figsize=(8, 4.8))
    plt.plot(grouped.index, grouped["obs"], "-o", label="Observed")
    plt.plot(grouped.index, grouped["pred"], "-o", label="Predicted")
    plt.xticks(range(1, 13))
    plt.xlabel("Month")
    plt.ylabel("Mean PRATE")
    plt.title("Seasonal Cycle")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def build_grid_maps(
    ilat: np.ndarray,
    ilon: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nlat: int,
    nlon: int,
    wet_threshold: float,
) -> dict[str, np.ndarray]:
    flat = ilat * nlon + ilon
    ngrid = nlat * nlon

    ones = np.ones_like(y_true, dtype=np.float64)
    count = np.bincount(flat, weights=ones, minlength=ngrid)

    s_true = np.bincount(flat, weights=y_true, minlength=ngrid)
    s_pred = np.bincount(flat, weights=y_pred, minlength=ngrid)
    s_true2 = np.bincount(flat, weights=y_true * y_true, minlength=ngrid)
    s_pred2 = np.bincount(flat, weights=y_pred * y_pred, minlength=ngrid)
    s_tp = np.bincount(flat, weights=y_true * y_pred, minlength=ngrid)

    wet_obs = np.bincount(
        flat, weights=(y_true > wet_threshold).astype(np.float64), minlength=ngrid
    )
    wet_pred = np.bincount(
        flat, weights=(y_pred > wet_threshold).astype(np.float64), minlength=ngrid
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_true = s_true / count
        mean_pred = s_pred / count
        bias = mean_pred - mean_true

        cov = s_tp / count - mean_true * mean_pred
        var_true = s_true2 / count - mean_true * mean_true
        var_pred = s_pred2 / count - mean_pred * mean_pred
        corr = cov / np.sqrt(np.maximum(var_true * var_pred, 1e-20))

        freq_obs = wet_obs / count
        freq_pred = wet_pred / count

    mask = count == 0
    for arr in (bias, corr, freq_obs, freq_pred):
        arr[mask] = np.nan

    return {
        "bias": bias.reshape(nlat, nlon),
        "corr": corr.reshape(nlat, nlon),
        "freq_obs": freq_obs.reshape(nlat, nlon),
        "freq_pred": freq_pred.reshape(nlat, nlon),
    }


def save_map(
    field: np.ndarray,
    title: str,
    out: Path,
    cmap: str,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(14, 5.5), subplot_kw={"projection": proj})

    # pcolormesh needs cell-edge coordinates; derive them from cell centres
    def _edges(centres: np.ndarray) -> np.ndarray:
        d = np.diff(centres)
        left = centres[0] - d[0] / 2
        right = centres[-1] + d[-1] / 2
        edges = np.concatenate([[left], centres[:-1] + d / 2, [right]])
        return edges

    lon_edges = _edges(grid_lon)
    lat_edges = _edges(grid_lat)
    lon2d, lat2d = np.meshgrid(lon_edges, lat_edges)

    im = ax.pcolormesh(
        lon2d, lat2d, field,
        cmap=cmap, vmin=vmin, vmax=vmax,
        transform=proj, rasterized=True,
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="grey")
    ax.set_global()
    ax.set_xticks(range(-180, 181, 60), crs=proj)
    ax.set_yticks(range(-90, 91, 30), crs=proj)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{int(v)}\u00b0E" if v >= 0 else f"{int(-v)}\u00b0W"
    ))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{int(v)}\u00b0N" if v >= 0 else f"{int(-v)}\u00b0S"
    ))
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.03)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def save_feature_importance(
    model: xgb.Booster, feature_names: list[str], out: Path
) -> None:
    gain = model.get_score(importance_type="gain")
    vals = np.array([gain.get(name, 0.0) for name in feature_names], dtype=np.float64)
    order = np.argsort(vals)[::-1]

    top_k = min(20, len(feature_names))
    order = order[:top_k][::-1]

    plt.figure(figsize=(9, 6.2))
    plt.barh(np.array(feature_names)[order], vals[order])
    plt.xlabel("Gain")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate XGBoost precipitation model(s)"
    )
    parser.add_argument("--test-file", default="data/test.h5")
    parser.add_argument("--model-file", default="models/xgb_precip.json")
    parser.add_argument(
        "--classifier-file",
        default=None,
        help="Optional classifier model for rain/no-rain",
    )
    parser.add_argument(
        "--regressor-file",
        default=None,
        help="Optional regressor model for amount (wet days only)",
    )
    parser.add_argument("--fig-dir", default="figures")
    parser.add_argument("--max-test-rows", type=int, default=2_000_000)
    parser.add_argument("--wet-threshold", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--base-dir",
        default="/data/scratch/philip.brohan/MLP/20CR/version_3/hourly",
        help="Base directory containing year/member netCDF files for full-grid snapshot",
    )
    parser.add_argument(
        "--snapshot-time",
        default="1969-03-12:15",
        help="Snapshot time for full-grid diagnostic (YYYY-MM-DD:HH)",
    )
    parser.add_argument(
        "--snapshot-member",
        type=int,
        default=1,
        help="Member number for full-grid snapshot diagnostic",
    )
    parser.add_argument(
        "--lags",
        type=int,
        default=8,
        help="Number of lag steps used in feature construction",
    )
    parser.add_argument(
        "--classifier-threshold",
        type=float,
        default=0.5,
        help="Probability threshold for rain class",
    )
    args = parser.parse_args()

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    test = load_test_sample(Path(args.test_file), args.max_test_rows, args.seed)
    feature_names = test["feature_names"].tolist()

    dtest = xgb.DMatrix(test["X"], label=test["y_log"], feature_names=feature_names)

    # Two-model case: classifier + regressor
    if args.classifier_file and args.regressor_file:
        print("Using two-model system (classifier + wet-days regressor)")

        # Load classifier
        classifier = xgb.Booster()
        classifier.load_model(args.classifier_file)
        y_wet_prob = classifier.predict(dtest)
        y_wet_pred = (y_wet_prob > args.classifier_threshold).astype(int)

        # Load regressor (trained on wet days only)
        regressor = xgb.Booster()
        regressor.load_model(args.regressor_file)
        y_pred_log_wet = regressor.predict(dtest)

        # Combined prediction: 0 if classifier says dry, else regressor prediction
        y_pred_log = np.where(y_wet_pred == 1, y_pred_log_wet, np.log1p(0.0))
        y_pred_log = np.maximum(y_pred_log, np.log1p(0.0))

        y_true = test["y_raw"]
        y_pred = np.expm1(y_pred_log)
        y_pred = np.maximum(y_pred, 0.0)

        # Print classifier metrics
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )

        y_true_wet = (y_true > args.wet_threshold).astype(int)

        acc = accuracy_score(y_true_wet, y_wet_pred)
        prec = precision_score(y_true_wet, y_wet_pred, zero_division=0)
        rec = recall_score(y_true_wet, y_wet_pred, zero_division=0)
        f1 = f1_score(y_true_wet, y_wet_pred, zero_division=0)
        auc = roc_auc_score(y_true_wet, y_wet_prob)

        print(f"\nClassifier metrics (rain/no-rain):")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1 Score  : {f1:.4f}")
        print(f"  ROC AUC   : {auc:.4f}")

        # Print regressor metrics on wet days only
        wet_mask = y_true_wet == 1
        if np.sum(wet_mask) > 0:
            y_true_wet_only = y_true[wet_mask]
            y_pred_wet_only = y_pred[wet_mask]
            rmse_wet = float(np.sqrt(np.mean((y_pred_wet_only - y_true_wet_only) ** 2)))
            mae_wet = float(np.mean(np.abs(y_pred_wet_only - y_true_wet_only)))
            print(f"\nRegressor metrics (wet days only, n={np.sum(wet_mask)}):")
            print(f"  RMSE: {rmse_wet:.6e}")
            print(f"  MAE : {mae_wet:.6e}")

        # Use classifier model for feature importance
        model = classifier

    else:
        # Single-model case (original behavior)
        print("Using single-model regressor")
        model = xgb.Booster()
        model.load_model(args.model_file)

        y_pred_log = model.predict(dtest)
        y_true = test["y_raw"]
        y_pred = np.expm1(y_pred_log)

        y_pred = np.maximum(y_pred, 0.0)

    save_scatter(y_true, y_pred, fig_dir / "scatter_obs_vs_pred.png")
    save_distribution(y_true, y_pred, fig_dir / "distribution_obs_vs_pred.png")
    save_qq(y_true, y_pred, fig_dir / "qq_plot.png")
    save_monthly_cycle(test["month"], y_true, y_pred, fig_dir / "monthly_cycle.png")

    nlat = test["grid_lat"].shape[0]
    nlon = test["grid_lon"].shape[0]
    maps = build_grid_maps(
        test["ilat"],
        test["ilon"],
        y_true,
        y_pred,
        nlat=nlat,
        nlon=nlon,
        wet_threshold=args.wet_threshold,
    )

    b = maps["bias"]
    c = maps["corr"]
    fo = maps["freq_obs"]
    fp = maps["freq_pred"]

    b_abs = np.nanpercentile(np.abs(b), 98)
    map_kwargs = {"grid_lat": test["grid_lat"], "grid_lon": test["grid_lon"]}
    save_map(
        b,
        "Bias map: mean(pred - obs)",
        fig_dir / "map_bias.png",
        cmap="RdBu_r",
        vmin=-b_abs,
        vmax=b_abs,
        **map_kwargs,
    )
    save_map(
        c,
        "Correlation map",
        fig_dir / "map_correlation.png",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        **map_kwargs,
    )
    save_map(
        fo,
        "Wet frequency observed",
        fig_dir / "map_wet_frequency_observed.png",
        cmap="Blues",
        vmin=0,
        vmax=1,
        **map_kwargs,
    )
    save_map(
        fp,
        "Wet frequency predicted",
        fig_dir / "map_wet_frequency_predicted.png",
        cmap="Blues",
        vmin=0,
        vmax=1,
        **map_kwargs,
    )
    save_map(
        fp - fo,
        "Wet frequency difference (pred - obs)",
        fig_dir / "map_wet_frequency_diff.png",
        cmap="RdBu_r",
        vmin=-0.5,
        vmax=0.5,
        **map_kwargs,
    )

    save_feature_importance(model, feature_names, fig_dir / "feature_importance.png")

    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))

    summary = {
        "n_test_rows": int(y_true.shape[0]),
        "rmse_raw": rmse,
        "mae_raw": mae,
        "wet_threshold": args.wet_threshold,
    }
    summary_path = fig_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    snapshot_time = parse_snapshot_time(args.snapshot_time)
    y_obs_snap, y_pred_snap, lat_snap, lon_snap = build_snapshot_fields(
        base_dir=Path(args.base_dir),
        snapshot_time=snapshot_time,
        member=args.snapshot_member,
        n_lags=args.lags,
        classifier_threshold=args.classifier_threshold,
        model=(None if args.classifier_file and args.regressor_file else model),
        classifier=(model if args.classifier_file and args.regressor_file else None),
        regressor=(regressor if args.classifier_file and args.regressor_file else None),
        feature_names=feature_names,
    )
    save_snapshot_pair(
        y_obs=y_obs_snap,
        y_pred=y_pred_snap,
        grid_lat=lat_snap,
        grid_lon=lon_snap,
        out=fig_dir / "snapshot_observed_vs_predicted.png",
        snapshot_label=f"{snapshot_time.strftime('%Y-%m-%d %H:00')} (mem{args.snapshot_member:03d})",
    )
    print(
        "Saved full-grid snapshot diagnostic to",
        fig_dir / "snapshot_observed_vs_predicted.png",
    )

    print("Saved diagnostics to", fig_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
