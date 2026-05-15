#!/usr/bin/env python3
"""Build sampled lagged training datasets from 20CR netCDF files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


def parse_year_range(spec: str) -> list[int]:
    start_s, end_s = spec.split(":", maxsplit=1)
    start = int(start_s)
    end = int(end_s)
    if end < start:
        raise ValueError(f"Invalid year range: {spec}")
    return list(range(start, end + 1))


def parse_member_spec(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-", maxsplit=1)
        start = int(a)
        end = int(b)
        if end < start:
            raise ValueError(f"Invalid member spec: {spec}")
        return list(range(start, end + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def first_data_var_name(ds: xr.Dataset, preferred: str) -> str:
    if preferred in ds.data_vars:
        return preferred
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
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


def init_h5(
    path: Path,
    n_features: int,
    feature_names: list[str],
    lat_values: np.ndarray,
    lon_values: np.ndarray,
) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    h5 = h5py.File(path, "w")
    h5.create_dataset(
        "X",
        shape=(0, n_features),
        maxshape=(None, n_features),
        dtype="f4",
        chunks=(8192, n_features),
    )
    h5.create_dataset(
        "y_log", shape=(0,), maxshape=(None,), dtype="f4", chunks=(65536,)
    )
    h5.create_dataset(
        "y_raw", shape=(0,), maxshape=(None,), dtype="f4", chunks=(65536,)
    )
    h5.create_dataset(
        "y_wet", shape=(0,), maxshape=(None,), dtype="u1", chunks=(65536,)
    )
    h5.create_dataset("ilat", shape=(0,), maxshape=(None,), dtype="i4", chunks=(65536,))
    h5.create_dataset("ilon", shape=(0,), maxshape=(None,), dtype="i4", chunks=(65536,))
    h5.create_dataset("lat", shape=(0,), maxshape=(None,), dtype="f4", chunks=(65536,))
    h5.create_dataset("lon", shape=(0,), maxshape=(None,), dtype="f4", chunks=(65536,))
    h5.create_dataset(
        "month", shape=(0,), maxshape=(None,), dtype="u1", chunks=(65536,)
    )
    h5.create_dataset("hour", shape=(0,), maxshape=(None,), dtype="u1", chunks=(65536,))
    h5.create_dataset(
        "time_ns", shape=(0,), maxshape=(None,), dtype="i8", chunks=(65536,)
    )

    h5.attrs["feature_names_json"] = json.dumps(feature_names)
    h5.create_dataset("grid_lat", data=lat_values.astype("f4"))
    h5.create_dataset("grid_lon", data=lon_values.astype("f4"))
    return h5


def append_1d(ds: h5py.Dataset, values: np.ndarray) -> None:
    old = ds.shape[0]
    new = old + values.shape[0]
    ds.resize((new,))
    ds[old:new] = values


def append_rows(h5: h5py.File, block: dict[str, np.ndarray]) -> None:
    n_new = block["X"].shape[0]
    old = h5["X"].shape[0]
    new = old + n_new
    h5["X"].resize((new, h5["X"].shape[1]))
    h5["X"][old:new, :] = block["X"]

    for name in (
        "y_log",
        "y_raw",
        "y_wet",
        "ilat",
        "ilon",
        "lat",
        "lon",
        "month",
        "hour",
        "time_ns",
    ):
        append_1d(h5[name], block[name])


def make_feature_names(n_lags: int) -> list[str]:
    names: list[str] = []
    for lag in range(0, n_lags + 1):
        names.append(f"TMP2m_lag{lag}")
    for lag in range(0, n_lags + 1):
        names.append(f"PRMSL_lag{lag}")
    names += ["lat", "lon", "sin_doy", "cos_doy", "sin_hour", "cos_hour"]
    return names


def build_split(
    split_name: str,
    years: list[int],
    members: list[int],
    base_dir: Path,
    out_file: Path,
    n_lags: int,
    samples_per_timestep: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)

    n_features = 2 * (n_lags + 1) + 6
    feature_names = make_feature_names(n_lags)

    h5: h5py.File | None = None
    total_rows = 0

    for year in years:
        for member in members:
            mem = f"mem{member:03d}"
            ydir = base_dir / f"{year}"
            f_prate = ydir / f"PRATE.{year}_{mem}.nc"
            f_tmp = ydir / f"TMP2m.{year}_{mem}.nc"
            f_prmsl = ydir / f"PRMSL.{year}_{mem}.nc"

            if not (f_prate.exists() and f_tmp.exists() and f_prmsl.exists()):
                print(f"Skipping missing set: {year} {mem}")
                continue

            with xr.open_dataset(f_prate) as ds_prate, xr.open_dataset(
                f_tmp
            ) as ds_tmp, xr.open_dataset(f_prmsl) as ds_prmsl:
                time_dim, lat_dim, lon_dim = detect_coord_names(ds_prate)

                v_prate = first_data_var_name(ds_prate, "PRATE")
                v_tmp = first_data_var_name(ds_tmp, "TMP2m")
                v_prmsl = first_data_var_name(ds_prmsl, "PRMSL")

                prate = (
                    ds_prate[v_prate]
                    .squeeze(drop=True)
                    .transpose(time_dim, lat_dim, lon_dim)
                    .values
                )
                tmp = (
                    ds_tmp[v_tmp]
                    .squeeze(drop=True)
                    .transpose(time_dim, lat_dim, lon_dim)
                    .values
                )
                prmsl = (
                    ds_prmsl[v_prmsl]
                    .squeeze(drop=True)
                    .transpose(time_dim, lat_dim, lon_dim)
                    .values
                )

                if prate.shape != tmp.shape or prate.shape != prmsl.shape:
                    raise RuntimeError(
                        f"Shape mismatch for {year} {mem}: PRATE {prate.shape}, TMP2m {tmp.shape}, PRMSL {prmsl.shape}"
                    )

                times = pd.to_datetime(ds_prate[time_dim].values)
                lat_values = ds_prate[lat_dim].values.astype(np.float32)
                lon_values = ds_prate[lon_dim].values.astype(np.float32)

                if h5 is None:
                    h5 = init_h5(
                        out_file, n_features, feature_names, lat_values, lon_values
                    )

                nt, nlat, nlon = prate.shape
                n_grid = nlat * nlon

                sin_doy = np.sin(
                    2.0 * np.pi * (times.dayofyear.values / 365.25)
                ).astype(np.float32)
                cos_doy = np.cos(
                    2.0 * np.pi * (times.dayofyear.values / 365.25)
                ).astype(np.float32)
                sin_hour = np.sin(2.0 * np.pi * (times.hour.values / 24.0)).astype(
                    np.float32
                )
                cos_hour = np.cos(2.0 * np.pi * (times.hour.values / 24.0)).astype(
                    np.float32
                )

                iterator = tqdm(
                    range(n_lags, nt),
                    desc=f"{split_name}: {year} {mem}",
                    leave=False,
                )
                for t in iterator:
                    n_pick = min(samples_per_timestep, n_grid)
                    idx = rng.choice(n_grid, size=n_pick, replace=False)
                    ilat = idx // nlon
                    ilon = idx % nlon

                    X = np.empty((n_pick, n_features), dtype=np.float32)

                    for lag in range(0, n_lags + 1):
                        X[:, lag] = tmp[t - lag, ilat, ilon]
                        X[:, (n_lags + 1) + lag] = prmsl[t - lag, ilat, ilon]

                    X[:, 2 * (n_lags + 1) + 0] = lat_values[ilat]
                    X[:, 2 * (n_lags + 1) + 1] = lon_values[ilon]
                    X[:, 2 * (n_lags + 1) + 2] = sin_doy[t]
                    X[:, 2 * (n_lags + 1) + 3] = cos_doy[t]
                    X[:, 2 * (n_lags + 1) + 4] = sin_hour[t]
                    X[:, 2 * (n_lags + 1) + 5] = cos_hour[t]

                    y_raw = np.maximum(prate[t, ilat, ilon].astype(np.float32), 0.0)
                    y_log = np.log1p(y_raw)
                    y_wet = (y_raw > 1e-6).astype(np.uint8)

                    good = np.isfinite(y_log)
                    good &= np.all(np.isfinite(X), axis=1)

                    if not np.any(good):
                        continue

                    block = {
                        "X": X[good],
                        "y_log": y_log[good].astype(np.float32),
                        "y_raw": y_raw[good].astype(np.float32),
                        "y_wet": y_wet[good].astype(np.uint8),
                        "ilat": ilat[good].astype(np.int32),
                        "ilon": ilon[good].astype(np.int32),
                        "lat": lat_values[ilat[good]].astype(np.float32),
                        "lon": lon_values[ilon[good]].astype(np.float32),
                        "month": np.full(np.sum(good), times[t].month, dtype=np.uint8),
                        "hour": np.full(np.sum(good), times[t].hour, dtype=np.uint8),
                        "time_ns": np.full(
                            np.sum(good), times[t].value, dtype=np.int64
                        ),
                    }
                    append_rows(h5, block)
                    total_rows += block["X"].shape[0]

                print(
                    f"Completed year/member {year}/{mem} for {split_name}. "
                    f"Accumulated rows: {total_rows:,}"
                )

    if h5 is None:
        raise RuntimeError(
            f"No data written for split {split_name}; check paths and ranges"
        )

    n_rows = int(h5["X"].shape[0])
    n_features = int(h5["X"].shape[1])
    h5.attrs["n_rows"] = n_rows
    h5.attrs["n_features"] = n_features
    h5.close()
    print(f"Wrote {n_rows:,} rows with {n_features} features to {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create train/val/test sampled HDF5 datasets"
    )
    parser.add_argument(
        "--base-dir",
        default="/data/scratch/philip.brohan/MLP/20CR/version_3/hourly",
        help="Root data directory",
    )
    parser.add_argument(
        "--out-dir", default="data", help="Output directory for HDF5 files"
    )
    parser.add_argument("--train-years", default="1961:1990")
    parser.add_argument("--val-years", default="1991:2000")
    parser.add_argument("--test-years", default="2001:2014")
    parser.add_argument(
        "--members", default="1-5", help="Member list, e.g. 1-5 or 1,2,3"
    )
    parser.add_argument("--lags", type=int, default=8, help="Number of lag steps")
    parser.add_argument(
        "--samples-per-timestep",
        type=int,
        default=200,
        help="Random spatial samples per timestep",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_years = parse_year_range(args.train_years)
    val_years = parse_year_range(args.val_years)
    test_years = parse_year_range(args.test_years)
    members = parse_member_spec(args.members)

    print("Building sampled datasets with settings:")
    print(f"  base_dir={base_dir}")
    print(f"  train_years={train_years[0]}..{train_years[-1]}")
    print(f"  val_years={val_years[0]}..{val_years[-1]}")
    print(f"  test_years={test_years[0]}..{test_years[-1]}")
    print(f"  members={members}")
    print(f"  lags={args.lags}, samples_per_timestep={args.samples_per_timestep}")

    build_split(
        split_name="train",
        years=train_years,
        members=members,
        base_dir=base_dir,
        out_file=out_dir / "train.h5",
        n_lags=args.lags,
        samples_per_timestep=args.samples_per_timestep,
        seed=args.seed,
    )
    build_split(
        split_name="val",
        years=val_years,
        members=members,
        base_dir=base_dir,
        out_file=out_dir / "val.h5",
        n_lags=args.lags,
        samples_per_timestep=args.samples_per_timestep,
        seed=args.seed + 1,
    )
    build_split(
        split_name="test",
        years=test_years,
        members=members,
        base_dir=base_dir,
        out_file=out_dir / "test.h5",
        n_lags=args.lags,
        samples_per_timestep=args.samples_per_timestep,
        seed=args.seed + 2,
    )


if __name__ == "__main__":
    main()
