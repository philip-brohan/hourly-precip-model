#!/usr/bin/env python3
"""Quick inspection utility for 20CR hourly netCDF files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def first_data_var_name(ds: xr.Dataset) -> str:
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    for preferred in ("PRATE", "TMP2m", "PRMSL"):
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


def inspect_file(path: Path) -> None:
    if not path.exists():
        print(f"Missing file: {path}")
        return

    print("=" * 88)
    print(f"File: {path}")
    print("=" * 88)

    with xr.open_dataset(path) as ds:
        var_name = first_data_var_name(ds)
        var = ds[var_name]
        time_dim, lat_dim, lon_dim = detect_coord_names(ds)

        print("Dimensions:")
        for k, v in ds.dims.items():
            print(f"  {k}: {v}")

        print("Data variables:")
        for k, v in ds.data_vars.items():
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")

        print("Coordinates:")
        for k, v in ds.coords.items():
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")

        print(
            f"Detected dims: time={time_dim}, lat={lat_dim}, lon={lon_dim}, data_var={var_name}"
        )

        lat_vals = ds[lat_dim].values
        lon_vals = ds[lon_dim].values
        print(
            f"Lat range: {float(np.nanmin(lat_vals)):.3f} to {float(np.nanmax(lat_vals)):.3f}"
        )
        print(
            f"Lon range: {float(np.nanmin(lon_vals)):.3f} to {float(np.nanmax(lon_vals)):.3f}"
        )

        times = pd.to_datetime(ds[time_dim].values)
        if len(times) > 1:
            dt = times[1] - times[0]
            print(f"Time range: {times[0]} to {times[-1]} ({len(times)} steps)")
            print(f"Time increment: {dt}")
        else:
            print("Time coordinate has fewer than 2 elements")

        vals = var.isel({time_dim: 0}).values
        print(
            "First-time-slice stats: "
            f"min={float(np.nanmin(vals)):.6g}, "
            f"max={float(np.nanmax(vals)):.6g}, "
            f"mean={float(np.nanmean(vals)):.6g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect representative netCDF files")
    parser.add_argument(
        "--base-dir",
        default="/data/scratch/philip.brohan/MLP/20CR/version_3/hourly",
        help="Root directory containing yearly folders",
    )
    parser.add_argument("--year", type=int, default=2000, help="Year to inspect")
    parser.add_argument("--member", type=int, default=1, help="Ensemble member number")
    args = parser.parse_args()

    mem = f"mem{args.member:03d}"
    year = args.year
    base = Path(args.base_dir) / f"{year}"

    files = [
        base / f"PRMSL.{year}_{mem}.nc",
        base / f"TMP2m.{year}_{mem}.nc",
        base / f"PRATE.{year}_{mem}.nc",
    ]

    for f in files:
        inspect_file(f)


if __name__ == "__main__":
    main()
