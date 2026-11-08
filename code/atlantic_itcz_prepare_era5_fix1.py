#!/usr/bin/env python3
"""Prepare daily, monthly, and seasonal ERA5 fields for the Atlantic ITCZ analysis.

Stage 1 reads DKRZ P1D GRIB products for 1979-2024 and bins native values to
a 1-degree grid covering 65 W-10 W and 25 S-35 N. Grid centers are at integer
degrees. The land-sea mask defines ocean cells as lsm < 0.5; ocean fraction
(1-lsm) supplies weights for subsequent analyses. Atmospheric fields remain
unmasked so each stage can apply the same ocean mask.

Outputs under --outdir are daily/era5_daily_YYYY.nc, era5_monthly.nc,
era5_seasonal.nc, static_fields.nc, and run_metadata.json. Monthly means have
dimensions year, month, lat, lon. Seasonal means are weighted by daily sample
counts with a separate finite-data denominator for each variable. DJF is
labeled by the January-February year.

Example for one year:
    python -B code/atlantic_itcz_prepare_era5_fix1.py --era5-root /path/to/ERA5/E5 \
        --years 2000 2000 --outdir /path/to/new/stage1

See README.md for the full-period settings and required archive layout.
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from atlantic_itcz_common import (
    SCHEMA_VERSION,
    bin_average_to_regular_grid,
    canonical_name,
    compute_lts,
    ensure_dir,
    expanded_source_extent,
    load_era5_year_param,
    sha256_file,
    standardize_units,
    target_grid,
    write_json,
    write_netcdf,
)

SCRIPT_VERSION = "1.0.1"
DEFAULT_ROOT = "/pool/data/ERA5/E5"
DEFAULT_OUTPUT = "out_atlantic_itcz/01_prepare"
DEFAULT_PARAMS = ("tcwv", "tp", "sst", "10u", "10v", "2t", "blh", "msl")
SEASONS = ("DJF", "MAM", "JJA", "SON")
SEASON_MONTHS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--era5-root", default=DEFAULT_ROOT)
    p.add_argument("--outdir", default=DEFAULT_OUTPUT)
    p.add_argument("--years", nargs=2, type=int, default=[1979, 2024], metavar=("START", "END"))
    p.add_argument("--lon-range", nargs=2, type=float, default=[-65.0, -10.0])
    p.add_argument("--lat-range", nargs=2, type=float, default=[-25.0, 35.0])
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--lsm-threshold", type=float, default=0.5)
    p.add_argument(
        "--params",
        nargs="+",
        default=list(DEFAULT_PARAMS),
        help="Surface/single-level P1D parameters. LSM and pressure-level temperature are handled separately.",
    )
    p.add_argument("--skip-lts", action="store_true", help="Do not load 700/1000-hPa temperature or compute LTS.")
    p.add_argument("--allow-incomplete-days", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-recursive-search", action="store_true")
    return p.parse_args()


def expected_days(year: int) -> int:
    return 366 if calendar.isleap(year) else 365


def assert_daily_axis(ds: xr.Dataset, year: int, allow_incomplete: bool) -> None:
    if "time" not in ds.dims:
        raise ValueError(f"Daily dataset for {year} has no time dimension")
    times = pd.DatetimeIndex(ds.time.values)
    if times.has_duplicates:
        raise ValueError(f"Duplicate daily labels remain for {year}")
    if not times.is_monotonic_increasing:
        raise ValueError(f"Daily labels are not increasing for {year}")
    expected = expected_days(year)
    if len(times) != expected:
        message = f"Expected {expected} days for {year}, found {len(times)}"
        if allow_incomplete:
            print(f"[WARN] {message}")
        else:
            raise RuntimeError(message + ". Use --allow-incomplete-days only after checking the archive.")


def select_pressure_level(t: xr.DataArray, level_hpa: float) -> xr.DataArray:
    if "level" not in t.dims:
        raise ValueError("Pressure-level temperature does not contain a level dimension")
    levels = np.asarray(t.level.values, dtype=float)
    j = int(np.nanargmin(np.abs(levels - level_hpa)))
    if abs(levels[j] - level_hpa) > 1.0:
        raise ValueError(f"Requested {level_hpa:g} hPa, nearest available level is {levels[j]:g} hPa")
    return t.isel(level=j, drop=True)


def prepare_static_fields(args: argparse.Namespace, lon: np.ndarray, lat: np.ndarray, extent: list[float]):
    root = Path(args.era5_root)
    lsm_native, source_files = load_era5_year_param(
        root,
        "lsm",
        args.years[0],
        extent,
        allow_recursive=not args.no_recursive_search,
    )
    lsm = bin_average_to_regular_grid(lsm_native, lon, lat, args.resolution).clip(0.0, 1.0)
    lsm.name = "lsm"
    lsm.attrs.update(
        {
            "long_name": "land-sea mask fraction on canonical grid",
            "units": "1",
            "ocean_definition": f"lsm < {args.lsm_threshold:g}",
        }
    )
    ocean_fraction = (1.0 - lsm).clip(0.0, 1.0).astype("float32")
    ocean_fraction.name = "ocean_fraction"
    ocean_fraction.attrs.update(
        {
            "long_name": "ocean fraction used for weighted spatial means",
            "units": "1",
            "definition": "1 - regridded ERA5 land-sea mask",
        }
    )
    ocean_mask = (lsm < args.lsm_threshold)
    ocean_mask.name = "ocean_mask"
    ocean_mask.attrs.update(
        {
            "long_name": "fixed Atlantic ocean mask",
            "definition": f"regridded ERA5 lsm < {args.lsm_threshold:g}",
        }
    )
    ds = xr.Dataset({"lsm": lsm.astype("float32"), "ocean_fraction": ocean_fraction, "ocean_mask": ocean_mask})
    ds.attrs.update(
        {
            "schema_version": SCHEMA_VERSION,
            "canonical_grid_resolution_degrees": float(args.resolution),
            "canonical_lon_range": f"{args.lon_range[0]:g},{args.lon_range[1]:g}",
            "canonical_lat_range": f"{args.lat_range[0]:g},{args.lat_range[1]:g}",
            "ocean_mask_definition": f"lsm < {args.lsm_threshold:g}",
        }
    )
    return ds, source_files


def prepare_year(
    args: argparse.Namespace,
    year: int,
    lon: np.ndarray,
    lat: np.ndarray,
    extent: list[float],
) -> tuple[xr.Dataset, dict[str, list[str]]]:
    """Load, standardize and regrid all canonical daily fields for one year."""
    root = Path(args.era5_root)
    fields: dict[str, xr.DataArray] = {}
    provenance: dict[str, list[str]] = {}

    for param in args.params:
        print(f"[INFO] {year}: loading {param}")
        native, source_files = load_era5_year_param(
            root,
            param,
            year,
            extent,
            allow_recursive=not args.no_recursive_search,
        )
        name = canonical_name(param)
        native = standardize_units(native, name)
        regular = bin_average_to_regular_grid(native, lon, lat, args.resolution)
        regular.name = name
        fields[name] = regular
        provenance[name] = source_files

    if not args.skip_lts:
        print(f"[INFO] {year}: loading pressure-level temperature for LTS")
        t_native, source_files = load_era5_year_param(
            root,
            "t",
            year,
            extent,
            allow_recursive=not args.no_recursive_search,
        )
        t_native = standardize_units(t_native, "t")
        t700_native = select_pressure_level(t_native, 700.0)
        t1000_native = select_pressure_level(t_native, 1000.0)
        t700 = bin_average_to_regular_grid(t700_native, lon, lat, args.resolution)
        t1000 = bin_average_to_regular_grid(t1000_native, lon, lat, args.resolution)
        t700.name = "t700"
        t1000.name = "t1000"
        t700.attrs.update({"long_name": "700-hPa air temperature", "units": "K"})
        t1000.attrs.update({"long_name": "1000-hPa air temperature", "units": "K"})
        fields["t700"] = t700
        fields["t1000"] = t1000
        fields["lts"] = compute_lts(t700, t1000)
        provenance["t"] = source_files

    ds = xr.merge([da.rename(name) for name, da in fields.items()], join="exact", compat="override")
    ds = ds.sortby("time")
    assert_daily_axis(ds, year, args.allow_incomplete_days)
    ds.attrs.update(
        {
            "title": "Canonical daily ERA5 fields for Atlantic ITCZ analysis",
            "schema_version": SCHEMA_VERSION,
            "script_version": SCRIPT_VERSION,
            "source_archive": str(args.era5_root),
            "source_time_resolution": "P1D",
            "calendar_day_label": "00:00 timestamp assigned to each P1D calendar day",
            "canonical_grid_resolution_degrees": float(args.resolution),
            "canonical_lon_range": f"{args.lon_range[0]:g},{args.lon_range[1]:g}",
            "canonical_lat_range": f"{args.lat_range[0]:g},{args.lat_range[1]:g}",
            "ocean_mask_note": "Fields are retained unmasked; apply static_fields.nc:ocean_mask downstream.",
        }
    )
    return ds, provenance


def monthly_from_daily(ds: xr.Dataset, year: int) -> xr.Dataset:
    """Calendar-month means of daily quantities for one year."""
    monthly = ds.groupby("time.month").mean("time", skipna=True)
    monthly = monthly.reindex(month=np.arange(1, 13, dtype=int))
    monthly = monthly.expand_dims(year=[year])
    counts = ds["tcwv"].notnull().any(("lat", "lon")).groupby("time.month").sum("time")
    counts = counts.reindex(month=np.arange(1, 13, dtype=int)).astype("int16")
    monthly["n_daily_samples"] = counts.expand_dims(year=[year])
    monthly["n_daily_samples"].attrs.update(
        {"long_name": "number of daily samples contributing to the monthly mean", "units": "day"}
    )
    return monthly


def seasonal_from_monthly(monthly: xr.Dataset) -> xr.Dataset:
    """Construct day-weighted seasonal means, with DJF labeled by Jan-Feb year."""
    years = np.asarray(monthly.year.values, dtype=int)
    variables = [v for v in monthly.data_vars if v != "n_daily_samples"]
    template = monthly[variables].isel(year=0, month=0, drop=True)
    season_datasets: list[xr.Dataset] = []

    for season in SEASONS:
        rows: list[xr.Dataset] = []
        for year in years:
            pieces: list[xr.Dataset] = []
            weights: list[float] = []
            complete = True
            for month in SEASON_MONTHS[season]:
                source_year = year - 1 if season == "DJF" and month == 12 else year
                if source_year not in years:
                    complete = False
                    break
                count = monthly["n_daily_samples"].sel(year=source_year, month=month)
                nday = int(count.values)
                if nday <= 0:
                    complete = False
                    break
                pieces.append(monthly[variables].sel(year=source_year, month=month, drop=True))
                weights.append(float(nday))
            if not complete:
                row = template * np.nan
                n_total = 0
            else:
                stacked = xr.concat(pieces, dim="_month")
                w = xr.DataArray(np.asarray(weights, dtype=float), dims="_month")

                # Use a field-specific valid-data denominator rather than the
                # scalar sum of month weights.  This is essential for variables
                # such as SST that are undefined over land.  A simple
                # ``sum(skipna=True) / w.sum()`` would turn locations that are
                # NaN in every constituent month into zero, creating artificial
                # SST values over land in the seasonal product.
                valid_weight = xr.where(stacked.notnull(), w, 0.0).sum("_month")
                numerator = (stacked * w).sum("_month", skipna=True)
                row = (numerator / valid_weight).where(valid_weight > 0.0)
                n_total = int(sum(weights))
            row = row.expand_dims(year=[int(year)])
            row["n_daily_samples"] = xr.DataArray([n_total], dims="year", coords={"year": [int(year)]})
            row["season_complete"] = xr.DataArray([bool(complete)], dims="year", coords={"year": [int(year)]})
            rows.append(row)
        season_ds = xr.concat(rows, dim="year").expand_dims(season=[season])
        season_datasets.append(season_ds)

    out = xr.concat(season_datasets, dim="season")
    out = out.transpose("year", "season", ...)
    out.attrs.update(monthly.attrs)
    out["season_complete"].attrs.update(
        {
            "long_name": "whether all three calendar months are present for the seasonal mean",
            "definition": "True only when all constituent months have at least one daily sample",
        }
    )
    out.attrs["season_year_definition"] = "DJF is labeled by the Jan-Feb year; other seasons use the calendar year"
    out.attrs["seasonal_weighting"] = "daily-sample-count weighted mean using a variable-specific valid-data denominator"
    return out


def main() -> None:
    args = parse_args()
    if args.years[0] > args.years[1]:
        raise ValueError("--years START END requires START <= END")

    outdir = Path(args.outdir)
    daily_dir = outdir / "daily"
    ensure_dir(daily_dir)

    lon, lat = target_grid(args.lon_range, args.lat_range, args.resolution)
    extent = expanded_source_extent(args.lon_range, args.lat_range, args.resolution)

    static_path = outdir / "static_fields.nc"
    if static_path.exists() and not args.overwrite:
        static_ds = xr.open_dataset(static_path)
        static_source_files: list[str] = []
        print(f"[INFO] Keeping existing {static_path}")
    else:
        print("[INFO] Preparing static land-sea mask")
        static_ds, static_source_files = prepare_static_fields(args, lon, lat, extent)
        write_netcdf(static_ds, static_path)

    monthly_parts: list[xr.Dataset] = []
    all_source_files: dict[str, dict[str, list[str]]] = {}

    for year in range(args.years[0], args.years[1] + 1):
        daily_path = daily_dir / f"era5_daily_{year:04d}.nc"
        if daily_path.exists() and not args.overwrite:
            print(f"[INFO] {year}: reading existing {daily_path}")
            daily = xr.open_dataset(daily_path)
            provenance: dict[str, list[str]] = {}
        else:
            daily, provenance = prepare_year(args, year, lon, lat, extent)
            write_netcdf(daily, daily_path)
            print(f"[INFO] {year}: wrote {daily_path}")
        all_source_files[str(year)] = provenance
        monthly_parts.append(monthly_from_daily(daily, year))
        try:
            daily.close()
        except Exception:
            pass

    monthly = xr.concat(monthly_parts, dim="year").sortby("year")
    monthly.attrs.update(
        {
            "title": "Canonical monthly ERA5 fields for Atlantic ITCZ analysis",
            "schema_version": SCHEMA_VERSION,
            "script_version": SCRIPT_VERSION,
            "period": f"{args.years[0]}-{args.years[1]}",
            "monthly_definition": "arithmetic mean of canonical P1D calendar-day values",
            "ocean_mask_note": "Apply static_fields.nc:ocean_mask; use ocean_fraction for weighted spatial means.",
        }
    )
    monthly_path = outdir / "era5_monthly.nc"
    seasonal_path = outdir / "era5_seasonal.nc"
    write_netcdf(monthly, monthly_path)
    seasonal = seasonal_from_monthly(monthly)
    seasonal.attrs["title"] = "Canonical seasonal ERA5 fields for Atlantic ITCZ analysis"
    write_netcdf(seasonal, seasonal_path)

    metadata = {
        "pipeline_stage": "01_prepare",
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "script_name": Path(__file__).name,
        "script_sha256": sha256_file(Path(__file__)),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "era5_root": args.era5_root,
        "years": args.years,
        "canonical_lon_range": args.lon_range,
        "canonical_lat_range": args.lat_range,
        "canonical_resolution_degrees": args.resolution,
        "native_source_extent": extent,
        "source_time_resolution": "P1D",
        "surface_params": args.params,
        "compute_lts": not args.skip_lts,
        "ocean_mask_definition": f"regridded ERA5 lsm < {args.lsm_threshold:g}",
        "spatial_processing": "mean of finite native ERA5 grid-point values within each canonical 1-degree cell",
        "daily_time_processing": "P1D representative timestamps are floored to calendar-day midnight; duplicate calendar-day streams are averaged",
        "tp_processing": "ERA5 P1D tp in metres is multiplied by 1000 and stored as daily accumulated precipitation in mm day-1",
        "season_year_definition": "DJF labeled by Jan-Feb year",
        "source_files_lsm": static_source_files,
        "source_files_by_year": all_source_files,
        "outputs": {
            "daily_pattern": str(daily_dir / "era5_daily_YYYY.nc"),
            "monthly": str(monthly_path),
            "seasonal": str(seasonal_path),
            "static": str(static_path),
        },
    }
    write_json(outdir / "run_metadata.json", metadata)
    print(f"[INFO] Wrote {monthly_path}")
    print(f"[INFO] Wrote {seasonal_path}")
    print(f"[INFO] Wrote {outdir / 'run_metadata.json'}")


if __name__ == "__main__":
    main()
