#!/usr/bin/env python3
"""Shared ERA5 preprocessing utilities for the Atlantic ITCZ analysis.

Functions cover DKRZ file discovery, GRIB decoding, coordinate and time
handling, finite-point binning to the 1-degree grid, units, and NetCDF output.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr

SCHEMA_VERSION = "1.0"
EARTH_RADIUS_M = 6_371_000.0
KAPPA = 287.0 / 1004.0

PARAM_CODE = {
    "t": "130",
    "u": "131",
    "v": "132",
    "sp": "134",
    "skt": "235",
    "w": "135",
    "tcwv": "137",
    "msl": "151",
    "d": "155",
    "blh": "159",
    "10u": "165",
    "10v": "166",
    "2t": "167",
    "t2m": "167",
    "sst": "34",
    "lsm": "172",
    "tp": "228",
}

CANONICAL_NAMES = {
    "10u": "u10",
    "u10": "u10",
    "10v": "v10",
    "v10": "v10",
    "2t": "t2m",
    "t2m": "t2m",
}

VAR_ALIASES = {
    "tcwv": ["tcwv", "TCWV", "prw", "PW", "cw", "cwv"],
    "tp": ["tp", "precip", "precipitation", "pr", "mtpr"],
    "sst": ["sst", "SST", "tos", "sea_surface_temperature"],
    "u10": ["10u", "u10", "uas", "u_10m"],
    "v10": ["10v", "v10", "vas", "v_10m"],
    "t2m": ["2t", "t2m", "tas", "t_2m", "air_temperature_2m"],
    "t": ["t", "ta", "T"],
    "blh": ["blh", "pblh", "boundary_layer_height"],
    "msl": ["msl", "mslp", "mean_sea_level_pressure", "psl"],
    "lsm": ["lsm", "sftlf", "land_sea_mask"],
}


def canonical_name(name: str) -> str:
    """Return the stable variable name used by downstream pipeline stages."""
    return CANONICAL_NAMES.get(str(name).lower(), str(name).lower())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")


def candidate_param_names(param: str) -> list[str]:
    """Return directory/file tokens that may identify an ERA5 parameter."""
    p = str(param).strip()
    names = [p, p.lower(), p.upper()]
    code = PARAM_CODE.get(p.lower())
    if code is not None:
        n = int(code)
        names.extend([str(n), f"{n:03d}", code])
    out: list[str] = []
    for name in names:
        if name not in out:
            out.append(name)
    return out


def preferred_time_res_dirs(time_res: str) -> list[str]:
    if time_res == "P1D":
        return ["1D", "day", "daily"]
    if time_res == "PT1H":
        return ["1H", "1h", "hourly", "1hr"]
    raise ValueError(f"Unsupported time resolution: {time_res}")


def local_era5_level_dirs_for_param(param: str) -> list[str]:
    p = str(param).lower()
    if p in {"u", "v", "d", "w", "t"}:
        return ["pl", "pressure-levels"]
    return ["sf", "sfc", "sl", "single", "single-levels"]


def _token_int_equal(a: str, b: str) -> bool:
    try:
        return int(str(a)) == int(str(b))
    except Exception:
        return str(a).lower() == str(b).lower()


def path_matches_param_token(path: Path, param: str) -> bool:
    parent = path.parent.name
    stem_token = path.stem.split("_")[-1] if "_" in path.stem else path.stem
    return any(
        _token_int_equal(parent, token) or _token_int_equal(stem_token, token)
        for token in candidate_param_names(param)
    )


def path_matches_time_res(path: Path, time_res: str) -> bool:
    text = "/" + str(path) + "/"
    return any(f"/{token}/" in text for token in preferred_time_res_dirs(time_res))


def candidate_param_dirs(root: Path, param: str, time_res: str) -> list[Path]:
    dirs: list[Path] = []
    for lev in local_era5_level_dirs_for_param(param):
        for typ in ["an", "fc", "analysis", "forecast"]:
            for tres in preferred_time_res_dirs(time_res):
                for pname in candidate_param_names(param):
                    dirs.append(root / lev / typ / tres / pname)
    return dirs


def find_param_files(
    root: Path,
    param: str,
    year: int,
    time_res: str = "P1D",
    allow_recursive: bool = True,
) -> list[Path]:
    """Locate all files for one ERA5 parameter and year.

    Exact parameter-token matching is important: a loose search for parameter
    34 (SST), for example, can otherwise match parameter 134 (surface pressure).
    """
    hits: list[Path] = []
    for directory in candidate_param_dirs(root, param, time_res):
        if directory.exists():
            hits.extend(directory.glob(f"*{year:04d}*.grb"))
            hits.extend(directory.glob(f"*{year:04d}*.grib"))
    hits = [p for p in hits if path_matches_param_token(p, param)]
    if hits or not allow_recursive:
        return sorted(set(hits))

    recursive: list[Path] = []
    for pname in candidate_param_names(param):
        recursive.extend(root.glob(f"**/*{year:04d}*_{pname}.grb"))
        recursive.extend(root.glob(f"**/*{year:04d}*_{pname}.grib"))
    return sorted(
        set(
            p
            for p in recursive
            if path_matches_time_res(p, time_res)
            and path_matches_param_token(p, param)
        )
    )


def find_static_param_files(
    root: Path,
    param: str,
    allow_recursive: bool = True,
) -> list[Path]:
    """Locate invariant ERA5 fields such as the land-sea mask."""
    hits: list[Path] = []
    for lev in local_era5_level_dirs_for_param(param):
        for typ in ["an", "fc", "analysis", "forecast"]:
            for tres in ["IV", "iv", "invariant", "constant", "const", "static"]:
                for pname in candidate_param_names(param):
                    directory = root / lev / typ / tres / pname
                    if directory.exists():
                        hits.extend(directory.glob("*.grb"))
                        hits.extend(directory.glob("*.grib"))
    hits = [p for p in hits if path_matches_param_token(p, param)]
    if hits or not allow_recursive:
        return sorted(set(hits))

    recursive: list[Path] = []
    for pname in candidate_param_names(param):
        recursive.extend(root.glob(f"**/*_{pname}.grb"))
        recursive.extend(root.glob(f"**/*_{pname}.grib"))
    return sorted(set(p for p in recursive if path_matches_param_token(p, param)))


def open_grib_dataset(path: Path, short_name: Optional[str] = None) -> xr.Dataset:
    """Open one GRIB file with cfgrib without writing index files."""
    filters: list[dict[str, object] | None] = []
    if short_name is not None:
        code = PARAM_CODE.get(str(short_name).lower())
        if code is not None:
            filters.extend(
                [
                    {"paramId": int(code)},
                    {"paramId": int(code), "typeOfLevel": "surface"},
                    {"paramId": int(code), "typeOfLevel": "isobaricInhPa"},
                ]
            )
        filters.extend(
            [
                {"shortName": str(short_name).lower()},
                {"shortName": str(short_name).lower(), "typeOfLevel": "surface"},
                {"shortName": str(short_name).lower(), "typeOfLevel": "isobaricInhPa"},
            ]
        )
    filters.append(None)

    errors: list[str] = []
    for filter_keys in filters:
        try:
            backend: dict[str, object] = {"indexpath": ""}
            if filter_keys is not None:
                backend["filter_by_keys"] = filter_keys
            return xr.open_dataset(path, engine="cfgrib", backend_kwargs=backend)
        except Exception as exc:
            errors.append(f"filter_by_keys={filter_keys}: {type(exc).__name__}: {exc}")

    try:
        import cfgrib

        datasets = cfgrib.open_datasets(str(path), indexpath="")
        if not datasets:
            raise RuntimeError("cfgrib.open_datasets returned no datasets")
        if short_name is None:
            return datasets[0]
        for ds in datasets:
            try:
                _ = find_grib_data_var(ds, short_name)
                return ds
            except Exception:
                continue
    except Exception as exc:
        errors.append(f"cfgrib.open_datasets: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        f"Failed to open {path} with cfgrib.\n" + "\n---\n".join(errors)
    )


def find_grib_data_var(ds: xr.Dataset, short_name: str) -> str:
    canonical = canonical_name(short_name)
    aliases = [short_name, canonical]
    aliases.extend(VAR_ALIASES.get(canonical, []))
    for name in aliases:
        if name in ds.data_vars:
            return name

    code = PARAM_CODE.get(str(short_name).lower())
    for name, da in ds.data_vars.items():
        attrs = da.attrs
        if str(attrs.get("GRIB_shortName", "")).lower() == str(short_name).lower():
            return name
        if code is not None and str(attrs.get("GRIB_paramId", "")) == str(int(code)):
            return name
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise KeyError(f"Could not identify {short_name!r}; data variables={list(ds.data_vars)}")


def _rename_grib_coords(obj: xr.Dataset | xr.DataArray):
    rename: dict[str, str] = {}
    for src in ["latitude", "grid_latitude"]:
        if src in obj.coords or src in obj.dims:
            rename[src] = "lat"
    for src in ["longitude", "grid_longitude"]:
        if src in obj.coords or src in obj.dims:
            rename[src] = "lon"
    for src in ["isobaricInhPa", "isobaric", "pressure_level", "plev"]:
        if src in obj.coords or src in obj.dims:
            rename[src] = "level"
    return obj.rename(rename) if rename else obj


def _ensure_cell_layout(da: xr.DataArray) -> xr.DataArray:
    da = _rename_grib_coords(da)
    if "values" in da.dims:
        da = da.rename({"values": "cell"})
    elif "lat" in da.dims and "lon" in da.dims:
        da = da.stack(cell=("lat", "lon")).reset_index("cell")
    elif "cell" not in da.dims:
        raise ValueError(f"Cannot identify spatial layout: dims={da.dims}")
    if "lat" not in da.coords or "lon" not in da.coords:
        raise ValueError("Spatial cell layout lacks lat/lon coordinates")
    lon = ((da["lon"].astype("float64") + 180.0) % 360.0) - 180.0
    return da.assign_coords(lon=("cell", np.asarray(lon)))


def _subset_cell_extent(da: xr.DataArray, extent: Sequence[float]) -> xr.DataArray:
    lon_min, lon_max, lat_min, lat_max = [float(v) for v in extent]
    lon_mask = (da.lon >= lon_min) & (da.lon <= lon_max)
    lat_mask = (da.lat >= lat_min) & (da.lat <= lat_max)
    return da.where(lon_mask & lat_mask, drop=True)


def _assign_valid_time(da: xr.DataArray, ds: xr.Dataset) -> xr.DataArray:
    preserve = {"cell", "level"}
    sample_dims = [d for d in da.dims if d not in preserve]
    if not sample_dims:
        return da

    ds2 = _rename_grib_coords(ds)
    if "valid_time" in ds2.coords:
        vt = _rename_grib_coords(ds2["valid_time"])
        vt_dims = [d for d in vt.dims if d in sample_dims]
        if sample_dims == ["time"] and vt_dims == ["time"]:
            return da.assign_coords(time=pd.to_datetime(vt.values))
        da_stacked = da.stack(sample=sample_dims)
        if len(vt_dims) > 1:
            vt_stacked = vt.stack(sample=vt_dims)
        elif len(vt_dims) == 1:
            vt_stacked = vt.rename({vt_dims[0]: "sample"})
        else:
            raise ValueError("valid_time is not aligned with the sample dimensions")
        out = da_stacked.assign_coords(
            time=("sample", pd.to_datetime(vt_stacked.values))
        ).swap_dims({"sample": "time"})
        return out.drop_vars(["sample"] + sample_dims, errors="ignore")

    if sample_dims == ["time"]:
        return da.assign_coords(time=pd.to_datetime(da.time.values))
    raise ValueError(f"No recognizable valid time: dims={da.dims}")


def open_one_grib_var(
    path: Path,
    short_name: str,
    extent: Sequence[float],
) -> xr.DataArray:
    ds = open_grib_dataset(path, short_name=short_name)
    try:
        ds = _rename_grib_coords(ds)
        name = find_grib_data_var(ds, short_name)
        source = ds[name]
        da = _ensure_cell_layout(source)
        da = _subset_cell_extent(da, extent)
        da = _assign_valid_time(da, ds)
        if "level" in da.dims:
            da = da.transpose("time", "level", "cell")
        elif "time" in da.dims:
            da = da.transpose("time", "cell")
        else:
            da = da.transpose("cell")
        da.name = canonical_name(short_name)
        da.attrs.update(source.attrs)
        da.attrs["source_file"] = str(path)
        return da.sortby("time") if "time" in da.dims else da
    finally:
        ds.close()


def force_calendar_day_labels(da: xr.DataArray) -> xr.DataArray:
    """Label each P1D record at midnight on its calendar day.

    DKRZ analysis and forecast products can have different representative times.
    Records sharing a day are averaged after calendar-day labels are assigned.
    """
    if "time" not in da.dims:
        return da
    dates = pd.to_datetime(da.time.values).floor("D")
    out = da.assign_coords(time=dates)
    if pd.Index(dates).has_duplicates:
        out = out.groupby("time").mean(skipna=True)
    return out.sortby("time")


def load_era5_year_param(
    root: Path,
    param: str,
    year: int,
    extent: Sequence[float],
    *,
    time_res: str = "P1D",
    allow_recursive: bool = True,
) -> tuple[xr.DataArray, list[str]]:
    """Read one parameter for one year from the DKRZ ERA5 tree."""
    if param == "lsm":
        files = find_static_param_files(root, "lsm", allow_recursive=allow_recursive)
    else:
        files = find_param_files(
            root, param, year, time_res=time_res, allow_recursive=allow_recursive
        )
    if not files:
        raise FileNotFoundError(f"No ERA5 files found for {param}, year={year}")

    arrays: list[xr.DataArray] = []
    used: list[str] = []
    for path in files:
        try:
            arrays.append(open_one_grib_var(path, param, extent))
            used.append(str(path))
        except Exception as exc:
            warnings.warn(f"Skipping unreadable ERA5 file {path}: {exc}")
    if not arrays:
        raise RuntimeError(f"All ERA5 files failed for {param}, year={year}")

    if param == "lsm":
        da = arrays[0]
        if "time" in da.dims:
            da = da.isel(time=0, drop=True)
        da.name = "lsm"
        return da, used

    da = xr.concat(arrays, dim="time") if len(arrays) > 1 else arrays[0]
    da = da.sortby("time")
    _, idx = np.unique(da.time.values, return_index=True)
    da = da.isel(time=np.sort(idx))
    if time_res == "P1D":
        da = force_calendar_day_labels(da)
    start = np.datetime64(f"{year:04d}-01-01")
    end = np.datetime64(f"{year:04d}-12-31")
    da = da.sel(time=slice(start, end))
    return da, used


def target_grid(
    lon_range: Sequence[float],
    lat_range: Sequence[float],
    resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical grid-center coordinates, including both end points."""
    lon0, lon1 = [float(v) for v in lon_range]
    lat0, lat1 = [float(v) for v in lat_range]
    lon = np.arange(lon0, lon1 + 0.25 * resolution, resolution, dtype=float)
    lat = np.arange(lat0, lat1 + 0.25 * resolution, resolution, dtype=float)
    return lon, lat


def expanded_source_extent(
    lon_range: Sequence[float],
    lat_range: Sequence[float],
    resolution: float,
) -> list[float]:
    """Extent needed to fill edge cells of a center-defined regular grid."""
    half = 0.5 * float(resolution) + 1.0e-6
    return [
        float(lon_range[0]) - half,
        float(lon_range[1]) + half,
        float(lat_range[0]) - half,
        float(lat_range[1]) + half,
    ]


def _regular_cell_index(
    source_lon: np.ndarray,
    source_lat: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
    resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map native grid points to 1-D target-cell indices."""
    lon_edge0 = target_lon[0] - 0.5 * resolution
    lat_edge0 = target_lat[0] - 0.5 * resolution
    ilon = np.floor((source_lon - lon_edge0) / resolution).astype(int)
    ilat = np.floor((source_lat - lat_edge0) / resolution).astype(int)
    valid = (
        (ilon >= 0)
        & (ilon < target_lon.size)
        & (ilat >= 0)
        & (ilat < target_lat.size)
        & np.isfinite(source_lon)
        & np.isfinite(source_lat)
    )
    flat = ilat * target_lon.size + ilon
    return flat, valid


def bin_average_to_regular_grid(
    da: xr.DataArray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
    resolution: float,
) -> xr.DataArray:
    """Bin finite native values into regular latitude-longitude cells.

    Each target cell contains the mean of finite native points within its bounds.
    The operation coarsens the source grid without interpolating new extrema.
    """
    if "cell" not in da.dims:
        raise ValueError("bin_average_to_regular_grid requires a 'cell' dimension")
    source_lon = np.asarray(da.lon.values, dtype=float)
    source_lat = np.asarray(da.lat.values, dtype=float)
    flat_index, valid_cell = _regular_cell_index(
        source_lon, source_lat, target_lon, target_lat, resolution
    )
    n_target = target_lon.size * target_lat.size

    other_dims = [d for d in da.dims if d != "cell"]
    transposed = da.transpose(*other_dims, "cell")
    values = np.asarray(transposed.values, dtype=float)
    shape_other = values.shape[:-1]
    values2 = values.reshape((-1, values.shape[-1]))
    output = np.full((values2.shape[0], n_target), np.nan, dtype=np.float32)

    idx_all = flat_index[valid_cell]
    for i, row in enumerate(values2):
        vals = row[valid_cell]
        good = np.isfinite(vals)
        if not np.any(good):
            continue
        idx = idx_all[good]
        numerator = np.bincount(idx, weights=vals[good], minlength=n_target)
        denominator = np.bincount(idx, minlength=n_target)
        target = np.full(n_target, np.nan, dtype=float)
        np.divide(numerator, denominator, out=target, where=denominator > 0)
        output[i] = target.astype(np.float32)

    output = output.reshape(shape_other + (target_lat.size, target_lon.size))
    coords = {d: transposed[d] for d in other_dims}
    coords.update({"lat": target_lat, "lon": target_lon})
    out = xr.DataArray(
        output,
        dims=tuple(other_dims) + ("lat", "lon"),
        coords=coords,
        name=da.name,
        attrs=dict(da.attrs),
    )
    out.attrs["horizontal_processing"] = (
        f"mean of finite native ERA5 grid-point values within {resolution:g}-degree "
        "latitude-longitude cells"
    )
    return out


def standardize_units(da: xr.DataArray, canonical: str) -> xr.DataArray:
    """Convert variables to the stable units used by downstream diagnostics."""
    name = canonical_name(canonical)
    out = da.astype("float32")
    source_units = str(da.attrs.get("units", ""))
    source_step_type = str(da.attrs.get("GRIB_stepType", ""))
    out.attrs["source_units"] = source_units
    out.attrs["source_GRIB_stepType"] = source_step_type

    if name == "tp":
        # Convert daily accumulated precipitation from meters to mm/day.
        # Monthly and seasonal means then represent mean daily precipitation.
        if source_units.strip().lower() in {"m", "metre", "meter"}:
            out = out * np.float32(1000.0)
        out.attrs["units"] = "mm day-1"
        out.attrs["daily_quantity"] = "daily accumulated precipitation expressed as mm day-1"
    elif name in {"sst", "t2m", "t"}:
        out.attrs["units"] = "K"
    elif name in {"u10", "v10"}:
        out.attrs["units"] = "m s-1"
    elif name == "tcwv":
        out.attrs["units"] = "kg m-2"
    elif name == "blh":
        out.attrs["units"] = "m"
    elif name == "msl":
        out.attrs["units"] = "Pa"
    return out


def compute_lts(t700: xr.DataArray, t1000: xr.DataArray) -> xr.DataArray:
    """Lower-tropospheric stability: theta(700 hPa) - theta(1000 hPa)."""
    theta700 = t700 * np.float32((1000.0 / 700.0) ** KAPPA)
    theta1000 = t1000
    lts = (theta700 - theta1000).astype("float32")
    lts.name = "lts"
    lts.attrs.update(
        {
            "long_name": "lower-tropospheric stability",
            "definition": "potential_temperature_700hPa - potential_temperature_1000hPa",
            "units": "K",
        }
    )
    return lts


def write_netcdf(ds: xr.Dataset, path: Path | str) -> None:
    """Write compressed NetCDF when a suitable backend is available."""
    path = Path(path)
    ensure_dir(path.parent)
    encoding = {
        name: {"zlib": True, "complevel": 3, "shuffle": True}
        for name in ds.data_vars
        if np.issubdtype(ds[name].dtype, np.number)
    }
    for engine in ("h5netcdf", "netcdf4"):
        try:
            ds.to_netcdf(path, engine=engine, encoding=encoding)
            return
        except Exception:
            continue
    warnings.warn("Compressed NetCDF backends unavailable; writing uncompressed NetCDF")
    ds.to_netcdf(path)
