#!/usr/bin/env python3
"""Relate Atlantic TCWV modes to SST and horizontal atmospheric fields.

Stage 3 reads selected_shape_modes.csv from Stage 2; it does not select the
shape mode again. The selected EOF can differ by calendar month, season, or
temporal treatment. Unless --shape-only is used, the highest-variance mode
classified as shift is also retained for each period and series type.

Outputs include SST metrics, PC-SST correlations and standardized two-predictor
regressions, regression maps, and top/bottom-PC composites. SST metrics include
the zonal-mean peak, peak-to-flank contrast, fixed equatorial bands, and level
and asymmetry coordinates around fixed climatological TCWV or SST peaks.

Raw and field-detrended modes are analyzed separately. For detrended targets,
horizontal fields and derived SST metric series are detrended across years.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from scipy.ndimage import gaussian_filter1d
except Exception:  # pragma: no cover
    gaussian_filter1d = None

try:
    from scipy.stats import pearsonr, t as student_t
except Exception:  # pragma: no cover
    pearsonr = None
    student_t = None

from atlantic_itcz_common import EARTH_RADIUS_M, SCHEMA_VERSION, ensure_dir, sha256_file, write_json, write_netcdf

SCRIPT_VERSION = "1.7.0"
DEFAULT_PREPARE = "out_atlantic_itcz/01_prepare"
DEFAULT_MODES = "out_atlantic_itcz/02_modes"
DEFAULT_OUTPUT = "out_atlantic_itcz/03_state"
SERIES_TYPES = ("raw", "linear_detrended")
COMPOSITES = ("high_anomaly", "low_anomaly", "high_minus_low", "high_state", "low_state")
SECONDS_PER_DAY = 86_400.0

BASE_MAP_VARIABLES = ("sst", "tcwv", "tp", "u10", "v10", "blh", "lts")
DERIVED_MAP_VARIABLES = ("neg_dudx", "neg_dvdy", "convergence")
SST_METRICS = (
    "sst_mean_K",
    "sst_p95_K",
    "sst_p05_K",
    "sst_warm_mean_p90_K",
    "sst_zm_peak_K",
    "sst_zm_peak_lat_deg",
    "sst_peak_to_flank_mean_15deg_K",
    "sst_peak_to_flank_mean_20deg_K",
    "sst_peak_to_flank_mean_25deg_K",
    "sst_amp_zm_p95_p05_K",
    "sst_eq_south_mean_K",
    "sst_eq_north_mean_K",
    "sst_eq_ns_K",
    "sst_tcwvclim_south_mean_K",
    "sst_tcwvclim_north_mean_K",
    "sst_tcwvclim_ns_K",
    "sst_sstclim_south_mean_K",
    "sst_sstclim_north_mean_K",
    "sst_sstclim_ns_K",
)
SHIFT_REFERENCE_SPECS = {
    "equator": ("sst_eq_south_mean_K", "sst_eq_north_mean_K", "sst_eq_ns_K"),
    "tcwv_clim_peak": (
        "sst_tcwvclim_south_mean_K",
        "sst_tcwvclim_north_mean_K",
        "sst_tcwvclim_ns_K",
    ),
    "sst_clim_peak": (
        "sst_sstclim_south_mean_K",
        "sst_sstclim_north_mean_K",
        "sst_sstclim_ns_K",
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare-dir", default=DEFAULT_PREPARE)
    p.add_argument("--modes-dir", default=DEFAULT_MODES)
    p.add_argument("--outdir", default=DEFAULT_OUTPUT)
    p.add_argument("--analysis-lon-range", nargs=2, type=float, default=[-50.0, -20.0])
    p.add_argument("--analysis-lat-range", nargs=2, type=float, default=[-15.0, 30.0])
    p.add_argument("--sst-metric-lon-range", nargs=2, type=float, default=[-50.0, -20.0])
    p.add_argument("--sst-metric-lat-range", nargs=2, type=float, default=[-25.0, 35.0])
    p.add_argument("--sst-peak-lat-range", nargs=2, type=float, default=[-5.0, 20.0])
    p.add_argument("--eq-south-lat-range", nargs=2, type=float, default=[-15.0, 0.0])
    p.add_argument("--eq-north-lat-range", nargs=2, type=float, default=[0.0, 15.0])
    p.add_argument(
        "--centered-sst-half-width-deg",
        type=float,
        default=15.0,
        help=(
            "Half-width of the south/north SST averaging bands centered on the "
            "fixed climatological TCWV or SST peak. Default: 15 degrees, matching "
            "the equator-centered south/north bands."
        ),
    )
    p.add_argument(
        "--sst-scale-sweep-deg",
        nargs="+",
        type=float,
        default=[2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0],
        help=(
            "Meridional scales (degrees) used only for SST sensitivity diagnostics. "
            "For peak-to-flank metrics, each value is the distance from the annual SST peak. "
            "For centered south/north metrics, each value is the half-width of each band. "
            "Values extending beyond the prepared SST latitude domain yield missing diagnostics "
            "for affected years rather than extrapolation."
        ),
    )
    p.add_argument(
        "--sst-scale-bootstrap-samples",
        type=int,
        default=1000,
        help=(
            "Number of year-resampling bootstrap replicates used for 95%% confidence "
            "intervals in SST meridional-scale diagnostics. Set to 0 to disable. "
            "Default: 1000."
        ),
    )
    p.add_argument(
        "--sst-scale-bootstrap-seed",
        type=int,
        default=260904,
        help="Random seed for SST meridional-scale bootstrap diagnostics.",
    )
    p.add_argument(
        "--skip-exhaustive-sst-figures",
        action="store_true",
        help=(
            "Compute exhaustive seasonal PC1/PC2 x SST-space diagnostics but skip "
            "the automatically generated state-space and scale-sensitivity figures."
        ),
    )
    p.add_argument(
        "--exhaustive-sst-series-types",
        nargs="+",
        default=["raw"],
        help=(
            "Series types used in the exhaustive seasonal PC1/PC2 SST-space comparison. "
            "Default: raw. Use 'raw linear_detrended' to make both sets."
        ),
    )
    p.add_argument(
        "--shape-only",
        action="store_true",
        help=(
            "Analyze only the canonical shape modes from selected_shape_modes.csv. "
            "By default, clearly classified shift modes are also retained as "
            "secondary targets so earlier migration analyses remain reproducible."
        ),
    )
    p.add_argument("--series-types", nargs="+", default=list(SERIES_TYPES))
    p.add_argument("--map-vars", nargs="+", default=list(BASE_MAP_VARIABLES))
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--bottom-n", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def safe_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    n = int(good.sum())
    if n < 3 or np.nanstd(x[good]) <= 0 or np.nanstd(y[good]) <= 0:
        return np.nan, np.nan, n
    if pearsonr is not None:
        result = pearsonr(x[good], y[good])
        return float(result.statistic), float(result.pvalue), n
    return float(np.corrcoef(x[good], y[good])[0, 1]), np.nan, n


def standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    good = np.isfinite(x)
    if good.sum() < 2:
        return out
    sd = np.std(x[good], ddof=1)
    if sd <= 0:
        return out
    out[good] = (x[good]-np.mean(x[good]))/sd
    return out


def detrend_vector(years: np.ndarray, values: np.ndarray, restore_mean: bool = True) -> np.ndarray:
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    good = np.isfinite(years) & np.isfinite(values)
    if good.sum() < 3:
        return out
    yc = years[good]-np.mean(years[good])
    slope, intercept = np.polyfit(yc, values[good], 1)
    fit = slope*(years-np.mean(years[good]))+intercept
    out[good] = values[good]-fit[good]
    if restore_mean:
        out[good] += np.mean(values[good])
    return out


def detrend_field(field: xr.DataArray) -> xr.DataArray:
    """Remove a linear trend at each grid point along the year dimension.

    The calculation is vectorized over the horizontal grid.  Missing values are
    handled independently at each grid point.  The original temporal mean is
    retained, so detrended fields remain in their physical units.
    """
    field = field.transpose("year", "lat", "lon")
    years = np.asarray(field.year.values, dtype=float)
    y = np.asarray(field.values, dtype=float)
    shape = y.shape
    yf = y.reshape((shape[0], -1))
    x = years[:, None]
    good = np.isfinite(yf)
    n = good.sum(axis=0).astype(float)
    y0 = np.where(good, yf, 0.0)
    x0 = np.where(good, x, 0.0)
    sumx = x0.sum(axis=0)
    sumy = y0.sum(axis=0)
    sumxx = (x0*x0).sum(axis=0)
    sumxy = (x0*y0).sum(axis=0)
    denom = sumxx - np.divide(sumx*sumx, n, out=np.zeros_like(sumx), where=n>0)
    numer = sumxy - np.divide(sumx*sumy, n, out=np.zeros_like(sumx), where=n>0)
    slope = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=(n>=3) & (denom>0))
    xmean = np.divide(sumx, n, out=np.full_like(sumx, np.nan), where=n>0)
    result = yf - (x-xmean[None, :])*slope[None, :]
    result[~good] = np.nan
    result = result.reshape(shape).astype("float32")
    out = xr.DataArray(result, dims=field.dims, coords=field.coords, name=field.name, attrs=dict(field.attrs))
    out.attrs["temporal_processing"] = "linear trend removed across years; original interannual mean restored"
    return out


def weighted_mean_2d(da: xr.DataArray, weight: xr.DataArray, mask: xr.DataArray) -> float:
    valid = np.isfinite(da) & mask
    w = weight.where(valid)
    den = float(w.sum(skipna=True))
    return float((da.where(valid)*w).sum(skipna=True)/den) if den > 0 else np.nan


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if good.sum() < 2:
        return np.nan
    v = values[good]
    w = weights[good]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    return float(np.interp(q, cdf, v))


def smooth_1d(values: np.ndarray, sigma_grid: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    if sigma_grid <= 0 or good.sum() < 3:
        return values.copy()
    if gaussian_filter1d is None:
        return values.copy()
    num = gaussian_filter1d(np.where(good, values, 0.0), sigma_grid, mode="nearest")
    den = gaussian_filter1d(good.astype(float), sigma_grid, mode="nearest")
    out = np.full_like(values, np.nan)
    use = den > 1e-8
    out[use] = num[use]/den[use]
    return out


def interp_scalar(x: np.ndarray, y: np.ndarray, point: float) -> float:
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 2 or point < np.nanmin(x[good]) or point > np.nanmax(x[good]):
        return np.nan
    return float(np.interp(point, x[good], y[good]))



def zonal_mean_sst_profile(
    sst: xr.DataArray,
    static: xr.Dataset,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the smoothed ocean-weighted zonal-mean SST profile."""
    sst = sst.sel(
        lon=slice(args.sst_metric_lon_range[0], args.sst_metric_lon_range[1]),
        lat=slice(args.sst_metric_lat_range[0], args.sst_metric_lat_range[1]),
    )
    ocean_fraction = static.ocean_fraction.sel(lat=sst.lat, lon=sst.lon)
    ocean_mask = static.ocean_mask.sel(lat=sst.lat, lon=sst.lon)
    valid = np.isfinite(sst) & ocean_mask
    num = (sst.where(valid) * ocean_fraction.where(valid)).sum("lon", skipna=True)
    den = ocean_fraction.where(valid).sum("lon", skipna=True)
    zm = num / den
    zlat = np.asarray(zm.lat.values, dtype=float)
    zval = smooth_1d(np.asarray(zm.values, dtype=float), 1.0)
    return zlat, zval


def peak_latitude_from_profile(
    lat: np.ndarray,
    values: np.ndarray,
    search_range: Sequence[float],
) -> float:
    search = (lat >= search_range[0]) & (lat <= search_range[1]) & np.isfinite(values)
    if not search.any():
        return np.nan
    j = np.where(search)[0][int(np.nanargmax(values[search]))]
    return float(lat[j])


def fixed_center_south_north_means(
    sst: xr.DataArray,
    static: xr.Dataset,
    args: argparse.Namespace,
    center_lat_deg: float,
) -> tuple[float, float, float]:
    """SST means in fixed bands immediately south/north of a reference latitude.

    The center is fixed for a given calendar month or season across all years.
    The south band is [center-half_width, center) and the north band is
    [center, center+half_width]. This avoids double counting a grid row at the
    reference latitude and, unlike a year-following TCWV center, does not make
    the predictor depend on the interannual response.
    """
    if not np.isfinite(center_lat_deg):
        return np.nan, np.nan, np.nan

    sst = sst.sel(
        lon=slice(args.sst_metric_lon_range[0], args.sst_metric_lon_range[1]),
        lat=slice(args.sst_metric_lat_range[0], args.sst_metric_lat_range[1]),
    )
    ocean_fraction = static.ocean_fraction.sel(lat=sst.lat, lon=sst.lon)
    ocean_mask = static.ocean_mask.sel(lat=sst.lat, lon=sst.lon)
    coslat = xr.DataArray(np.cos(np.deg2rad(sst.lat)), dims="lat", coords={"lat": sst.lat})
    area_weight = ocean_fraction * coslat
    lat2d = sst.lat.broadcast_like(sst)
    h = float(args.centered_sst_half_width_deg)

    south_mask = ocean_mask & (lat2d >= center_lat_deg - h) & (lat2d < center_lat_deg)
    north_mask = ocean_mask & (lat2d >= center_lat_deg) & (lat2d <= center_lat_deg + h)
    south = weighted_mean_2d(sst, area_weight, south_mask)
    north = weighted_mean_2d(sst, area_weight, north_mask)
    contrast = north - south if np.isfinite(north) and np.isfinite(south) else np.nan
    return south, north, contrast


def fixed_reference_centers(
    source: xr.Dataset,
    modes_monthly: xr.Dataset,
    modes_seasonal: xr.Dataset,
    period_type: str,
    period: object,
    static: xr.Dataset,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Diagnose fixed reference latitudes for south/north SST bands.

    ``tcwv_clim_peak`` is the maximum of the Stage-2 climatological zonal-mean
    TCWV profile. ``sst_clim_peak`` is the maximum of the climatological
    zonal-mean SST profile. Both are fixed across years and therefore avoid
    response-following sampling in interannual regressions.
    """
    if period_type == "monthly":
        qclim = modes_monthly["tcwv_climatology"].sel(month=int(period))
    else:
        qclim = modes_seasonal["tcwv_climatology"].sel(season=str(period))
    qlat = np.asarray(qclim.lat.values, dtype=float)
    qval = np.asarray(qclim.values, dtype=float)
    qsearch = np.isfinite(qval)
    tcwv_peak = float(qlat[np.where(qsearch)[0][int(np.nanargmax(qval[qsearch]))]]) if qsearch.any() else np.nan

    sst_yearly = get_period_field(source, period_type, period, "sst")
    if period_type == "seasonal" and "season_complete" in source:
        try:
            complete = source["season_complete"].sel(season=str(period))
            good_years = complete.year.values[np.asarray(complete.values).astype(bool)]
            sst_yearly = sst_yearly.sel(year=good_years)
        except Exception:
            pass
    sst_clim = sst_yearly.mean("year", skipna=True)
    slat, sval = zonal_mean_sst_profile(sst_clim, static, args)
    sst_peak = peak_latitude_from_profile(slat, sval, args.sst_peak_lat_range)

    return {
        "equator": 0.0,
        "tcwv_clim_peak": tcwv_peak,
        "sst_clim_peak": sst_peak,
    }

def sst_metrics_for_field(
    sst: xr.DataArray,
    static: xr.Dataset,
    args: argparse.Namespace,
    reference_centers: dict[str, float],
) -> dict[str, float]:
    """Diagnose SST-state metrics from one monthly or seasonal SST field."""
    sst = sst.sel(
        lon=slice(args.sst_metric_lon_range[0], args.sst_metric_lon_range[1]),
        lat=slice(args.sst_metric_lat_range[0], args.sst_metric_lat_range[1]),
    )
    ocean_fraction = static.ocean_fraction.sel(lat=sst.lat, lon=sst.lon)
    ocean_mask = static.ocean_mask.sel(lat=sst.lat, lon=sst.lon)
    coslat = xr.DataArray(np.cos(np.deg2rad(sst.lat)), dims="lat", coords={"lat": sst.lat})
    area_weight = ocean_fraction*coslat
    valid_weight = np.asarray(area_weight.where(ocean_mask & np.isfinite(sst)).values, dtype=float)
    values = np.asarray(sst.where(ocean_mask).values, dtype=float)

    mean = weighted_mean_2d(sst, area_weight, ocean_mask)
    p95 = weighted_quantile(values, valid_weight, 0.95)
    p05 = weighted_quantile(values, valid_weight, 0.05)
    p90 = weighted_quantile(values, valid_weight, 0.90)
    warm_mask = ocean_mask & (sst >= p90)
    warm_mean = weighted_mean_2d(sst, area_weight, warm_mask)

    # Zonal-mean SST profile. Longitude means use fractional ocean area;
    # latitude weighting is not used because each latitude is a distinct
    # point in the meridional profile.
    zlat, zval = zonal_mean_sst_profile(sst, static, args)
    peak_lat = peak_latitude_from_profile(zlat, zval, args.sst_peak_lat_range)
    peak = interp_scalar(zlat, zval, peak_lat) if np.isfinite(peak_lat) else np.nan

    out = {
        "sst_mean_K": mean,
        "sst_p95_K": p95,
        "sst_p05_K": p05,
        "sst_warm_mean_p90_K": warm_mean,
        "sst_zm_peak_K": peak,
        "sst_zm_peak_lat_deg": peak_lat,
    }
    for distance in (15.0, 20.0, 25.0):
        south = interp_scalar(zlat, zval, peak_lat-distance) if np.isfinite(peak_lat) else np.nan
        north = interp_scalar(zlat, zval, peak_lat+distance) if np.isfinite(peak_lat) else np.nan
        flank = np.nanmean([south, north]) if np.isfinite([south, north]).any() else np.nan
        out[f"sst_peak_to_flank_mean_{int(distance)}deg_K"] = peak-flank if np.isfinite(peak) and np.isfinite(flank) else np.nan

    lat_weight = np.cos(np.deg2rad(zlat))
    out["sst_amp_zm_p95_p05_K"] = weighted_quantile(zval, lat_weight, 0.95)-weighted_quantile(zval, lat_weight, 0.05)

    # Fixed equator-relative means.  Half-open boxes prevent the equator from
    # being counted in both hemispheres when a 0 degree grid row is present.
    lat2d = sst.lat.broadcast_like(sst)
    south_mask = ocean_mask & (lat2d >= args.eq_south_lat_range[0]) & (lat2d < args.eq_south_lat_range[1])
    north_mask = ocean_mask & (lat2d >= args.eq_north_lat_range[0]) & (lat2d <= args.eq_north_lat_range[1])
    south = weighted_mean_2d(sst, area_weight, south_mask)
    north = weighted_mean_2d(sst, area_weight, north_mask)
    out["sst_eq_south_mean_K"] = south
    out["sst_eq_north_mean_K"] = north
    out["sst_eq_ns_K"] = north-south if np.isfinite(north) and np.isfinite(south) else np.nan

    # Fixed climatological-center alternatives. These use the same physical SST
    # field as above, but the latitude bands are centered on a reference that is
    # fixed for the calendar month/season across all years.
    for ref_name, prefix in (("tcwv_clim_peak", "sst_tcwvclim"), ("sst_clim_peak", "sst_sstclim")):
        s, n, d = fixed_center_south_north_means(
            sst, static, args, float(reference_centers.get(ref_name, np.nan))
        )
        out[f"{prefix}_south_mean_K"] = s
        out[f"{prefix}_north_mean_K"] = n
        out[f"{prefix}_ns_K"] = d

    # Reference latitudes are saved for provenance but are not treated as
    # interannual predictors.
    out["tcwv_clim_peak_lat_deg"] = float(reference_centers.get("tcwv_clim_peak", np.nan))
    out["sst_clim_peak_lat_deg"] = float(reference_centers.get("sst_clim_peak", np.nan))
    out["centered_sst_half_width_deg"] = float(args.centered_sst_half_width_deg)
    return out



def fixed_center_south_north_means_at_width(
    sst: xr.DataArray,
    static: xr.Dataset,
    args: argparse.Namespace,
    center_lat_deg: float,
    half_width_deg: float,
) -> tuple[float, float, float]:
    """As fixed_center_south_north_means, but with an explicit half-width.

    This helper is used only for meridional-scale sensitivity diagnostics so
    the primary metric definition controlled by --centered-sst-half-width-deg
    remains unchanged.
    """
    if not np.isfinite(center_lat_deg) or not np.isfinite(half_width_deg) or half_width_deg <= 0:
        return np.nan, np.nan, np.nan
    sst = sst.sel(
        lon=slice(args.sst_metric_lon_range[0], args.sst_metric_lon_range[1]),
        lat=slice(args.sst_metric_lat_range[0], args.sst_metric_lat_range[1]),
    )
    lat_min = float(sst.lat.min())
    lat_max = float(sst.lat.max())
    h = float(half_width_deg)
    if center_lat_deg-h < lat_min or center_lat_deg+h > lat_max:
        return np.nan, np.nan, np.nan
    ocean_fraction = static.ocean_fraction.sel(lat=sst.lat, lon=sst.lon)
    ocean_mask = static.ocean_mask.sel(lat=sst.lat, lon=sst.lon)
    coslat = xr.DataArray(np.cos(np.deg2rad(sst.lat)), dims="lat", coords={"lat": sst.lat})
    area_weight = ocean_fraction * coslat
    lat2d = sst.lat.broadcast_like(sst)
    south_mask = ocean_mask & (lat2d >= center_lat_deg-h) & (lat2d < center_lat_deg)
    north_mask = ocean_mask & (lat2d >= center_lat_deg) & (lat2d <= center_lat_deg+h)
    south = weighted_mean_2d(sst, area_weight, south_mask)
    north = weighted_mean_2d(sst, area_weight, north_mask)
    contrast = north-south if np.isfinite(north) and np.isfinite(south) else np.nan
    return south, north, contrast


def sst_scale_sensitivity_for_field(
    sst: xr.DataArray,
    static: xr.Dataset,
    args: argparse.Namespace,
    reference_centers: dict[str, float],
) -> list[dict]:
    """Diagnose SST metrics across a common set of meridional scales.

    Two metric families are returned at every requested scale:
      * peak_to_flank: peak SST and SST at +/- scale from the annual SST peak;
      * centered_bands: south/north SST means in fixed bands of the same
        half-width centered on each fixed reference latitude.

    No extrapolation is performed beyond the prepared latitude domain.
    """
    zlat, zval = zonal_mean_sst_profile(sst, static, args)
    peak_lat = peak_latitude_from_profile(zlat, zval, args.sst_peak_lat_range)
    peak = interp_scalar(zlat, zval, peak_lat) if np.isfinite(peak_lat) else np.nan
    rows: list[dict] = []
    for scale in sorted(set(float(x) for x in args.sst_scale_sweep_deg)):
        south_flank = interp_scalar(zlat, zval, peak_lat-scale) if np.isfinite(peak_lat) else np.nan
        north_flank = interp_scalar(zlat, zval, peak_lat+scale) if np.isfinite(peak_lat) else np.nan
        if np.isfinite(south_flank) and np.isfinite(north_flank):
            flank_mean = 0.5*(south_flank+north_flank)
            contrast = peak-flank_mean if np.isfinite(peak) else np.nan
        else:
            flank_mean = np.nan
            contrast = np.nan
        rows.append({
            "metric_family": "peak_to_flank",
            "reference": "annual_sst_peak",
            "scale_deg": scale,
            "center_lat_deg": peak_lat,
            "peak_sst_K": peak,
            "south_value_K": south_flank,
            "north_value_K": north_flank,
            "contrast_K": contrast,
        })
        for ref_name in ("tcwv_clim_peak", "sst_clim_peak", "equator"):
            center = float(reference_centers.get(ref_name, np.nan))
            south, north, ns = fixed_center_south_north_means_at_width(
                sst, static, args, center, scale
            )
            rows.append({
                "metric_family": "centered_bands",
                "reference": ref_name,
                "scale_deg": scale,
                "center_lat_deg": center,
                "peak_sst_K": peak,
                "south_value_K": south,
                "north_value_K": north,
                "contrast_K": ns,
            })
    return rows

def _period_sort_key(period_type: str, period: object) -> tuple[int, int]:
    """Return a stable scientific ordering for calendar months and seasons."""
    if period_type == "monthly":
        return 0, int(period)
    season_order = {"DJF": 1, "MAM": 2, "JJA": 3, "SON": 4}
    return 1, season_order.get(str(period), 99)


def select_targets(
    summary: pd.DataFrame,
    selected_shape: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Read shape targets from Stage 2 and optionally add shift targets.

    Shape targets come directly from selected_shape_modes.csv. Unless --shape-only
    is set, the highest-variance mode classified as shift is also included for each
    period and series type. Mixed modes are not used as shift targets.
    """
    rows: list[dict] = []

    shape = selected_shape[
        selected_shape["series_type"].isin(args.series_types)
        & (selected_shape["selection_status"] == "selected_shape")
        & selected_shape["selected_mode"].notna()
    ].copy()

    for _, chosen in shape.iterrows():
        period_type = str(chosen["period_type"])
        period = int(chosen["period"]) if period_type == "monthly" else str(chosen["period"])
        mode = int(chosen["selected_mode"])
        series_type = str(chosen["series_type"])

        # Require the selection table and summary to identify the same Stage 2 mode.
        match = summary[
            (summary["period_type"] == period_type)
            & (summary["period"].astype(str) == str(period))
            & (summary["series_type"] == series_type)
            & (summary["mode_family"] == "total_abs")
            & (summary["mode"].astype(int) == mode)
        ]
        if match.empty:
            raise ValueError(
                f"Selected shape mode not found in tcwv_mode_summary.csv: "
                f"{period_type} {period} {series_type} EOF{mode}"
            )
        detail = match.iloc[0]
        if str(detail["mode_type"]) != "shape":
            raise ValueError(
                f"Stage-2 selection is inconsistent: {period_type} {period} "
                f"{series_type} EOF{mode} is {detail['mode_type']!r}, not 'shape'."
            )

        rows.append({
            "target": f"{period_type}_{period}_{series_type}_selected_shape_eof{mode}",
            "period_type": period_type,
            "period": period,
            "series_type": series_type,
            "role": "selected_shape",
            "mode": mode,
            "mode_type": "shape",
            "selection_source": "selected_shape_modes.csv",
            "selection_status": str(chosen["selection_status"]),
            "explained_variance_fraction": float(chosen["explained_variance_fraction"]),
            "shift_score": float(chosen["shift_score"]),
            "shape_score": float(chosen["shape_score"]),
        })

    if not args.shape_only:
        shift = summary[
            (summary["mode_family"] == "total_abs")
            & (summary["mode_type"] == "shift")
            & summary["series_type"].isin(args.series_types)
        ].copy()
        for (period_type, period, series_type), group in shift.groupby(
            ["period_type", "period", "series_type"], sort=False
        ):
            valid = group[np.isfinite(group["explained_variance_fraction"])].copy()
            if valid.empty:
                continue
            chosen = valid.sort_values(
                ["explained_variance_fraction", "shift_score", "mode"],
                ascending=[False, False, True],
            ).iloc[0]
            period_value = int(period) if str(period_type) == "monthly" else str(period)
            mode = int(chosen["mode"])
            rows.append({
                "target": f"{period_type}_{period_value}_{series_type}_selected_shift_eof{mode}",
                "period_type": str(period_type),
                "period": period_value,
                "series_type": str(series_type),
                "role": "selected_shift",
                "mode": mode,
                "mode_type": "shift",
                "selection_source": "tcwv_mode_summary.csv",
                "selection_status": "selected_shift",
                "explained_variance_fraction": float(chosen["explained_variance_fraction"]),
                "shift_score": float(chosen["shift_score"]),
                "shape_score": float(chosen["shape_score"]),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No Stage-3 targets were selected.")

    role_order = {"selected_shape": 0, "selected_shift": 1}
    order = [
        (*_period_sort_key(str(t), p), role_order.get(str(r), 99), str(st))
        for t, p, r, st in zip(
            out["period_type"], out["period"], out["role"], out["series_type"]
        )
    ]
    out["_sort"] = order
    out = out.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
    return out


def get_period_field(ds: xr.Dataset, period_type: str, period, variable: str) -> xr.DataArray:
    if period_type == "monthly":
        return ds[variable].sel(month=int(period)).transpose("year", "lat", "lon")
    return ds[variable].sel(season=str(period)).transpose("year", "lat", "lon")


def get_target_pc(modes_monthly: xr.Dataset, modes_seasonal: xr.Dataset, target: pd.Series) -> xr.DataArray:
    source = modes_monthly if target.period_type == "monthly" else modes_seasonal
    selector = {"series_type": target.series_type, "mode": int(target["mode"])}
    selector["month" if target.period_type == "monthly" else "season"] = int(target.period) if target.period_type == "monthly" else str(target.period)
    return source.total_abs_pc.sel(**selector).rename("pc")


def convergence_from_uv(u: xr.DataArray, v: xr.DataArray) -> xr.Dataset:
    u = u.transpose("year", "lat", "lon")
    v = v.transpose("year", "lat", "lon")
    lat = np.asarray(u.lat.values, dtype=float)
    lon = np.asarray(u.lon.values, dtype=float)
    U = np.asarray(u.values, dtype=float)
    V = np.asarray(v.values, dtype=float)
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    dudlambda = np.gradient(U, lon_rad, axis=2)
    dvdphi = np.gradient(V, lat_rad, axis=1)
    coslat = np.cos(lat_rad)
    dudx = dudlambda/(EARTH_RADIUS_M*coslat[None, :, None])
    dvdy = dvdphi/EARTH_RADIUS_M
    neg_dudx = -dudx*SECONDS_PER_DAY
    neg_dvdy = -dvdy*SECONDS_PER_DAY
    coords = {"year": u.year, "lat": u.lat, "lon": u.lon}
    dims = ("year", "lat", "lon")
    return xr.Dataset({
        "neg_dudx": xr.DataArray(neg_dudx.astype("float32"), dims=dims, coords=coords, attrs={"units": "day-1"}),
        "neg_dvdy": xr.DataArray(neg_dvdy.astype("float32"), dims=dims, coords=coords, attrs={"units": "day-1"}),
        "convergence": xr.DataArray((neg_dudx+neg_dvdy).astype("float32"), dims=dims, coords=coords, attrs={"units": "day-1"}),
    })


def regress_field(field: xr.DataArray, pc: xr.DataArray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Regress a field onto a standardized PC, vectorized over grid points."""
    years = np.intersect1d(field.year.values.astype(int), pc.year.values.astype(int))
    y = np.asarray(field.sel(year=years).transpose("year", "lat", "lon").values, dtype=float)
    x = np.asarray(pc.sel(year=years).values, dtype=float)
    valid_year = np.isfinite(x)
    x = x[valid_year]
    y = y[valid_year]
    shape = y.shape[1:]
    if x.size < 3:
        nan = np.full(shape, np.nan)
        return nan, nan, nan, np.zeros(shape, dtype=np.int16)
    x = standardize(x)
    yf = y.reshape((y.shape[0], -1))
    good = np.isfinite(yf) & np.isfinite(x[:, None])
    n = good.sum(axis=0).astype(float)
    xx = np.where(good, x[:, None], 0.0)
    yy = np.where(good, yf, 0.0)
    sumx = xx.sum(axis=0)
    sumy = yy.sum(axis=0)
    meanx = np.divide(sumx, n, out=np.full_like(sumx, np.nan), where=n>0)
    meany = np.divide(sumy, n, out=np.full_like(sumy, np.nan), where=n>0)
    dx = np.where(good, x[:, None]-meanx[None, :], 0.0)
    dy = np.where(good, yf-meany[None, :], 0.0)
    sxx = (dx*dx).sum(axis=0)
    syy = (dy*dy).sum(axis=0)
    sxy = (dx*dy).sum(axis=0)
    slope = np.divide(sxy, sxx, out=np.full_like(sxy, np.nan), where=(n>=3) & (sxx>0))
    r = np.divide(sxy, np.sqrt(sxx*syy), out=np.full_like(sxy, np.nan), where=(n>=3) & (sxx>0) & (syy>0))
    p = np.full_like(r, np.nan)
    if student_t is not None:
        ok = (n>2) & np.isfinite(r) & (np.abs(r)<1)
        tval = np.zeros_like(r)
        tval[ok] = r[ok]*np.sqrt((n[ok]-2)/(1-r[ok]**2))
        p[ok] = 2*student_t.sf(np.abs(tval[ok]), df=n[ok]-2)
        p[(n>2) & np.isfinite(r) & (np.abs(r)>=1)] = 0.0
    return slope.reshape(shape), r.reshape(shape), p.reshape(shape), n.astype(np.int16).reshape(shape)


def composite_field(field: xr.DataArray, pc: xr.DataArray, top_n: int, bottom_n: int) -> tuple[np.ndarray, list[int], list[int]]:
    years = np.intersect1d(field.year.values.astype(int), pc.year.values.astype(int))
    p = np.asarray(pc.sel(year=years).values, dtype=float)
    good = np.isfinite(p)
    years = years[good]
    p = p[good]
    if years.size < max(top_n, bottom_n)+2:
        shape = (len(COMPOSITES), field.sizes["lat"], field.sizes["lon"])
        return np.full(shape, np.nan), [], []
    order = np.argsort(p)
    low_years = years[order[:bottom_n]].tolist()
    high_years = years[order[-top_n:]].tolist()
    state = field.sel(year=years)
    clim = state.mean("year", skipna=True)
    anom = state-clim
    high_anom = anom.sel(year=high_years).mean("year", skipna=True)
    low_anom = anom.sel(year=low_years).mean("year", skipna=True)
    high_state = state.sel(year=high_years).mean("year", skipna=True)
    low_state = state.sel(year=low_years).mean("year", skipna=True)
    arrays = [high_anom, low_anom, high_anom-low_anom, high_state, low_state]
    return np.stack([np.asarray(a.values, dtype=np.float32) for a in arrays]), high_years, low_years


def multiple_r2(y: np.ndarray, predictors: Sequence[np.ndarray]) -> tuple[float, int, list[float]]:
    yz = standardize(y)
    cols = [standardize(x) for x in predictors]
    good = np.isfinite(yz)
    for x in cols:
        good &= np.isfinite(x)
    n = int(good.sum())
    if n < len(cols)+3:
        return np.nan, n, [np.nan]*len(cols)
    X = np.column_stack([x[good] for x in cols])
    beta = np.linalg.lstsq(X, yz[good], rcond=None)[0]
    fit = X@beta
    ssr = np.sum((yz[good]-fit)**2)
    sst = np.sum((yz[good]-np.mean(yz[good]))**2)
    return float(1-ssr/sst), n, [float(b) for b in beta]



def partial_corr_control(x: np.ndarray, y: np.ndarray, controls: Sequence[np.ndarray]) -> tuple[float, int]:
    """Partial correlation between x and y after linearly removing controls."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    c = [np.asarray(v, dtype=float) for v in controls]
    good = np.isfinite(x) & np.isfinite(y)
    for v in c:
        good &= np.isfinite(v)
    n = int(good.sum())
    if n < len(c) + 4:
        return np.nan, n
    xv = x[good]
    yv = y[good]
    if not c:
        r, _, _ = safe_corr(xv, yv)
        return r, n
    C = np.column_stack([np.ones(n)] + [v[good] for v in c])
    bx = np.linalg.lstsq(C, xv, rcond=None)[0]
    by = np.linalg.lstsq(C, yv, rcond=None)[0]
    rx = xv - C @ bx
    ry = yv - C @ by
    if np.std(rx, ddof=1) <= 0 or np.std(ry, ddof=1) <= 0:
        return np.nan, n
    return float(np.corrcoef(rx, ry)[0, 1]), n


def peak_flank_statistics(pc: np.ndarray, peak: np.ndarray, contrast: np.ndarray) -> dict[str, float]:
    """Statistics separating peak-SST and peak-to-flank contrast information."""
    pc = np.asarray(pc, dtype=float)
    peak = np.asarray(peak, dtype=float)
    contrast = np.asarray(contrast, dtype=float)
    good = np.isfinite(pc) & np.isfinite(peak) & np.isfinite(contrast)
    pc, peak, contrast = pc[good], peak[good], contrast[good]
    n = int(pc.size)
    if n < 5:
        return {
            "n_joint": n, "r_peak": np.nan, "r_contrast_jointsample": np.nan,
            "r2_peak": np.nan, "r2_contrast": np.nan, "r2_joint": np.nan,
            "delta_r2_contrast_given_peak": np.nan,
            "partial_r_contrast_given_peak": np.nan,
            "beta_peak_std": np.nan, "beta_contrast_std": np.nan,
        }
    r_peak, _, _ = safe_corr(pc, peak)
    r_contrast, _, _ = safe_corr(pc, contrast)
    r2_joint, _, beta = multiple_r2(pc, [peak, contrast])
    partial_r, _ = partial_corr_control(pc, contrast, [peak])
    r2_peak = r_peak*r_peak if np.isfinite(r_peak) else np.nan
    r2_contrast = r_contrast*r_contrast if np.isfinite(r_contrast) else np.nan
    delta = r2_joint-r2_peak if np.isfinite(r2_joint) and np.isfinite(r2_peak) else np.nan
    return {
        "n_joint": n,
        "r_peak": r_peak,
        "r_contrast_jointsample": r_contrast,
        "r2_peak": r2_peak,
        "r2_contrast": r2_contrast,
        "r2_joint": r2_joint,
        "delta_r2_contrast_given_peak": delta,
        "partial_r_contrast_given_peak": partial_r,
        "beta_peak_std": beta[0] if len(beta) > 0 else np.nan,
        "beta_contrast_std": beta[1] if len(beta) > 1 else np.nan,
    }


def bootstrap_peak_flank_statistics(
    pc: np.ndarray,
    peak: np.ndarray,
    contrast: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """Year-resampling percentile CIs for peak/flank scale diagnostics."""
    pc = np.asarray(pc, dtype=float)
    peak = np.asarray(peak, dtype=float)
    contrast = np.asarray(contrast, dtype=float)
    good = np.isfinite(pc) & np.isfinite(peak) & np.isfinite(contrast)
    pc, peak, contrast = pc[good], peak[good], contrast[good]
    metrics = [
        "r_peak", "r_contrast_jointsample", "r2_peak", "r2_contrast", "r2_joint",
        "delta_r2_contrast_given_peak", "partial_r_contrast_given_peak",
        "beta_peak_std", "beta_contrast_std",
    ]
    out = {}
    if n_boot <= 0 or pc.size < 5:
        for m in metrics:
            out[f"{m}_ci_low"] = np.nan
            out[f"{m}_ci_high"] = np.nan
        out["bootstrap_samples"] = int(max(n_boot, 0))
        return out
    rng = np.random.default_rng(seed)
    store = {m: np.full(n_boot, np.nan, dtype=float) for m in metrics}
    n = pc.size
    for ib in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stat = peak_flank_statistics(pc[idx], peak[idx], contrast[idx])
        for m in metrics:
            store[m][ib] = stat[m]
    for m in metrics:
        vals = store[m][np.isfinite(store[m])]
        if vals.size:
            out[f"{m}_ci_low"], out[f"{m}_ci_high"] = [float(x) for x in np.percentile(vals, [2.5, 97.5])]
        else:
            out[f"{m}_ci_low"] = np.nan
            out[f"{m}_ci_high"] = np.nan
    out["bootstrap_samples"] = int(n_boot)
    return out


def two_predictor_statistics(
    pc: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
) -> dict[str, float]:
    """Statistics for a generic two-coordinate SST state space."""
    pc = np.asarray(pc, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    good = np.isfinite(pc) & np.isfinite(x1) & np.isfinite(x2)
    pc, x1, x2 = pc[good], x1[good], x2[good]
    n = int(pc.size)
    if n < 5:
        return {
            "n_joint": n, "r_x1": np.nan, "r_x2": np.nan,
            "r_asym": np.nan, "r_level": np.nan,
            "r_contrast": np.nan, "r_mean": np.nan,
            "r2_x1": np.nan, "r2_x2": np.nan,
            "r2_asym": np.nan, "r2_level": np.nan,
            "r2_contrast": np.nan, "r2_mean": np.nan,
            "r2_joint": np.nan, "beta_x1_std": np.nan, "beta_x2_std": np.nan,
        }
    r1, _, _ = safe_corr(pc, x1)
    r2, _, _ = safe_corr(pc, x2)
    # Symmetric coordinates about the fixed reference latitude.
    # T_level measures the SST level in the two flanking bands; T_asym is the
    # signed north-south half-difference, so x2=T_level+T_asym and
    # x1=T_level-T_asym.  The factor 1/2 does not affect correlations or R2.
    sst_asym = 0.5*(x2-x1)
    sst_level = 0.5*(x1+x2)
    ra, _, _ = safe_corr(pc, sst_asym)
    rl, _, _ = safe_corr(pc, sst_level)
    r2joint, _, beta = multiple_r2(pc, [x1, x2])
    return {
        "n_joint": n,
        "r_x1": r1,
        "r_x2": r2,
        "r_asym": ra,
        "r_level": rl,
        "r2_x1": r1*r1 if np.isfinite(r1) else np.nan,
        "r2_x2": r2*r2 if np.isfinite(r2) else np.nan,
        "r2_asym": ra*ra if np.isfinite(ra) else np.nan,
        "r2_level": rl*rl if np.isfinite(rl) else np.nan,
        "r2_joint": r2joint,
        "beta_x1_std": beta[0] if len(beta) > 0 else np.nan,
        "beta_x2_std": beta[1] if len(beta) > 1 else np.nan,
        # Backward-compatible aliases for older plotting/analysis code.
        "r_contrast": ra,
        "r_mean": rl,
        "r2_contrast": ra*ra if np.isfinite(ra) else np.nan,
        "r2_mean": rl*rl if np.isfinite(rl) else np.nan,
    }


def bootstrap_two_predictor_statistics(
    pc: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """Year-resampling percentile CIs for a generic two-coordinate SST space."""
    pc = np.asarray(pc, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    good = np.isfinite(pc) & np.isfinite(x1) & np.isfinite(x2)
    pc, x1, x2 = pc[good], x1[good], x2[good]
    metrics = [
        "r_x1", "r_x2", "r_asym", "r_level", "r_contrast", "r_mean",
        "r2_x1", "r2_x2", "r2_asym", "r2_level", "r2_contrast", "r2_mean", "r2_joint",
        "beta_x1_std", "beta_x2_std",
    ]
    out: dict[str, float] = {}
    if n_boot <= 0 or pc.size < 5:
        for m in metrics:
            out[f"{m}_ci_low"] = np.nan
            out[f"{m}_ci_high"] = np.nan
        out["bootstrap_samples"] = int(max(n_boot, 0))
        return out
    rng = np.random.default_rng(seed)
    store = {m: np.full(n_boot, np.nan, dtype=float) for m in metrics}
    n = pc.size
    for ib in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stat = two_predictor_statistics(pc[idx], x1[idx], x2[idx])
        for m in metrics:
            store[m][ib] = stat[m]
    for m in metrics:
        vals = store[m][np.isfinite(store[m])]
        if vals.size:
            lo, hi = np.percentile(vals, [2.5, 97.5])
            out[f"{m}_ci_low"] = float(lo)
            out[f"{m}_ci_high"] = float(hi)
        else:
            out[f"{m}_ci_low"] = np.nan
            out[f"{m}_ci_high"] = np.nan
    out["bootstrap_samples"] = int(n_boot)
    return out


def exhaustive_seasonal_sst_diagnostics(
    seasonal: xr.Dataset,
    static: xr.Dataset,
    modes_seasonal: xr.Dataset,
    summary: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate seasonal EOF1/EOF2 in two SST coordinate systems.

    tcwv_centered_ns uses south and north SST means around the fixed TCWV
    climatological maximum. peak_contrast uses the annual SST peak and its
    peak-to-flank contrast. Both use the same width sweep for all four seasons,
    independently of the selected shape/shift targets.
    """
    seasons = ["DJF", "MAM", "JJA", "SON"]
    scales = sorted(set(float(v) for v in args.sst_scale_sweep_deg))
    metric_rows: list[dict] = []
    relation_rows: list[dict] = []

    valid_series = [s for s in args.exhaustive_sst_series_types if s in set(summary["series_type"].astype(str))]
    if not valid_series:
        raise ValueError("No --exhaustive-sst-series-types are present in tcwv_mode_summary.csv")

    for season_index, season in enumerate(seasons):
        ref_centers = fixed_reference_centers(
            seasonal, modes_seasonal, modes_seasonal, "seasonal", season, static, args
        )
        sst_field = get_period_field(seasonal, "seasonal", season, "sst")
        base_rows: list[dict] = []
        for year in np.asarray(sst_field.year.values, dtype=int):
            field_y = sst_field.sel(year=int(year))
            for row in sst_scale_sensitivity_for_field(field_y, static, args, ref_centers):
                # Exhaustive comparison uses the physically motivated TCWV-climatology
                # center for the north/south coordinate system and annual SST peak for
                # the peak/contrast coordinate system.
                if row["metric_family"] == "centered_bands" and row["reference"] != "tcwv_clim_peak":
                    continue
                row = dict(row)
                row["year"] = int(year)
                row["season"] = season
                base_rows.append(row)
        base = pd.DataFrame(base_rows)

        for series_type in valid_series:
            work = base.copy()
            if series_type == "linear_detrended" and not work.empty:
                for _, inds in work.groupby(["metric_family", "reference", "scale_deg"]).groups.items():
                    inds = list(inds)
                    yrs = work.loc[inds, "year"].to_numpy(float)
                    for col in ("peak_sst_K", "south_value_K", "north_value_K", "contrast_K"):
                        work.loc[inds, col] = detrend_vector(
                            yrs, work.loc[inds, col].to_numpy(float), restore_mean=True
                        )

            for mode in (1, 2):
                pc = modes_seasonal.total_abs_pc.sel(
                    series_type=series_type, season=season, mode=mode
                ).rename("pc")
                pc_df = pd.DataFrame({"year": pc.year.values.astype(int), "pc": np.asarray(pc.values, dtype=float)})
                detail = summary[
                    (summary["period_type"] == "seasonal")
                    & (summary["period"].astype(str) == season)
                    & (summary["series_type"] == series_type)
                    & (summary["mode_family"] == "total_abs")
                    & (summary["mode"].astype(int) == mode)
                ]
                mode_type = str(detail.iloc[0]["mode_type"]) if not detail.empty else "unknown"
                evf = float(detail.iloc[0]["explained_variance_fraction"]) if not detail.empty else np.nan

                for (family, ref, scale), g in work.groupby(["metric_family", "reference", "scale_deg"], sort=True):
                    joined = pd.merge(pc_df, g, on="year", how="inner")
                    if family == "centered_bands":
                        space = "tcwv_centered_ns"
                        x1 = joined["south_value_K"].to_numpy(float)
                        x2 = joined["north_value_K"].to_numpy(float)
                        stat = two_predictor_statistics(joined.pc.to_numpy(float), x1, x2)
                        group_seed = (
                            int(args.sst_scale_bootstrap_seed)
                            + season_index*100000 + mode*10000 + int(round(float(scale)*100))
                            + (0 if series_type == "raw" else 500000)
                        )
                        ci = bootstrap_two_predictor_statistics(
                            joined.pc.to_numpy(float), x1, x2,
                            int(args.sst_scale_bootstrap_samples), group_seed
                        )
                        sst_level = 0.5*(x1+x2)
                        sst_asym = 0.5*(x2-x1)
                        peak_same = joined["peak_sst_K"].to_numpy(float)
                        r_peak_level, _, _ = safe_corr(peak_same, sst_level)
                        r_pc_peak, _, _ = safe_corr(joined.pc.to_numpy(float), peak_same)
                        row = {
                            "season": season, "mode": mode, "mode_type": mode_type,
                            "explained_variance_fraction": evf, "series_type": series_type,
                            "sst_space": space, "scale_deg": float(scale),
                            "scale_definition": "half-width of south/north SST bands around climatological TCWV peak",
                            "reference": ref,
                            "center_lat_deg": float(ref_centers.get("tcwv_clim_peak", np.nan)),
                            **stat, **ci,
                            # Direct bridge to the idealized peak-SST control parameter.
                            "r_peak_level": r_peak_level,
                            "r2_peak_level": r_peak_level*r_peak_level if np.isfinite(r_peak_level) else np.nan,
                            "r_pc_peak_same_sample": r_pc_peak,
                            "r2_pc_peak_same_sample": r_pc_peak*r_pc_peak if np.isfinite(r_pc_peak) else np.nan,
                            "r_peak": np.nan, "r_contrast_jointsample": np.nan,
                            "r2_peak": np.nan,
                            "delta_r2_contrast_given_peak": np.nan,
                            "partial_r_contrast_given_peak": np.nan,
                            "beta_peak_std": np.nan, "beta_contrast_std": np.nan,
                        }
                    else:
                        space = "peak_contrast"
                        peak = joined["peak_sst_K"].to_numpy(float)
                        contrast = joined["contrast_K"].to_numpy(float)
                        stat_pc = peak_flank_statistics(joined.pc.to_numpy(float), peak, contrast)
                        group_seed = (
                            int(args.sst_scale_bootstrap_seed) + 2000000
                            + season_index*100000 + mode*10000 + int(round(float(scale)*100))
                            + (0 if series_type == "raw" else 500000)
                        )
                        ci_pc = bootstrap_peak_flank_statistics(
                            joined.pc.to_numpy(float), peak, contrast,
                            int(args.sst_scale_bootstrap_samples), group_seed
                        )
                        row = {
                            "season": season, "mode": mode, "mode_type": mode_type,
                            "explained_variance_fraction": evf, "series_type": series_type,
                            "sst_space": space, "scale_deg": float(scale),
                            "scale_definition": "distance from annual zonal-mean SST peak to each flank",
                            "reference": ref,
                            "center_lat_deg": np.nan,
                            "r_asym": np.nan, "r_level": np.nan,
                            "r2_asym": np.nan, "r2_level": np.nan,
                            "r_peak_level": np.nan, "r2_peak_level": np.nan,
                            "r_pc_peak_same_sample": np.nan, "r2_pc_peak_same_sample": np.nan,
                            "r_x1": stat_pc["r_peak"],
                            "r_x2": stat_pc["r_contrast_jointsample"],
                            "r_contrast": stat_pc["r_contrast_jointsample"],
                            "r2_x1": stat_pc["r2_peak"],
                            "r2_x2": stat_pc["r2_contrast"],
                            "beta_x1_std": stat_pc["beta_peak_std"],
                            "beta_x2_std": stat_pc["beta_contrast_std"],
                            **stat_pc, **ci_pc,
                        }
                    relation_rows.append(row)

                    for _, rr in joined.iterrows():
                        metric_rows.append({
                            "season": season, "mode": mode, "mode_type": mode_type,
                            "explained_variance_fraction": evf, "series_type": series_type,
                            "sst_space": space, "scale_deg": float(scale), "year": int(rr.year),
                            "pc": float(rr.pc),
                            "x1_K": float(rr.south_value_K) if family == "centered_bands" else float(rr.peak_sst_K),
                            "x2_K": float(rr.north_value_K) if family == "centered_bands" else float(rr.contrast_K),
                            "south_sst_K": float(rr.south_value_K) if family == "centered_bands" else np.nan,
                            "north_sst_K": float(rr.north_value_K) if family == "centered_bands" else np.nan,
                            "sst_level_K": 0.5*float(rr.south_value_K + rr.north_value_K) if family == "centered_bands" else np.nan,
                            "sst_asym_K": 0.5*float(rr.north_value_K - rr.south_value_K) if family == "centered_bands" else np.nan,
                            "peak_sst_K": float(rr.peak_sst_K),
                            "contrast_K": float(rr.contrast_K),
                        })
    return pd.DataFrame(metric_rows), pd.DataFrame(relation_rows)


def plot_exhaustive_scale_sensitivity(rel: pd.DataFrame, outdir: Path, series_type: str) -> None:
    """4x2 grid: R2 versus scale for both SST spaces in every season and PC."""
    if plt is None:
        return
    d = rel[rel["series_type"] == series_type].copy()
    if d.empty:
        return
    seasons = ["DJF", "MAM", "JJA", "SON"]
    fig, axes = plt.subplots(4, 2, figsize=(9.0, 11.0), sharex=True, sharey=True)
    for i, season in enumerate(seasons):
        for j, mode in enumerate((1, 2)):
            ax = axes[i, j]
            for space, label in (("tcwv_centered_ns", "south and north SST predictors"), ("peak_contrast", "peak SST+contrast")):
                g = d[(d.season == season) & (d["mode"] == mode) & (d.sst_space == space)].sort_values("scale_deg")
                if not g.empty:
                    ax.plot(g.scale_deg, g.r2_joint, marker="o", markersize=3, label=label)
                    if space == "peak_contrast" and "r2_peak" in g:
                        ax.plot(g.scale_deg, g.r2_peak, linestyle="--", linewidth=1.2, label="peak SST only")
            ax.set_title(f"{season} EOF{mode}")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
            if i == 3:
                ax.set_xlabel("Meridional scale (deg)")
            if j == 0:
                ax.set_ylabel(r"$R^2$")
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Seasonal PC-SST state-space sensitivity ({series_type})")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outdir/f"exhaustive_sst_scale_sensitivity_{series_type}.pdf", bbox_inches="tight")
    fig.savefig(outdir/f"exhaustive_sst_scale_sensitivity_{series_type}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_exhaustive_contrast_correlation(rel: pd.DataFrame, outdir: Path, series_type: str) -> None:
    """4x2 grid: PC correlation with the contrast coordinate versus scale."""
    if plt is None:
        return
    d = rel[rel["series_type"] == series_type].copy()
    if d.empty:
        return
    seasons = ["DJF", "MAM", "JJA", "SON"]
    fig, axes = plt.subplots(4, 2, figsize=(9.0, 11.0), sharex=True, sharey=True)
    for i, season in enumerate(seasons):
        for j, mode in enumerate((1, 2)):
            ax = axes[i, j]
            g_ns = d[(d.season == season) & (d["mode"] == mode) & (d.sst_space == "tcwv_centered_ns")].sort_values("scale_deg")
            g_pc = d[(d.season == season) & (d["mode"] == mode) & (d.sst_space == "peak_contrast")].sort_values("scale_deg")
            if not g_ns.empty:
                ax.plot(g_ns.scale_deg, g_ns.r_contrast, marker="o", markersize=3, label="north-south contrast")
            if not g_pc.empty:
                ax.plot(g_pc.scale_deg, g_pc.r_contrast, marker="o", markersize=3, label="peak-to-flank contrast")
            ax.axhline(0.0, linewidth=0.7, color="0.5")
            ax.set_title(f"{season} EOF{mode}")
            ax.set_ylim(-1, 1)
            ax.grid(alpha=0.25)
            if i == 3:
                ax.set_xlabel("Meridional scale (deg)")
            if j == 0:
                ax.set_ylabel("Pearson r")
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"PC correlation with SST contrast coordinate ({series_type})")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outdir/f"exhaustive_sst_contrast_correlation_{series_type}.pdf", bbox_inches="tight")
    fig.savefig(outdir/f"exhaustive_sst_contrast_correlation_{series_type}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_exhaustive_ns_decomposition(rel: pd.DataFrame, outdir: Path, series_type: str) -> None:
    """4x2 grid: decompose the two-sided SST predictors into mean and N-S contrast.

    For the TCWV-centered south/north bands, the two-predictor model spans the
    same linear space as the pair [mean SST, north-minus-south SST contrast].
    Plotting the single-predictor R2 values therefore shows whether the joint
    skill is associated primarily with background SST level or meridional
    contrast.
    """
    if plt is None:
        return
    d = rel[(rel["series_type"] == series_type) & (rel["sst_space"] == "tcwv_centered_ns")].copy()
    if d.empty:
        return
    seasons = ["DJF", "MAM", "JJA", "SON"]
    fig, axes = plt.subplots(4, 2, figsize=(9.0, 11.0), sharex=True, sharey=True)
    for i, season in enumerate(seasons):
        for j, mode in enumerate((1, 2)):
            ax = axes[i, j]
            g = d[(d.season == season) & (d["mode"] == mode)].sort_values("scale_deg")
            if not g.empty:
                ax.plot(g.scale_deg, g.r2_joint, marker="o", markersize=3, label="south+north predictors")
                ax.plot(g.scale_deg, g.r2_contrast, marker="o", markersize=3, label=r"SST asymmetry $T_{asym}$ only")
                ax.plot(g.scale_deg, g.r2_mean, marker="o", markersize=3, label=r"rainbelt-centered SST level $T_{level}$ only")
            ax.set_title(f"{season} EOF{mode}")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
            if i == 3:
                ax.set_xlabel("Band half-width (deg)")
            if j == 0:
                ax.set_ylabel(r"$R^2$")
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Decomposition of TCWV-centered south/north SST predictors ({series_type})")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outdir/f"exhaustive_sst_ns_decomposition_{series_type}.pdf", bbox_inches="tight")
    fig.savefig(outdir/f"exhaustive_sst_ns_decomposition_{series_type}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_exhaustive_state_spaces(metrics: pd.DataFrame, rel: pd.DataFrame, outdir: Path, series_type: str) -> None:
    """For every tested scale, draw all 16 seasonal PC x SST-space state planes."""
    if plt is None:
        return
    md = metrics[metrics["series_type"] == series_type].copy()
    rd = rel[rel["series_type"] == series_type].copy()
    if md.empty:
        return
    seasons = ["DJF", "MAM", "JJA", "SON"]
    for scale in sorted(md.scale_deg.dropna().unique()):
        fig, axes = plt.subplots(4, 4, figsize=(13.0, 11.0))
        for i, season in enumerate(seasons):
            col = 0
            for mode in (1, 2):
                for space in ("tcwv_centered_ns", "peak_contrast"):
                    ax = axes[i, col]
                    col += 1
                    g = md[(md.season == season) & (md["mode"] == mode) & (md.sst_space == space) & np.isclose(md.scale_deg, scale)]
                    r = rd[(rd.season == season) & (rd["mode"] == mode) & (rd.sst_space == space) & np.isclose(rd.scale_deg, scale)]
                    if not g.empty:
                        pcstd = standardize(g.pc.to_numpy(float))
                        sc = ax.scatter(g.x1_K, g.x2_K, c=pcstd, cmap="coolwarm", vmin=-2.5, vmax=2.5, s=20, edgecolors="none")
                    if space == "tcwv_centered_ns":
                        ax.set_xlabel("South SST (K)")
                        ax.set_ylabel("North SST (K)")
                    else:
                        ax.set_xlabel("Peak SST (K)")
                        ax.set_ylabel("Peak-to-flank contrast (K)")
                    mode_type = str(g.mode_type.iloc[0]) if not g.empty else ""
                    r2txt = f"R2={float(r.r2_joint.iloc[0]):.2f}" if not r.empty and np.isfinite(r.r2_joint.iloc[0]) else "R2=NA"
                    ax.set_title(f"{season} EOF{mode} {mode_type}\n{r2txt}", fontsize=9)
                    ax.grid(alpha=0.2)
        fig.suptitle(f"All seasonal SST state spaces: scale={scale:g} deg ({series_type})")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        tag = str(scale).replace(".", "p")
        fig.savefig(outdir/f"exhaustive_sst_state_spaces_{series_type}_{tag}deg.pdf", bbox_inches="tight")
        fig.savefig(outdir/f"exhaustive_sst_state_spaces_{series_type}_{tag}deg.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    prepare = Path(args.prepare_dir)
    modes_dir = Path(args.modes_dir)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    monthly = xr.open_dataset(prepare/"era5_monthly.nc")
    seasonal = xr.open_dataset(prepare/"era5_seasonal.nc")
    static = xr.open_dataset(prepare/"static_fields.nc")
    modes_monthly = xr.open_dataset(modes_dir/"tcwv_modes_monthly.nc")
    modes_seasonal = xr.open_dataset(modes_dir/"tcwv_modes_seasonal.nc")
    summary = pd.read_csv(modes_dir/"tcwv_mode_summary.csv")
    selected_shape_path = modes_dir/"selected_shape_modes.csv"
    if not selected_shape_path.exists():
        raise FileNotFoundError(
            f"Missing canonical shape-selection table: {selected_shape_path}. "
            "Rerun Stage 2 with atlantic_itcz_tcwv_modes.py >= 1.1.0."
        )
    selected_shape = pd.read_csv(selected_shape_path)
    targets = select_targets(summary, selected_shape, args)
    targets.to_csv(outdir/"target_table.csv", index=False)

    map_lon = monthly.lon.sel(lon=slice(args.analysis_lon_range[0], args.analysis_lon_range[1]))
    map_lat = monthly.lat.sel(lat=slice(args.analysis_lat_range[0], args.analysis_lat_range[1]))
    map_vars = [v for v in args.map_vars if v in monthly.data_vars and v in seasonal.data_vars]
    variables = map_vars + list(DERIVED_MAP_VARIABLES)

    ntarget = len(targets)
    nvar = len(variables)
    nlat, nlon = map_lat.size, map_lon.size
    slopes = np.full((ntarget, nvar, nlat, nlon), np.nan, np.float32)
    corrmaps = np.full_like(slopes, np.nan)
    pmaps = np.full_like(slopes, np.nan)
    nmap = np.zeros((ntarget, nvar, nlat, nlon), np.int16)
    composites = np.full((ntarget, len(COMPOSITES), nvar, nlat, nlon), np.nan, np.float32)

    sst_rows: list[dict] = []
    relationship_rows: list[dict] = []
    composite_year_rows: list[dict] = []
    sst_metric_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    reference_center_cache: dict[tuple[str, str], dict[str, float]] = {}
    field_cache: dict[tuple[str, str, str], dict[str, xr.DataArray]] = {}
    shift_reference_rows: list[dict] = []
    scale_metric_rows: list[dict] = []
    scale_relationship_rows: list[dict] = []
    sst_scale_cache: dict[tuple[str, str, str], pd.DataFrame] = {}

    for it, target in targets.iterrows():
        print(f"[INFO] {it+1}/{ntarget}: {target.target}")
        source = monthly if target.period_type == "monthly" else seasonal
        pc = get_target_pc(modes_monthly, modes_seasonal, target)
        years = np.asarray(pc.year.values, dtype=int)

        # SST metrics depend only on period and temporal treatment, not on
        # which EOF mode is used as the target.  Cache them so primary_shape
        # and primary_shift analyses reuse exactly the same SST diagnostics.
        cache_key = (str(target.period_type), str(target.period), str(target.series_type))
        reference_key = (str(target.period_type), str(target.period))
        if reference_key not in reference_center_cache:
            reference_center_cache[reference_key] = fixed_reference_centers(
                source, modes_monthly, modes_seasonal,
                str(target.period_type), target.period, static, args
            )
        reference_centers = reference_center_cache[reference_key]
        if cache_key not in sst_metric_cache:
            # SST-state quantities are nonlinear functions (maxima, percentile
            # ranges and peak-to-flank contrasts).  Diagnose those quantities
            # from the physical SST field first.  For the detrended analysis,
            # detrend each resulting metric time series rather than detrending
            # SST before taking a maximum.  This keeps the metric definition
            # identical between raw and detrended analyses.
            sst_field = get_period_field(source, target.period_type, target.period, "sst")
            metric_table: list[dict] = []
            for year in years:
                try:
                    field_y = sst_field.sel(year=year)
                except Exception:
                    continue
                row = sst_metrics_for_field(field_y, static, args, reference_centers)
                row["year"] = int(year)
                metric_table.append(row)
            metric_df_new = pd.DataFrame(metric_table)
            if target.series_type == "linear_detrended" and not metric_df_new.empty:
                metric_years = metric_df_new["year"].to_numpy(float)
                for metric in SST_METRICS:
                    if metric in metric_df_new:
                        metric_df_new[metric] = detrend_vector(
                            metric_years, metric_df_new[metric].to_numpy(float), restore_mean=True
                        )
            sst_metric_cache[cache_key] = metric_df_new
        metric_df = sst_metric_cache[cache_key]
        pc_df = pd.DataFrame({"year": years, "pc": np.asarray(pc.values, dtype=float)})
        joined = pd.merge(pc_df, metric_df, on="year", how="inner")
        joined["target"] = target.target
        joined["period_type"] = target.period_type
        joined["period"] = target.period
        joined["series_type"] = target.series_type
        joined["role"] = target.role
        joined["mode"] = int(target["mode"])
        sst_rows.extend(joined.to_dict("records"))

        # Record each requested width separately from the fixed 15/20/25-degree
        # metrics used above.
        if cache_key not in sst_scale_cache:
            sst_field = get_period_field(source, target.period_type, target.period, "sst")
            sens_rows: list[dict] = []
            for year in years:
                try:
                    field_y = sst_field.sel(year=year)
                except Exception:
                    continue
                for row in sst_scale_sensitivity_for_field(field_y, static, args, reference_centers):
                    row["year"] = int(year)
                    sens_rows.append(row)
            sens_df = pd.DataFrame(sens_rows)
            if target.series_type == "linear_detrended" and not sens_df.empty:
                # Detrend each metric independently within metric family,
                # reference, and scale, mirroring the primary Stage-3 logic.
                for _, inds in sens_df.groupby(["metric_family", "reference", "scale_deg"]).groups.items():
                    inds = list(inds)
                    yrs = sens_df.loc[inds, "year"].to_numpy(float)
                    for col in ("peak_sst_K", "south_value_K", "north_value_K", "contrast_K"):
                        vals = sens_df.loc[inds, col].to_numpy(float)
                        sens_df.loc[inds, col] = detrend_vector(yrs, vals, restore_mean=True)
            sst_scale_cache[cache_key] = sens_df
        sens_df = sst_scale_cache[cache_key]
        if not sens_df.empty:
            sens_join = pd.merge(pc_df, sens_df, on="year", how="inner")
            sens_join["target"] = target.target
            sens_join["period_type"] = target.period_type
            sens_join["period"] = target.period
            sens_join["series_type"] = target.series_type
            sens_join["role"] = target.role
            sens_join["mode"] = int(target["mode"])
            scale_metric_rows.extend(sens_join.to_dict("records"))
            for (family, ref_name, scale), g in sens_join.groupby(
                ["metric_family", "reference", "scale_deg"], sort=True
            ):
                pcv = g.pc.to_numpy(float)
                contrast = g.contrast_K.to_numpy(float)
                r_contrast, p_contrast, n_contrast = safe_corr(pcv, contrast)
                row = {
                    "target": target.target,
                    "period_type": target.period_type,
                    "period": target.period,
                    "series_type": target.series_type,
                    "role": target.role,
                    "mode": int(target["mode"]),
                    "mode_type": target.mode_type,
                    "metric_family": family,
                    "reference": ref_name,
                    "scale_deg": float(scale),
                    "r_contrast": r_contrast,
                    "p_contrast": p_contrast,
                    "n_contrast": n_contrast,
                    "r2_2d": np.nan,
                    "beta_1_std": np.nan,
                    "beta_2_std": np.nan,
                    "predictor_1": "",
                    "predictor_2": "",
                }
                if family == "peak_to_flank":
                    peakv = g.peak_sst_K.to_numpy(float)
                    r2, n2, beta = multiple_r2(pcv, [peakv, contrast])
                    row.update({
                        "r2_2d": r2,
                        "n_years": n2,
                        "beta_1_std": beta[0] if len(beta) > 0 else np.nan,
                        "beta_2_std": beta[1] if len(beta) > 1 else np.nan,
                        "predictor_1": "peak_sst_K",
                        "predictor_2": "contrast_K",
                    })
                    # Separate the information carried by peak SST from the
                    # incremental information carried by the meridional contrast.
                    # All quantities below use the same complete-case year sample.
                    pf = peak_flank_statistics(pcv, peakv, contrast)
                    row.update(pf)
                    # Use a deterministic but group-specific seed so results are
                    # reproducible while bootstrap draws differ across scales/targets.
                    group_seed = (
                        int(args.sst_scale_bootstrap_seed)
                        + sum(ord(c) for c in str(target.target))
                        + int(round(float(scale)*100))
                    ) % (2**32-1)
                    row.update(bootstrap_peak_flank_statistics(
                        pcv, peakv, contrast,
                        int(args.sst_scale_bootstrap_samples), group_seed
                    ))
                else:
                    r2, n2, beta = multiple_r2(
                        pcv, [g.south_value_K.to_numpy(float), g.north_value_K.to_numpy(float)]
                    )
                    row.update({
                        "r2_2d": r2,
                        "n_years": n2,
                        "beta_1_std": beta[0] if len(beta) > 0 else np.nan,
                        "beta_2_std": beta[1] if len(beta) > 1 else np.nan,
                        "predictor_1": "south_value_K",
                        "predictor_2": "north_value_K",
                    })
                scale_relationship_rows.append(row)

        for metric in SST_METRICS:
            if metric not in joined:
                continue
            r, p, n = safe_corr(joined.pc.to_numpy(float), joined[metric].to_numpy(float))
            relationship_rows.append({
                "target": target.target,
                "relationship": "single_metric_correlation",
                "metric_or_model": metric,
                "pearson_r": r,
                "p_value": p,
                "r2": r*r if np.isfinite(r) else np.nan,
                "n_years": n,
            })
        for model in [
            ("sst_zm_peak_K+sst_peak_to_flank_mean_20deg_K", ["sst_zm_peak_K", "sst_peak_to_flank_mean_20deg_K"]),
            ("sst_eq_south_mean_K+sst_eq_north_mean_K", ["sst_eq_south_mean_K", "sst_eq_north_mean_K"]),
            ("sst_tcwvclim_south_mean_K+sst_tcwvclim_north_mean_K", ["sst_tcwvclim_south_mean_K", "sst_tcwvclim_north_mean_K"]),
            ("sst_sstclim_south_mean_K+sst_sstclim_north_mean_K", ["sst_sstclim_south_mean_K", "sst_sstclim_north_mean_K"]),
        ]:
            if all(m in joined for m in model[1]):
                r2, n, beta = multiple_r2(joined.pc.to_numpy(float), [joined[m].to_numpy(float) for m in model[1]])
                relationship_rows.append({
                    "target": target.target,
                    "relationship": "standardized_multiple_regression",
                    "metric_or_model": model[0],
                    "pearson_r": np.nan,
                    "p_value": np.nan,
                    "r2": r2,
                    "n_years": n,
                    "standardized_coefficients": ",".join(f"{b:.6g}" for b in beta),
                })

        # Reference-latitude sensitivity for the two-coordinate meridional SST
        # state space. This is exported for all targets; Supporting figures can
        # filter to modes classified as shift.
        for ref_name, (south_metric, north_metric, contrast_metric) in SHIFT_REFERENCE_SPECS.items():
            if not all(m in joined for m in (south_metric, north_metric, contrast_metric)):
                continue
            r_contrast, p_contrast, n_contrast = safe_corr(
                joined.pc.to_numpy(float), joined[contrast_metric].to_numpy(float)
            )
            r2_ref, n_ref, beta_ref = multiple_r2(
                joined.pc.to_numpy(float),
                [joined[south_metric].to_numpy(float), joined[north_metric].to_numpy(float)],
            )
            shift_reference_rows.append({
                "target": target.target,
                "period_type": target.period_type,
                "period": target.period,
                "series_type": target.series_type,
                "role": target.role,
                "mode": int(target["mode"]),
                "mode_type": target.mode_type,
                "reference": ref_name,
                "center_lat_deg": float(reference_centers.get(ref_name, 0.0 if ref_name == "equator" else np.nan)),
                "half_width_deg": float(args.centered_sst_half_width_deg if ref_name != "equator" else max(abs(args.eq_south_lat_range[0]), abs(args.eq_north_lat_range[1]))),
                "south_metric": south_metric,
                "north_metric": north_metric,
                "contrast_metric": contrast_metric,
                "r_contrast": r_contrast,
                "p_contrast": p_contrast,
                "r2_2d": r2_ref,
                "beta_south_std": beta_ref[0] if len(beta_ref) > 0 else np.nan,
                "beta_north_std": beta_ref[1] if len(beta_ref) > 1 else np.nan,
                "n_years": min(n_contrast, n_ref),
            })

        # Horizontal fields are likewise shared by all EOF targets for a given
        # period and temporal treatment.  Cache once, then regress/composite
        # them against each selected PC.
        if cache_key not in field_cache:
            cached_fields: dict[str, xr.DataArray] = {}
            for variable in map_vars:
                field = get_period_field(source, target.period_type, target.period, variable)
                field = field.sel(lat=map_lat, lon=map_lon)
                if target.series_type == "linear_detrended":
                    field = detrend_field(field)
                cached_fields[variable] = field
            if "u10" in cached_fields and "v10" in cached_fields:
                conv = convergence_from_uv(cached_fields["u10"], cached_fields["v10"])
                for variable in DERIVED_MAP_VARIABLES:
                    cached_fields[variable] = conv[variable]
            field_cache[cache_key] = cached_fields
        fields = field_cache[cache_key]

        high_years_ref: list[int] | None = None
        low_years_ref: list[int] | None = None
        for iv, variable in enumerate(variables):
            if variable not in fields:
                continue
            slope, rmap, pmap, n = regress_field(fields[variable], pc)
            slopes[it, iv] = slope
            corrmaps[it, iv] = rmap
            pmaps[it, iv] = pmap
            nmap[it, iv] = n
            comp, high_years, low_years = composite_field(fields[variable], pc, args.top_n, args.bottom_n)
            composites[it, :, iv] = comp
            if high_years_ref is None:
                high_years_ref, low_years_ref = high_years, low_years
        for group, selected in [("high", high_years_ref or []), ("low", low_years_ref or [])]:
            for rank, year in enumerate(selected, start=1):
                composite_year_rows.append({"target": target.target, "group": group, "rank": rank, "year": int(year)})

    target_coord = targets.target.astype(str).tolist()
    reg_ds = xr.Dataset(
        {
            "slope_per_pc_std": (("target", "variable", "lat", "lon"), slopes),
            "pearson_r": (("target", "variable", "lat", "lon"), corrmaps),
            "p_value": (("target", "variable", "lat", "lon"), pmaps),
            "n_years": (("target", "variable", "lat", "lon"), nmap),
        },
        coords={"target": target_coord, "variable": variables, "lat": map_lat, "lon": map_lon},
    )
    reg_ds.attrs.update({
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "regression_definition": "horizontal field anomaly regressed onto standardized TCWV PC",
        "detrended_definition": "for linear_detrended targets, a linear trend is removed independently from each field grid point and the mean is restored",
    })
    comp_ds = xr.Dataset(
        {"field": (("target", "composite", "variable", "lat", "lon"), composites)},
        coords={"target": target_coord, "composite": list(COMPOSITES), "variable": variables, "lat": map_lat, "lon": map_lon},
    )
    comp_ds.attrs.update({
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "top_n": args.top_n,
        "bottom_n": args.bottom_n,
        "composite_definition": "high/low groups are selected separately for each oriented TCWV PC",
    })

    write_netcdf(reg_ds, outdir/"regression_maps.nc")
    write_netcdf(comp_ds, outdir/"composite_maps.nc")
    pd.DataFrame(sst_rows).to_csv(outdir/"sst_state_metrics.csv", index=False)
    pd.DataFrame(relationship_rows).to_csv(outdir/"pc_sst_relationships.csv", index=False)
    pd.DataFrame(composite_year_rows).to_csv(outdir/"composite_years.csv", index=False)
    pd.DataFrame(shift_reference_rows).to_csv(outdir/"shift_sst_reference_sensitivity.csv", index=False)
    pd.DataFrame(scale_metric_rows).to_csv(outdir/"sst_meridional_scale_metrics.csv", index=False)
    pd.DataFrame(scale_relationship_rows).to_csv(outdir/"sst_meridional_scale_sensitivity.csv", index=False)

    # Evaluate all seasons, EOF1/EOF2, and both SST spaces on the same width grid.
    exhaustive_metrics, exhaustive_rel = exhaustive_seasonal_sst_diagnostics(
        seasonal, static, modes_seasonal, summary, args
    )
    exhaustive_metrics.to_csv(outdir/"seasonal_pc12_sst_space_metrics.csv", index=False)
    exhaustive_rel.to_csv(outdir/"seasonal_pc12_sst_space_scale_sensitivity.csv", index=False)
    if not args.skip_exhaustive_sst_figures:
        figdir = outdir/"exhaustive_sst_figures"
        ensure_dir(figdir)
        for series_type in args.exhaustive_sst_series_types:
            plot_exhaustive_scale_sensitivity(exhaustive_rel, figdir, series_type)
            plot_exhaustive_contrast_correlation(exhaustive_rel, figdir, series_type)
            plot_exhaustive_ns_decomposition(exhaustive_rel, figdir, series_type)
            plot_exhaustive_state_spaces(exhaustive_metrics, exhaustive_rel, figdir, series_type)

    metadata = {
        "pipeline_stage": "03_state",
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "script_name": Path(__file__).name,
        "script_sha256": sha256_file(Path(__file__)),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "prepare_dir": str(prepare),
        "modes_dir": str(modes_dir),
        "analysis_lon_range": args.analysis_lon_range,
        "analysis_lat_range": args.analysis_lat_range,
        "sst_metric_lon_range": args.sst_metric_lon_range,
        "sst_metric_lat_range": args.sst_metric_lat_range,
        "sst_peak_lat_range": args.sst_peak_lat_range,
        "equator_centered_sst_bands": {
            "south": args.eq_south_lat_range,
            "north": args.eq_north_lat_range,
        },
        "centered_sst_half_width_deg": args.centered_sst_half_width_deg,
        "sst_scale_sweep_deg": [float(x) for x in args.sst_scale_sweep_deg],
        "sst_scale_bootstrap_samples": int(args.sst_scale_bootstrap_samples),
        "sst_scale_bootstrap_seed": int(args.sst_scale_bootstrap_seed),
        "exhaustive_sst_series_types": list(args.exhaustive_sst_series_types),
        "exhaustive_sst_figures_skipped": bool(args.skip_exhaustive_sst_figures),
        "exhaustive_sst_diagnostic": (
            "All DJF/MAM/JJA/SON x EOF1/EOF2 combinations are evaluated in both "
            "climatological-TCWV-centered south/north SST space and annual-peak-SST/"
            "peak-to-flank-contrast space at every requested meridional scale."
        ),
        "sst_scale_sensitivity_note": (
            "Sensitivity diagnostics use the same requested degree scales for peak-to-flank "
            "distance and fixed-center south/north band half-width. Values outside the prepared "
            "latitude domain are not extrapolated. For peak-to-flank metrics the output also "
            "separates peak-SST-only, contrast-only, and joint explained variance, the incremental "
            "Delta R2 from adding contrast after peak SST, partial correlation of contrast with the "
            "PC controlling for peak SST, standardized regression coefficients, and year-bootstrap "
            "95% confidence intervals."
        ),
        "fixed_shift_sst_references": {
            "equator": "0 deg; fixed",
            "tcwv_clim_peak": "Stage-2 climatological zonal-mean TCWV maximum for each month/season; fixed across years",
            "sst_clim_peak": "climatological zonal-mean SST maximum for each month/season; fixed across years",
        },
        "shape_mode_selection": (
            "read directly from Stage-2 selected_shape_modes.csv; Stage 3 does not "
            "re-rank shape EOFs"
        ),
        "shift_mode_selection": (
            "highest explained variance fraction among total_abs modes classified "
            "as shift; omitted when --shape-only is used"
        ),
        "shape_only": bool(args.shape_only),
        "selected_shape_modes_file": str(selected_shape_path),
        "selected_shape_modes_sha256": sha256_file(selected_shape_path),
        "series_types": args.series_types,
        "top_n": args.top_n,
        "bottom_n": args.bottom_n,
    }
    write_json(outdir/"run_metadata.json", metadata)
    print(f"[INFO] Wrote state-analysis products to {outdir}")


if __name__ == "__main__":
    main()
