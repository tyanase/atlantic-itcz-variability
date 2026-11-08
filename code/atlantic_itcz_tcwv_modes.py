#!/usr/bin/env python3
"""Calculate monthly and seasonal modes of Atlantic TCWV variability.

Inputs are the Stage 1 products from atlantic_itcz_prepare_era5_fix1.py.
Diagnostics include zonal-mean climatology and variance, profile geometry,
EOFs and PCs, and the translated-reference shift/shape decomposition.

The raw analysis uses interannual anomalies. In linear_detrended analysis,
each latitude's trend is removed before calculating EOFs. Geometric metric
series are detrended within each month or season, retaining their means.

EOF signs follow a physical convention. Shift modes have positive PCs for
northward displacement; shape modes have positive PCs for broader or moister
profiles. Classification uses correlations with the TCWV center, integrated
TCWV, and two width measures. Ambiguous modes are labeled mixed. Scores and
correlations are saved in tcwv_mode_summary.csv.

Each calendar month, season, and temporal treatment is analyzed independently.
Among modes classified as shape, the mode explaining the most variance is
selected. If none qualifies, the selection remains missing and is flagged as
uncertain. Downstream stages use selected_shape_modes.csv.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

try:
    from scipy.ndimage import gaussian_filter1d
except Exception:  # pragma: no cover
    gaussian_filter1d = None

try:
    from scipy.stats import pearsonr
except Exception:  # pragma: no cover
    pearsonr = None

from atlantic_itcz_common import SCHEMA_VERSION, ensure_dir, sha256_file, write_json, write_netcdf

SCRIPT_VERSION = "1.1.0"
DEFAULT_PREPARE = "out_atlantic_itcz/01_prepare"
DEFAULT_OUTPUT = "out_atlantic_itcz/02_modes"
SERIES_TYPES = ("raw", "linear_detrended")
MODE_TYPE_CODE = {"mixed": 0, "shift": 1, "shape": 2}
SEASONS = ("DJF", "MAM", "JJA", "SON")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare-dir", default=DEFAULT_PREPARE)
    p.add_argument("--outdir", default=DEFAULT_OUTPUT)
    p.add_argument("--analysis-lon-range", nargs=2, type=float, default=[-50.0, -20.0])
    p.add_argument("--analysis-lat-range", nargs=2, type=float, default=[-15.0, 30.0])
    p.add_argument("--center-search-lat", nargs=2, type=float, default=[-5.0, 15.0])
    p.add_argument("--rel-lat-range", nargs=2, type=float, default=[-20.0, 20.0])
    p.add_argument("--metric-rel-range", nargs=2, type=float, default=[-10.0, 15.0])
    p.add_argument("--smooth-sigma-deg", type=float, default=1.0)
    p.add_argument("--fractional-thresholds", nargs="+", type=float, default=[0.7, 0.8, 0.9])
    p.add_argument("--absolute-thresholds", nargs="+", type=float, default=[45.0, 48.0, 50.0])
    p.add_argument("--n-modes", type=int, default=3)
    p.add_argument("--classification-margin", type=float, default=0.15)
    p.add_argument("--min-ocean-fraction", type=float, default=0.5)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def safe_mean(a: np.ndarray, axis=None) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    count = np.sum(np.isfinite(a), axis=axis)
    total = np.nansum(a, axis=axis)
    out = np.full(np.asarray(total).shape, np.nan, dtype=float)
    np.divide(total, count, out=out, where=count > 0)
    return out


def safe_var(a: np.ndarray, axis=0, ddof: int = 1) -> np.ndarray:
    """NaN-aware variance without warnings for insufficient local samples."""
    a = np.asarray(a, dtype=float)
    mean = safe_mean(a, axis=axis)
    expanded = np.expand_dims(mean, axis=axis)
    good = np.isfinite(a)
    count = good.sum(axis=axis)
    ss = np.nansum((a-expanded)**2, axis=axis)
    denom = count-ddof
    out = np.full(np.asarray(ss).shape, np.nan, dtype=float)
    np.divide(ss, denom, out=out, where=denom>0)
    return out


def nan_smooth(values: np.ndarray, sigma_grid: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if sigma_grid <= 0:
        return values.copy()
    good = np.isfinite(values)
    if good.sum() < 3:
        return values.copy()
    if gaussian_filter1d is None:
        radius = max(1, int(round(2.0 * sigma_grid)))
        out = np.full_like(values, np.nan)
        for i in range(values.size):
            out[i] = np.nanmean(values[max(0, i-radius):min(values.size, i+radius+1)])
        return out
    num = gaussian_filter1d(np.where(good, values, 0.0), sigma_grid, mode="nearest")
    den = gaussian_filter1d(good.astype(float), sigma_grid, mode="nearest")
    out = np.full_like(values, np.nan)
    valid = den > 1.0e-8
    out[valid] = num[valid] / den[valid]
    return out


def interp1(x: np.ndarray, y: np.ndarray, xnew: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 2:
        return np.full(np.asarray(xnew).shape, np.nan)
    xx = x[good]
    yy = y[good]
    order = np.argsort(xx)
    xx, yy = xx[order], yy[order]
    xx, first = np.unique(xx, return_index=True)
    yy = yy[first]
    return np.interp(xnew, xx, yy, left=np.nan, right=np.nan)


def quadratic_peak(lat: np.ndarray, profile: np.ndarray, search: Sequence[float]) -> float:
    use = (lat >= search[0]) & (lat <= search[1]) & np.isfinite(profile)
    idx = np.where(use)[0]
    if idx.size == 0:
        return np.nan
    j = idx[int(np.nanargmax(profile[idx]))]
    if j == 0 or j == lat.size - 1:
        return float(lat[j])
    x = lat[j-1:j+2]
    y = profile[j-1:j+2]
    if not np.all(np.isfinite(y)):
        return float(lat[j])
    try:
        a, b, _ = np.polyfit(x, y, 2)
    except Exception:
        return float(lat[j])
    if not np.isfinite(a) or a >= 0 or np.isclose(a, 0):
        return float(lat[j])
    vertex = -b / (2*a)
    return float(vertex) if x[0] <= vertex <= x[-1] else float(lat[j])


def outward_crossing(rel: np.ndarray, q: np.ndarray, threshold: float, side: str) -> float:
    if side == "north":
        idx = np.where(rel >= 0)[0]
    elif side == "south":
        idx = np.where(rel <= 0)[0][::-1]
    else:
        raise ValueError(side)
    for i0, i1 in zip(idx[:-1], idx[1:]):
        q0, q1 = q[i0], q[i1]
        if not (np.isfinite(q0) and np.isfinite(q1)):
            continue
        d0, d1 = q0-threshold, q1-threshold
        if np.isclose(d0, 0):
            return float(rel[i0])
        if d0*d1 <= 0:
            if np.isclose(q0, q1):
                return float(0.5*(rel[i0]+rel[i1]))
            f = (threshold-q0)/(q1-q0)
            return float(rel[i0] + f*(rel[i1]-rel[i0]))
    return np.nan


def connected_width(rel: np.ndarray, q: np.ndarray, threshold: float) -> tuple[float, float, float]:
    q0 = float(interp1(rel, q, np.asarray([0.0]))[0])
    if not np.isfinite(q0) or q0 < threshold:
        return np.nan, np.nan, np.nan
    south = outward_crossing(rel, q, threshold, "south")
    north = outward_crossing(rel, q, threshold, "north")
    if not (np.isfinite(south) and np.isfinite(north)):
        return np.nan, np.nan, np.nan
    return float(north-south), float(-south), float(north)


def integrate_window(rel: np.ndarray, q: np.ndarray, limits: Sequence[float]) -> float:
    use = (rel >= limits[0]) & (rel <= limits[1]) & np.isfinite(q)
    if use.sum() < 2:
        return np.nan
    x = rel[use]
    y = q[use]
    return float(np.trapezoid(y, x))


def metric_name_fraction(frac: float) -> str:
    return f"width_frac_{str(frac).replace('.', 'p')}_deg"


def metric_name_absolute(value: float) -> str:
    tag = f"{value:g}".replace(".", "p")
    return f"width_abs_{tag}kgm2_deg"


def profile_metrics(
    lat: np.ndarray,
    q: np.ndarray,
    rel: np.ndarray,
    *,
    center_search: Sequence[float],
    smooth_sigma_grid: float,
    metric_rel_range: Sequence[float],
    fractional_thresholds: Sequence[float],
    absolute_thresholds: Sequence[float],
) -> tuple[dict[str, float], np.ndarray]:
    smooth = nan_smooth(q, smooth_sigma_grid)
    center = quadratic_peak(lat, smooth, center_search)
    row: dict[str, float] = {"center_deg": center}
    if not np.isfinite(center):
        row.update({"q_peak_kgm2": np.nan, "integrated_tcwv_kgm2_deg": np.nan})
        for f in fractional_thresholds:
            row[metric_name_fraction(f)] = np.nan
        for a in absolute_thresholds:
            row[metric_name_absolute(a)] = np.nan
        return row, np.full(rel.shape, np.nan)

    q_rel = interp1(lat, q, center + rel)
    peak_window = np.abs(rel) <= 2.0
    q_peak = float(np.nanmax(q_rel[peak_window])) if np.isfinite(q_rel[peak_window]).any() else np.nan
    row["q_peak_kgm2"] = q_peak
    row["integrated_tcwv_kgm2_deg"] = integrate_window(rel, q_rel, metric_rel_range)

    for frac in fractional_thresholds:
        threshold = frac*q_peak if np.isfinite(q_peak) else np.nan
        width, _, _ = connected_width(rel, q_rel, threshold)
        row[metric_name_fraction(frac)] = width
    for absolute in absolute_thresholds:
        width, _, _ = connected_width(rel, q_rel, absolute)
        row[metric_name_absolute(absolute)] = width
    return row, q_rel


def weighted_zonal_mean(
    field: xr.DataArray,
    ocean_fraction: xr.DataArray,
    ocean_mask: xr.DataArray,
    lon_range: Sequence[float],
    lat_range: Sequence[float],
) -> xr.DataArray:
    field = field.sel(lon=slice(lon_range[0], lon_range[1]), lat=slice(lat_range[0], lat_range[1]))
    of = ocean_fraction.sel(lon=field.lon, lat=field.lat)
    om = ocean_mask.sel(lon=field.lon, lat=field.lat)
    weight = of.where(om)
    valid = np.isfinite(field)
    numerator = (field.where(om) * weight.where(valid)).sum("lon", skipna=True)
    denominator = weight.where(valid).sum("lon", skipna=True)
    return numerator / denominator


def detrend_matrix(year: np.ndarray, x: np.ndarray, restore_mean: bool = True) -> np.ndarray:
    year = np.asarray(year, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    for j in range(x.shape[1]):
        y = x[:, j]
        good = np.isfinite(year) & np.isfinite(y)
        if good.sum() < 3:
            continue
        yc = year[good] - np.mean(year[good])
        coef = np.polyfit(yc, y[good], 1)
        fit = coef[0]*(year-np.mean(year[good])) + coef[1]
        out[good, j] = y[good] - fit[good]
        if restore_mean:
            out[good, j] += np.mean(y[good])
    return out


def detrend_vector(year: np.ndarray, values: np.ndarray, restore_mean: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    good = np.isfinite(year) & np.isfinite(values)
    if good.sum() < 3:
        return out
    yc = year[good] - np.mean(year[good])
    slope, intercept = np.polyfit(yc, values[good], 1)
    fit = slope*(year-np.mean(year[good])) + intercept
    out[good] = values[good]-fit[good]
    if restore_mean:
        out[good] += np.mean(values[good])
    return out


def weighted_eof(x: np.ndarray, lat: np.ndarray, n_modes: int) -> dict[str, np.ndarray]:
    """Covariance EOF with sqrt(cos(latitude)) weighting."""
    x = np.asarray(x, dtype=float)
    sample_valid = np.mean(np.isfinite(x), axis=1) >= 0.8
    sample_idx = np.where(sample_valid)[0]
    xuse = x[sample_valid]
    eof = np.full((n_modes, lat.size), np.nan)
    pc = np.full((n_modes, x.shape[0]), np.nan)
    eigen = np.full(n_modes, np.nan)
    evf = np.full(n_modes, np.nan)
    if xuse.shape[0] < 4:
        return {"eof": eof, "pc": pc, "eigenvalue": eigen, "evf": evf, "n": xuse.shape[0]}
    valid_lat = np.all(np.isfinite(xuse), axis=0)
    if valid_lat.sum() < n_modes+2:
        return {"eof": eof, "pc": pc, "eigenvalue": eigen, "evf": evf, "n": xuse.shape[0]}
    xv = xuse[:, valid_lat]
    xv = xv - xv.mean(axis=0, keepdims=True)
    w = np.sqrt(np.clip(np.cos(np.deg2rad(lat[valid_lat])), 1.0e-8, None))
    xw = xv*w[None, :]
    u, s, vt = np.linalg.svd(xw, full_matrices=False)
    eig_all = s**2/max(1, xw.shape[0]-1)
    total = eig_all.sum()
    nm = min(n_modes, s.size)
    for k in range(nm):
        e = vt[k]/w
        norm = math.sqrt(float(np.sum((e*w)**2)))
        e = e/norm
        projection = e*w**2
        p = np.sum(xv*projection[None, :], axis=1)
        # Deterministic provisional sign.  Physical orientation is applied later.
        j = int(np.nanargmax(np.abs(e)))
        if e[j] < 0:
            e *= -1
            p *= -1
        eof[k, valid_lat] = e
        pc[k, sample_idx] = p
        eigen[k] = eig_all[k]
        evf[k] = eig_all[k]/total if total > 0 else np.nan
    return {"eof": eof, "pc": pc, "eigenvalue": eigen, "evf": evf, "n": xuse.shape[0]}


def finite_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
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


def pattern_corr(a: np.ndarray, b: np.ndarray, lat: np.ndarray) -> float:
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 3:
        return np.nan
    w = np.cos(np.deg2rad(lat[good]))
    aa, bb = a[good], b[good]
    aa = aa - np.sum(w*aa)/np.sum(w)
    bb = bb - np.sum(w*bb)/np.sum(w)
    den = math.sqrt(float(np.sum(w*aa**2)*np.sum(w*bb**2)))
    return float(np.sum(w*aa*bb)/den) if den > 0 else np.nan


def split_half_robustness(x: np.ndarray, lat: np.ndarray, full_eof: np.ndarray, n_modes: int) -> np.ndarray:
    valid = np.where(np.mean(np.isfinite(x), axis=1) >= 0.8)[0]
    out = np.full(n_modes, np.nan)
    if valid.size < 8:
        return out
    half = valid.size//2
    first = weighted_eof(x[valid[:half]], lat, n_modes)["eof"]
    second = weighted_eof(x[valid[half:]], lat, n_modes)["eof"]
    for k in range(n_modes):
        candidates = [abs(pattern_corr(first[k], second[j], lat)) for j in range(n_modes)]
        if np.isfinite(candidates).any():
            out[k] = float(np.nanmax(candidates))
    return out


def shift_shape_variance(q: np.ndarray, lat: np.ndarray, centers: np.ndarray, rel: np.ndarray) -> dict[str, np.ndarray]:
    q_rel = np.full((q.shape[0], rel.size), np.nan)
    for i in range(q.shape[0]):
        if np.isfinite(centers[i]):
            q_rel[i] = interp1(lat, q[i], centers[i]+rel)
    reference = safe_mean(q_rel, axis=0)
    shifted = np.full_like(q, np.nan)
    for i in range(q.shape[0]):
        if np.isfinite(centers[i]):
            shifted[i] = interp1(rel, reference, lat-centers[i])
    common = np.isfinite(q) & np.isfinite(shifted)
    total_raw = np.where(common, q, np.nan)
    shift_raw = np.where(common, shifted, np.nan)
    shape_raw = total_raw-shift_raw
    total = total_raw-safe_mean(total_raw, axis=0)[None, :]
    shift = shift_raw-safe_mean(shift_raw, axis=0)[None, :]
    shape = shape_raw-safe_mean(shape_raw, axis=0)[None, :]
    total_var = safe_var(total, axis=0, ddof=1)
    shift_var = safe_var(shift, axis=0, ddof=1)
    shape_var = safe_var(shape, axis=0, ddof=1)
    cross = total_var-shift_var-shape_var
    denom = shift_var+shape_var
    dominance = np.divide(shift_var-shape_var, denom, out=np.full_like(denom, np.nan), where=denom>0)
    return {
        "relative_profile": q_rel,
        "relative_reference": reference,
        "variance_total": total_var,
        "variance_shift": shift_var,
        "variance_shape": shape_var,
        "variance_cross": cross,
        "shift_shape_dominance": dominance,
    }


def classify_and_orient(
    eof: np.ndarray,
    pc: np.ndarray,
    metrics: pd.DataFrame,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, str, float, float, dict[str, tuple[float, float, int]]]:
    correlations: dict[str, tuple[float, float, int]] = {}
    for metric in metrics.columns:
        if metric in {"year", "period"}:
            continue
        correlations[metric] = finite_corr(pc, metrics[metric].to_numpy(float))
    r_center = correlations.get("center_deg", (np.nan, np.nan, 0))[0]
    shape_keys = [
        "integrated_tcwv_kgm2_deg",
        metric_name_fraction(0.8),
        metric_name_absolute(48.0),
    ]
    signed_shape = [correlations[k][0] for k in shape_keys if k in correlations and np.isfinite(correlations[k][0])]
    shift_score = abs(r_center) if np.isfinite(r_center) else np.nan
    shape_score = float(np.mean(np.abs(signed_shape))) if signed_shape else np.nan
    if np.isfinite(shift_score) and np.isfinite(shape_score):
        if shift_score >= shape_score + margin:
            mode_type = "shift"
        elif shape_score >= shift_score + margin:
            mode_type = "shape"
        else:
            mode_type = "mixed"
    elif np.isfinite(shift_score):
        mode_type = "shift"
    elif np.isfinite(shape_score):
        mode_type = "shape"
    else:
        mode_type = "mixed"

    sign = 1.0
    if mode_type == "shift" and np.isfinite(r_center) and r_center < 0:
        sign = -1.0
    elif mode_type == "shape" and signed_shape and np.nanmean(signed_shape) < 0:
        sign = -1.0
    if sign < 0:
        eof = -eof
        pc = -pc
        correlations = {
            key: ((-r if np.isfinite(r) else r), p, n)
            for key, (r, p, n) in correlations.items()
        }
    return eof, pc, mode_type, shift_score, shape_score, correlations


def period_labels(ds: xr.Dataset, kind: str) -> list[object]:
    return list(range(1, 13)) if kind == "monthly" else list(SEASONS)


def get_period_profiles(zm: xr.DataArray, kind: str, label: object) -> tuple[np.ndarray, np.ndarray]:
    if kind == "monthly":
        da = zm.sel(month=int(label))
    else:
        da = zm.sel(season=str(label))
    years = np.asarray(da.year.values, dtype=int)
    values = np.asarray(da.transpose("year", "lat").values, dtype=float)
    return years, values


def metrics_for_profiles(
    years: np.ndarray,
    q: np.ndarray,
    lat: np.ndarray,
    rel: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, np.ndarray]:
    dlat = float(np.nanmedian(np.diff(lat)))
    sigma_grid = args.smooth_sigma_deg/abs(dlat)
    rows: list[dict[str, float]] = []
    qrel = np.full((years.size, rel.size), np.nan)
    for i, year in enumerate(years):
        row, qrel[i] = profile_metrics(
            lat,
            q[i],
            rel,
            center_search=args.center_search_lat,
            smooth_sigma_grid=sigma_grid,
            metric_rel_range=args.metric_rel_range,
            fractional_thresholds=args.fractional_thresholds,
            absolute_thresholds=args.absolute_thresholds,
        )
        row["year"] = int(year)
        rows.append(row)
    return pd.DataFrame(rows), qrel


def analyze_dataset(
    ds: xr.Dataset,
    static: xr.Dataset,
    kind: str,
    args: argparse.Namespace,
) -> tuple[xr.Dataset, pd.DataFrame, pd.DataFrame]:
    ocean_fraction = static.ocean_fraction
    ocean_mask = static.ocean_mask
    zm = weighted_zonal_mean(
        ds.tcwv,
        ocean_fraction,
        ocean_mask,
        args.analysis_lon_range,
        args.analysis_lat_range,
    )
    lat = np.asarray(zm.lat.values, dtype=float)
    dlat = float(np.nanmedian(np.diff(lat)))
    rel = np.arange(args.rel_lat_range[0], args.rel_lat_range[1]+0.5*abs(dlat), abs(dlat))
    labels = period_labels(ds, kind)
    period_dim = "month" if kind == "monthly" else "season"
    years_all = np.asarray(ds.year.values, dtype=int)
    nperiod = len(labels)
    nmodes = args.n_modes

    # Output arrays use the complete year coordinate.  Missing samples, notably
    # the first DJF when the preceding December is unavailable, remain NaN.
    climatology = np.full((nperiod, lat.size), np.nan, np.float32)
    rel_clim = np.full((nperiod, rel.size), np.nan, np.float32)
    relative_profile = np.full((nperiod, years_all.size, rel.size), np.nan, np.float32)
    variance_total = np.full((nperiod, lat.size), np.nan, np.float32)
    variance_shift = np.full_like(variance_total, np.nan)
    variance_shape = np.full_like(variance_total, np.nan)
    variance_cross = np.full_like(variance_total, np.nan)
    dominance = np.full_like(variance_total, np.nan)

    eof_abs = np.full((len(SERIES_TYPES), nperiod, nmodes, lat.size), np.nan, np.float32)
    pc_abs = np.full((len(SERIES_TYPES), nperiod, nmodes, years_all.size), np.nan, np.float32)
    evf_abs = np.full((len(SERIES_TYPES), nperiod, nmodes), np.nan, np.float32)
    eig_abs = np.full_like(evf_abs, np.nan)
    type_code = np.full((len(SERIES_TYPES), nperiod, nmodes), -1, np.int8)
    split_abs = np.full_like(evf_abs, np.nan)
    north_z = np.full_like(evf_abs, np.nan)

    eof_rel = np.full((len(SERIES_TYPES), nperiod, nmodes, rel.size), np.nan, np.float32)
    pc_rel = np.full((len(SERIES_TYPES), nperiod, nmodes, years_all.size), np.nan, np.float32)
    evf_rel = np.full((len(SERIES_TYPES), nperiod, nmodes), np.nan, np.float32)

    metric_rows: list[dict] = []
    summary_rows: list[dict] = []

    for ip, label in enumerate(labels):
        years, q_all = get_period_profiles(zm, kind, label)
        sample_good = np.mean(np.isfinite(q_all), axis=1) >= 0.8
        years = years[sample_good]
        q = q_all[sample_good]
        if years.size < 4:
            continue

        raw_metrics, qrel = metrics_for_profiles(years, q, lat, rel, args)
        centers = raw_metrics.center_deg.to_numpy(float)
        decomposition = shift_shape_variance(q, lat, centers, rel)
        climatology[ip] = safe_mean(q, axis=0)
        rel_clim[ip] = decomposition["relative_reference"]
        variance_total[ip] = decomposition["variance_total"]
        variance_shift[ip] = decomposition["variance_shift"]
        variance_shape[ip] = decomposition["variance_shape"]
        variance_cross[ip] = decomposition["variance_cross"]
        dominance[ip] = decomposition["shift_shape_dominance"]
        year_index = {int(y): i for i, y in enumerate(years_all)}
        for i, y in enumerate(years):
            relative_profile[ip, year_index[int(y)]] = decomposition["relative_profile"][i]

        for iser, series_type in enumerate(SERIES_TYPES):
            if series_type == "raw":
                q_series = q.copy()
                qrel_series = qrel.copy()
                metrics = raw_metrics.copy()
            else:
                q_series = detrend_matrix(years, q, restore_mean=True)
                qrel_series = detrend_matrix(years, qrel, restore_mean=True)
                metrics = raw_metrics.copy()
                for col in metrics.columns:
                    if col != "year":
                        metrics[col] = detrend_vector(years, metrics[col].to_numpy(float), restore_mean=True)

            for _, row in metrics.iterrows():
                rec = row.to_dict()
                rec.update({
                    "period_type": kind,
                    "period": label,
                    "series_type": series_type,
                })
                metric_rows.append(rec)

            abs_anom = q_series-safe_mean(q_series, axis=0)[None, :]
            rel_anom = qrel_series-safe_mean(qrel_series, axis=0)[None, :]
            eof_a = weighted_eof(abs_anom, lat, nmodes)
            eof_r = weighted_eof(rel_anom, rel, nmodes)
            split = split_half_robustness(abs_anom, lat, eof_a["eof"], nmodes)

            for k in range(nmodes):
                e = eof_a["eof"][k].copy()
                p = eof_a["pc"][k].copy()
                e, p, mode_type, shift_score, shape_score, corr = classify_and_orient(
                    e, p, metrics, args.classification_margin
                )
                eof_abs[iser, ip, k] = e
                evf_abs[iser, ip, k] = eof_a["evf"][k]
                eig_abs[iser, ip, k] = eof_a["eigenvalue"][k]
                type_code[iser, ip, k] = MODE_TYPE_CODE[mode_type]
                split_abs[iser, ip, k] = split[k]
                if k+1 < nmodes and np.isfinite(eof_a["eigenvalue"][k:k+2]).all():
                    l1, l2 = eof_a["eigenvalue"][k:k+2]
                    n = max(3, int(eof_a["n"]))
                    err1 = l1*math.sqrt(2/n)
                    err2 = l2*math.sqrt(2/n)
                    den = math.sqrt(err1**2+err2**2)
                    north_z[iser, ip, k] = (l1-l2)/den if den > 0 else np.nan
                for i, y in enumerate(years):
                    pc_abs[iser, ip, k, year_index[int(y)]] = p[i]

                row_summary = {
                    "period_type": kind,
                    "period": label,
                    "series_type": series_type,
                    "mode_family": "total_abs",
                    "mode": k+1,
                    "mode_type": mode_type,
                    "explained_variance_fraction": eof_a["evf"][k],
                    "eigenvalue": eof_a["eigenvalue"][k],
                    "shift_score": shift_score,
                    "shape_score": shape_score,
                    "split_half_best_pattern_r": split[k],
                    "north_separation_z_to_next": north_z[iser, ip, k],
                    "n_years": int(eof_a["n"]),
                }
                for metric, (r, pval, n) in corr.items():
                    row_summary[f"corr_{metric}"] = r
                    row_summary[f"p_{metric}"] = pval
                    row_summary[f"n_{metric}"] = n
                summary_rows.append(row_summary)

                # Relative-profile EOFs retain their deterministic sign.  They
                # are supporting diagnostics and are not used to classify shift
                # versus shape in the absolute-latitude field.
                eof_rel[iser, ip, k] = eof_r["eof"][k]
                evf_rel[iser, ip, k] = eof_r["evf"][k]
                for i, y in enumerate(years):
                    pc_rel[iser, ip, k, year_index[int(y)]] = eof_r["pc"][k, i]
                summary_rows.append({
                    "period_type": kind,
                    "period": label,
                    "series_type": series_type,
                    "mode_family": "shape_relative",
                    "mode": k+1,
                    "mode_type": "aligned_shape",
                    "explained_variance_fraction": eof_r["evf"][k],
                    "eigenvalue": eof_r["eigenvalue"][k],
                    "shift_score": np.nan,
                    "shape_score": np.nan,
                    "split_half_best_pattern_r": np.nan,
                    "north_separation_z_to_next": np.nan,
                    "n_years": int(eof_r["n"]),
                })

    coords = {
        period_dim: labels,
        "series_type": list(SERIES_TYPES),
        "mode": np.arange(1, nmodes+1, dtype=int),
        "year": years_all,
        "lat": lat,
        "rel_lat": rel,
    }
    out = xr.Dataset(
        {
            "tcwv_zonal_mean": zm.transpose("year", period_dim, "lat").astype("float32"),
            "tcwv_climatology": ((period_dim, "lat"), climatology),
            "tcwv_relative_climatology": ((period_dim, "rel_lat"), rel_clim),
            "tcwv_relative_profile": ((period_dim, "year", "rel_lat"), relative_profile),
            "tcwv_variance_total": ((period_dim, "lat"), variance_total),
            "tcwv_variance_shift": ((period_dim, "lat"), variance_shift),
            "tcwv_variance_shape": ((period_dim, "lat"), variance_shape),
            "tcwv_variance_cross": ((period_dim, "lat"), variance_cross),
            "tcwv_shift_shape_dominance": ((period_dim, "lat"), dominance),
            "total_abs_eof": (("series_type", period_dim, "mode", "lat"), eof_abs),
            "total_abs_pc": (("series_type", period_dim, "mode", "year"), pc_abs),
            "total_abs_evf": (("series_type", period_dim, "mode"), evf_abs),
            "total_abs_eigenvalue": (("series_type", period_dim, "mode"), eig_abs),
            "total_abs_mode_type_code": (("series_type", period_dim, "mode"), type_code),
            "total_abs_split_half_best_pattern_r": (("series_type", period_dim, "mode"), split_abs),
            "total_abs_north_separation_z_to_next": (("series_type", period_dim, "mode"), north_z),
            "shape_relative_eof": (("series_type", period_dim, "mode", "rel_lat"), eof_rel),
            "shape_relative_pc": (("series_type", period_dim, "mode", "year"), pc_rel),
            "shape_relative_evf": (("series_type", period_dim, "mode"), evf_rel),
        },
        coords=coords,
    )
    out["total_abs_mode_type_code"].attrs["mapping"] = "0=mixed, 1=shift, 2=shape"
    out["tcwv_shift_shape_dominance"].attrs["definition"] = "(Var(shift)-Var(shape))/(Var(shift)+Var(shape))"
    out["tcwv_variance_cross"].attrs["definition"] = "Var(total)-Var(shift)-Var(shape) = 2 Cov(shift,shape)"
    out.attrs.update(
        {
            "schema_version": SCHEMA_VERSION,
            "script_version": SCRIPT_VERSION,
            "analysis_lon_range": f"{args.analysis_lon_range[0]:g},{args.analysis_lon_range[1]:g}",
            "analysis_lat_range": f"{args.analysis_lat_range[0]:g},{args.analysis_lat_range[1]:g}",
            "zonal_mean_ocean_weight": "static ocean_fraction=1-lsm, evaluated only where ocean_mask is true",
            "ocean_mask_definition": static.ocean_mask.attrs.get("definition", "lsm < 0.5"),
            "eof_weighting": "sqrt(cos(latitude)) covariance EOF",
            "EOF_sign_convention": "shift: positive PC=northward; shape: positive PC=broader/moister; mixed: deterministic largest-loading sign",
            "shift_shape_decomposition": "total anomaly = translated center-relative reference-profile shift anomaly + residual shape anomaly",
        }
    )
    return out, pd.DataFrame(metric_rows), pd.DataFrame(summary_rows)



def select_representative_shape_modes(summary: pd.DataFrame) -> pd.DataFrame:
    """Select the highest-variance mode already classified as shape.

    Only total-latitude EOFs are eligible. If none is classified as shape,
    selected_mode remains missing with status uncertain_no_shape. The separate
    selection table supplies downstream stages; the full summary retains all modes.
    """
    source = summary[summary["mode_family"] == "total_abs"].copy()
    rows: list[dict] = []

    group_keys = ["period_type", "period", "series_type"]
    for keys, group in source.groupby(group_keys, sort=False, dropna=False):
        period_type, period, series_type = keys
        shape = group[group["mode_type"] == "shape"].copy()

        base = {
            "period_type": period_type,
            "period": period,
            "series_type": series_type,
            "selection_rule": "highest EVF among modes classified as shape",
            "n_shape_candidates": int(len(shape)),
            "shape_candidate_modes": ";".join(str(int(m)) for m in sorted(shape["mode"].dropna().astype(int))),
        }

        if shape.empty:
            # Retain the strongest shape-score mode only as a diagnostic.  It is
            # not promoted to the selected shape mode.
            finite = group[np.isfinite(group["shape_score"])].copy()
            diagnostic = None
            if not finite.empty:
                diagnostic = finite.sort_values(
                    ["shape_score", "explained_variance_fraction", "mode"],
                    ascending=[False, False, True],
                ).iloc[0]
            row = {
                **base,
                "selection_status": "uncertain_no_shape",
                "selected_mode": pd.NA,
                "mode_type": pd.NA,
                "explained_variance_fraction": np.nan,
                "shift_score": np.nan,
                "shape_score": np.nan,
                "best_shape_score_mode": int(diagnostic["mode"]) if diagnostic is not None else pd.NA,
                "best_shape_score": float(diagnostic["shape_score"]) if diagnostic is not None else np.nan,
            }
            rows.append(row)
            continue

        chosen = shape.sort_values(
            ["explained_variance_fraction", "shape_score", "mode"],
            ascending=[False, False, True],
        ).iloc[0]
        row = {
            **base,
            "selection_status": "selected_shape",
            "selected_mode": int(chosen["mode"]),
            "mode_type": str(chosen["mode_type"]),
            "explained_variance_fraction": float(chosen["explained_variance_fraction"]),
            "shift_score": float(chosen["shift_score"]),
            "shape_score": float(chosen["shape_score"]),
            "best_shape_score_mode": int(chosen["mode"]),
            "best_shape_score": float(chosen["shape_score"]),
        }

        # Copy the most useful diagnostics into the selection table.  Missing
        # columns are allowed so additions to the mode-summary schema remain
        # backward compatible.
        diagnostic_columns = [
            "corr_center_deg",
            "corr_integrated_tcwv_kgm2_deg",
            f"corr_{metric_name_fraction(0.8)}",
            f"corr_{metric_name_absolute(48.0)}",
            "split_half_best_pattern_r",
            "north_separation_z_to_next",
            "n_years",
        ]
        for name in diagnostic_columns:
            if name in chosen.index:
                row[name] = chosen[name]
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Use a predictable scientific order instead of lexical ordering of month
    # labels (1, 10, 11, ...).
    type_order = {"monthly": 0, "seasonal": 1}
    season_order = {name: i for i, name in enumerate(SEASONS)}
    out["_type_order"] = out["period_type"].map(type_order).fillna(99)
    out["_period_order"] = [
        int(p) if t == "monthly" else season_order.get(str(p), 99)
        for t, p in zip(out["period_type"], out["period"])
    ]
    out["_series_order"] = out["series_type"].map({name: i for i, name in enumerate(SERIES_TYPES)}).fillna(99)
    out = out.sort_values(["_type_order", "_period_order", "_series_order"]).drop(
        columns=["_type_order", "_period_order", "_series_order"]
    ).reset_index(drop=True)
    return out


def attach_shape_selection(ds: xr.Dataset, selection: pd.DataFrame, kind: str) -> xr.Dataset:
    """Attach selected shape-mode indices to a monthly or seasonal mode file."""
    period_dim = "month" if kind == "monthly" else "season"
    periods = list(ds[period_dim].values)
    series = list(ds["series_type"].values)
    selected = np.full((len(series), len(periods)), -1, dtype=np.int8)
    status = np.zeros((len(series), len(periods)), dtype=np.int8)

    subset = selection[selection["period_type"] == kind]
    for i, stype in enumerate(series):
        for j, period in enumerate(periods):
            match = subset[
                (subset["series_type"] == str(stype))
                & (subset["period"].astype(str) == str(period))
            ]
            if match.empty:
                continue
            row = match.iloc[0]
            if row["selection_status"] == "selected_shape" and pd.notna(row["selected_mode"]):
                selected[i, j] = int(row["selected_mode"])
                status[i, j] = 1

    out = ds.copy()
    out["selected_shape_mode"] = (("series_type", period_dim), selected)
    out["selected_shape_mode"].attrs.update({
        "long_name": "representative total-latitude EOF shape mode",
        "selection_rule": "highest explained variance fraction among modes classified as shape",
        "missing_value_convention": "-1 means no mode classified as shape",
    })
    out["selected_shape_mode_status_code"] = (("series_type", period_dim), status)
    out["selected_shape_mode_status_code"].attrs["mapping"] = "0=uncertain_no_shape, 1=selected_shape"
    return out

def main() -> None:
    args = parse_args()
    prepare = Path(args.prepare_dir)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    monthly = xr.open_dataset(prepare/"era5_monthly.nc")
    seasonal = xr.open_dataset(prepare/"era5_seasonal.nc")
    static = xr.open_dataset(prepare/"static_fields.nc")

    outputs = []
    all_metrics = []
    all_summary = []
    for kind, source in [("monthly", monthly), ("seasonal", seasonal)]:
        print(f"[INFO] Analyzing {kind} TCWV modes")
        dsout, metrics, summary = analyze_dataset(source, static, kind, args)
        all_metrics.append(metrics)
        all_summary.append(summary)
        outputs.append((kind, dsout))

    summary_all = pd.concat(all_summary, ignore_index=True)
    summary_all.to_csv(outdir/"tcwv_mode_summary.csv", index=False)

    selected_shape = select_representative_shape_modes(summary_all)
    selected_shape.to_csv(outdir/"selected_shape_modes.csv", index=False)

    output_paths: list[str] = []
    for kind, dsout in outputs:
        dsout = attach_shape_selection(dsout, selected_shape, kind)
        path = outdir/f"tcwv_modes_{kind}.nc"
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; use --overwrite")
        write_netcdf(dsout, path)
        output_paths.append(str(path))

    for kind, metrics in zip(("monthly", "seasonal"), all_metrics):
        metrics.to_csv(outdir/f"tcwv_metrics_{kind}.csv", index=False)

    metadata = {
        "pipeline_stage": "02_modes",
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "script_name": Path(__file__).name,
        "script_sha256": sha256_file(Path(__file__)),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "prepare_dir": str(prepare),
        "analysis_lon_range": args.analysis_lon_range,
        "analysis_lat_range": args.analysis_lat_range,
        "center_search_lat": args.center_search_lat,
        "rel_lat_range": args.rel_lat_range,
        "metric_rel_range": args.metric_rel_range,
        "smooth_sigma_deg": args.smooth_sigma_deg,
        "fractional_thresholds": args.fractional_thresholds,
        "absolute_thresholds_kgm2": args.absolute_thresholds,
        "n_modes": args.n_modes,
        "classification_margin": args.classification_margin,
        "series_types": list(SERIES_TYPES),
        "shape_mode_selection": "highest explained variance fraction among total_abs modes classified as shape; no forced selection when none is classified as shape",
        "selected_shape_modes_file": str(outdir/"selected_shape_modes.csv"),
        "outputs": output_paths,
    }
    write_json(outdir/"run_metadata.json", metadata)
    print(f"[INFO] Wrote {outdir/'tcwv_mode_summary.csv'}")
    print(f"[INFO] Wrote {outdir/'selected_shape_modes.csv'}")


if __name__ == "__main__":
    main()
