#!/usr/bin/env python3
"""Calculate daily organization metrics from Atlantic TCWV fields.

All metrics use daily TCWV on the 1-degree grid. In the manuscript, Agrad is
AI_grad, AI_PW is AI_var, and Hx is H_x.

Variance anisotropy
------------------
Within each latitude window,
    VarX_PW = <(q-[q]_x)^2>,
    VarY_PW = <(q-[q]_y)^2>,
    AI_PW = (VarX_PW-VarY_PW)/(VarX_PW+VarY_PW),
where [q]_x is the zonal mean at each latitude, [q]_y is the meridional mean at
each longitude, and brackets denote area-weighted ocean means. A narrow,
zonally uniform belt approaches AI_PW = -1.

Zonal heterogeneity
-------------------
For q' = q-[q]_x, Hx = <q'^2>/<(q-<q>)^2>. Larger Hx indicates a greater
fraction of horizontal variance in departures from the zonal mean.

Gradient anisotropy
-------------------
For physical horizontal derivatives,
    Ex = <(dq/dx)^2>, Ey = <(dq/dy)^2>,
    Agrad = (Ex-Ey)/(Ex+Ey).
Negative values indicate stronger meridional gradients; positive values
indicate stronger zonal gradients. Cells whose centered gradient stencil
crosses land are excluded from this metric.

Windows and temporal summaries
------------------------------
Metrics are evaluated in a fixed box and in windows following the daily peak
of the smoothed, ocean-weighted zonal-mean TCWV profile. Options include a
1536-km half-width, named degree widths, and a dense width sweep. A correlation
maximum across widths does not by itself identify a unique physical scale.

Daily metrics are not detrended. Monthly or seasonal statistics are related
to the selected Stage 2 PCs; annual statistics are detrended when paired with
a linearly detrended PC.

Upper and lower events use P90 and P10 by default. Thresholds pool all years
within each calendar month. Seasonal fractions combine flags classified
against their constituent months, rather than a single seasonal threshold.
These fractions count days, not independent episodes. Top/bottom raw-PC years
also supply daily samples for distribution plots.

Representative upper- and lower-tail days can be selected with a minimum
temporal separation. Maps retain their observed longitudinal phase; they are
neither shifted nor composited. Optional maps contain TCWV, zonal TCWV
anomalies, and precipitation.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

try:
    from scipy.stats import pearsonr
except Exception:  # pragma: no cover
    pearsonr = None

try:
    from scipy.ndimage import gaussian_filter1d
except Exception:  # pragma: no cover
    gaussian_filter1d = None

from atlantic_itcz_common import EARTH_RADIUS_M, SCHEMA_VERSION, ensure_dir, sha256_file, write_json, write_netcdf

SCRIPT_VERSION = "1.7.0"
DEFAULT_PREPARE = "out_atlantic_itcz/01_prepare"
DEFAULT_MODES = "out_atlantic_itcz/02_modes"
DEFAULT_OUTPUT = "out_atlantic_itcz/04_daily_organization"
METRICS = ("Hx", "Agrad", "AI_PW")
DEFAULT_CENTER_SEARCH_LAT = (-5.0, 15.0)
DEFAULT_CENTER_SMOOTH_SIGMA_DEG = 1.0
DEFAULT_RELATIVE_HALF_WIDTH_KM = (1536.0,)
DEFAULT_RELATIVE_HALF_WIDTH_DEG = (10.0, 7.5)
DEFAULT_SWEEP_HALF_WIDTH_DEG = (5.0, 18.0, 0.5)
DEFAULT_EVENT_PERCENTILE = 90.0
DEFAULT_EVENT_WINDOWS = ("rel_1536km", "rel_10deg", "rel_7p5deg")
DEFAULT_REPRESENTATIVE_WINDOW = "rel_1536km"
DEFAULT_REPRESENTATIVE_N = 3
DEFAULT_REPRESENTATIVE_MIN_SEPARATION_DAYS = 5
EVENT_FRACTIONS = (
    "high_Hx_fraction", "high_Agrad_fraction", "joint_high_fraction",
    "low_Hx_fraction", "low_Agrad_fraction", "joint_low_fraction",
    "high_AI_PW_fraction", "low_AI_PW_fraction",
)
SUMMARY_STATS = ("mean", "median", "p25", "p75", "p90")
SEASON_MONTHS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare-dir", default=DEFAULT_PREPARE)
    p.add_argument("--modes-dir", default=DEFAULT_MODES)
    p.add_argument(
        "--selected-shape-modes",
        default=None,
        help=(
            "Stage-2 selected_shape_modes.csv. By default this is read from "
            "<modes-dir>/selected_shape_modes.csv."
        ),
    )
    p.add_argument("--outdir", default=DEFAULT_OUTPUT)
    p.add_argument("--analysis-lon-range", nargs=2, type=float, default=[-50.0, -20.0])
    p.add_argument(
        "--analysis-lat-range", nargs=2, type=float, default=[-15.0, 30.0],
        help="Fixed-latitude reference box retained for backward comparison.",
    )
    p.add_argument(
        "--center-search-lat", nargs=2, type=float, default=list(DEFAULT_CENTER_SEARCH_LAT),
        help="Latitude range used to locate the daily peak of zonal-mean TCWV.",
    )
    p.add_argument(
        "--center-smooth-sigma-deg", type=float, default=DEFAULT_CENTER_SMOOTH_SIGMA_DEG,
        help="Gaussian smoothing scale applied to the daily zonal-mean TCWV profile before peak detection.",
    )
    p.add_argument(
        "--relative-half-width-km", nargs="*", type=float, default=list(DEFAULT_RELATIVE_HALF_WIDTH_KM),
        help="Center-following half-widths in km. Default 1536 km matches half of the 3072-km idealized meridional domain.",
    )
    p.add_argument(
        "--relative-half-width-deg", nargs="*", type=float, default=list(DEFAULT_RELATIVE_HALF_WIDTH_DEG),
        help="Named center-following sensitivity half-widths in degrees latitude.",
    )
    p.add_argument(
        "--sweep-half-width-deg", nargs=3, type=float, default=list(DEFAULT_SWEEP_HALF_WIDTH_DEG),
        metavar=("MIN", "MAX", "STEP"),
        help=(
            "Dense center-following half-width sweep in degrees latitude. "
            "Default: 5 to 18 degrees every 0.5 degree. Named windows such as "
            "10 and 7.5 degrees are not duplicated in the sweep."
        ),
    )
    p.add_argument(
        "--no-width-sweep", action="store_true",
        help="Disable the dense half-width sweep and retain only the named windows.",
    )
    p.add_argument("--period-types", nargs="+", choices=["monthly", "seasonal"], default=["monthly", "seasonal"])
    p.add_argument("--series-types", nargs="+", choices=["raw", "linear_detrended"], default=["raw", "linear_detrended"])
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--bottom-n", type=int, default=5)
    p.add_argument(
        "--event-percentile", type=float, default=DEFAULT_EVENT_PERCENTILE,
        help=(
            "Upper percentile defining high-Hx/high-Agrad events. Monthly targets "
            "use a calendar-month threshold; seasonal targets use constituent-month "
            "thresholds before aggregation. The lower tail is the symmetric "
            "100-percentile complement (default: P90/P10)."
        ),
    )
    p.add_argument(
        "--event-windows", nargs="+", default=list(DEFAULT_EVENT_WINDOWS),
        help=(
            "Named windows used for daily-distribution and event-frequency analyses. "
            "Dense sweep windows are intentionally excluded by default."
        ),
    )
    p.add_argument(
        "--representative-window", default=DEFAULT_REPRESENTATIVE_WINDOW,
        help="Window used to select actual representative organization-event days.",
    )
    p.add_argument(
        "--representative-n", type=int, default=DEFAULT_REPRESENTATIVE_N,
        help="Number of distinct representative days retained per period and event type.",
    )
    p.add_argument(
        "--representative-min-separation-days", type=int,
        default=DEFAULT_REPRESENTATIVE_MIN_SEPARATION_DAYS,
        help=(
            "Minimum separation between representative dates of the same period/event type, "
            "to avoid selecting several consecutive days from one persistent episode."
        ),
    )
    p.add_argument(
        "--no-representative-maps", action="store_true",
        help="Skip extraction of actual TCWV/precipitation maps for representative event days.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def weighted_mean(values: np.ndarray, weights: np.ndarray, mask: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    valid = np.isfinite(values) & mask
    w = np.where(valid, weights, 0.0)
    num = np.sum(np.where(valid, values, 0.0)*w, axis=axes)
    den = np.sum(w, axis=axes)
    return np.divide(num, den, out=np.full(np.asarray(num).shape, np.nan), where=den>0)


def _nan_gaussian_smooth_1d(values: np.ndarray, sigma_grid: float) -> np.ndarray:
    """Smooth a one-dimensional profile while retaining missing-value support.

    The center diagnostic should not react to one-grid-point noise in the daily
    zonal-mean TCWV profile.  A one-degree Gaussian scale is therefore used by
    default, matching the profile smoothing used in the slower TCWV-mode
    diagnostics.  Missing values are handled by smoothing the data and valid
    mask separately and renormalizing.
    """
    arr = np.asarray(values, dtype=float)
    if sigma_grid <= 0:
        return arr.copy()
    good = np.isfinite(arr)
    if good.sum() < 3:
        return arr.copy()
    if gaussian_filter1d is None:
        half = max(1, int(round(2.0 * sigma_grid)))
        out = np.full_like(arr, np.nan)
        for i in range(arr.size):
            lo = max(0, i-half)
            hi = min(arr.size, i+half+1)
            vals = arr[lo:hi]
            out[i] = np.nanmean(vals) if np.isfinite(vals).any() else np.nan
        return out
    num = gaussian_filter1d(np.where(good, arr, 0.0), sigma_grid, mode="nearest")
    den = gaussian_filter1d(good.astype(float), sigma_grid, mode="nearest")
    out = np.full_like(arr, np.nan)
    use = den > 1.0e-8
    out[use] = num[use]/den[use]
    return out


def _quadratic_peak_latitude(
    lat: np.ndarray,
    profile: np.ndarray,
    search_range: Sequence[float],
    smooth_sigma_deg: float,
) -> tuple[float, float]:
    """Return a sub-grid peak latitude and peak value for one daily profile."""
    lat = np.asarray(lat, dtype=float)
    prof = np.asarray(profile, dtype=float)
    if lat.size < 3:
        return np.nan, np.nan
    dlat = float(np.nanmedian(np.diff(lat)))
    sigma_grid = smooth_sigma_deg/abs(dlat) if np.isfinite(dlat) and dlat != 0 else 0.0
    smooth = _nan_gaussian_smooth_1d(prof, sigma_grid)
    use = (
        (lat >= min(search_range))
        & (lat <= max(search_range))
        & np.isfinite(smooth)
    )
    idx = np.where(use)[0]
    if idx.size == 0:
        return np.nan, np.nan
    j = int(idx[np.nanargmax(smooth[idx])])
    peak_lat = float(lat[j])
    peak_q = float(smooth[j])
    if 0 < j < lat.size-1 and np.all(np.isfinite(smooth[j-1:j+2])):
        x = lat[j-1:j+2]
        y = smooth[j-1:j+2]
        try:
            a, b, c = np.polyfit(x, y, 2)
            if np.isfinite(a) and a < 0 and not np.isclose(a, 0.0):
                xv = float(-b/(2.0*a))
                if x[0] <= xv <= x[-1] and min(search_range) <= xv <= max(search_range):
                    peak_lat = xv
                    peak_q = float(a*xv*xv+b*xv+c)
        except Exception:
            pass
    return peak_lat, peak_q


def _window_specifications(
    analysis_lat_range: Sequence[float],
    relative_half_width_km: Sequence[float],
    relative_half_width_deg: Sequence[float],
    sweep_half_width_deg: Sequence[float] | None = None,
) -> list[dict[str, float | str]]:
    """Construct fixed, named, and densely swept center-following windows.

    The fixed analysis box is retained only as a reference.  The 1536-km
    half-width is kept as the physically motivated reference because it
    corresponds to half of the 3072-km meridional domain of the idealized
    experiments.  Additional named windows provide easy point comparisons.

    A dense sweep is added to diagnose how the PC--organization relationship
    depends on the meridional scale over which organization is measured.  The
    sweep is a sensitivity analysis, not an optimization criterion: the
    half-width giving the largest correlation is not automatically treated as
    a uniquely preferred physical scale.
    """
    specs: list[dict[str, float | str]] = [{
        "window": "fixed_analysis",
        "kind": "fixed",
        "role": "fixed_reference",
        "half_width_deg": np.nan,
        "half_width_km": np.nan,
        "lat_min": float(min(analysis_lat_range)),
        "lat_max": float(max(analysis_lat_range)),
    }]

    def already_present(half_deg: float, atol: float = 1.0e-6) -> bool:
        return any(
            spec["kind"] == "relative"
            and np.isfinite(float(spec["half_width_deg"]))
            and np.isclose(float(spec["half_width_deg"]), half_deg, atol=atol, rtol=0.0)
            for spec in specs
        )

    for km in relative_half_width_km:
        if km <= 0:
            raise ValueError("Relative half-widths must be positive")
        deg = float(km) * 1000.0 / EARTH_RADIUS_M * 180.0 / np.pi
        tag = f"rel_{float(km):g}km".replace(".", "p")
        specs.append({
            "window": tag,
            "kind": "relative",
            "role": "primary_reference" if np.isclose(float(km), 1536.0) else "named_km",
            "half_width_deg": deg,
            "half_width_km": float(km),
            "lat_min": np.nan,
            "lat_max": np.nan,
        })

    for deg in relative_half_width_deg:
        if deg <= 0:
            raise ValueError("Relative half-widths must be positive")
        if already_present(float(deg)):
            continue
        km = float(deg) * np.pi / 180.0 * EARTH_RADIUS_M / 1000.0
        tag = f"rel_{float(deg):g}deg".replace(".", "p")
        specs.append({
            "window": tag,
            "kind": "relative",
            "role": "named_sensitivity",
            "half_width_deg": float(deg),
            "half_width_km": km,
            "lat_min": np.nan,
            "lat_max": np.nan,
        })

    if sweep_half_width_deg is not None:
        if len(sweep_half_width_deg) != 3:
            raise ValueError("sweep_half_width_deg must contain MIN MAX STEP")
        lo, hi, step = [float(v) for v in sweep_half_width_deg]
        if lo <= 0 or hi < lo or step <= 0:
            raise ValueError("Require 0 < sweep MIN <= MAX and STEP > 0")
        n = int(np.floor((hi - lo) / step + 1.0e-9))
        sweep = lo + step * np.arange(n + 1, dtype=float)
        if sweep.size == 0 or sweep[-1] < hi - 1.0e-8:
            sweep = np.append(sweep, hi)
        for deg in sweep:
            deg = float(np.round(deg, 8))
            if already_present(deg):
                continue
            km = deg * np.pi / 180.0 * EARTH_RADIUS_M / 1000.0
            tag = f"sweep_{deg:05.1f}deg".replace(".", "p")
            specs.append({
                "window": tag,
                "kind": "relative",
                "role": "sweep",
                "half_width_deg": deg,
                "half_width_km": km,
                "lat_min": np.nan,
                "lat_max": np.nan,
            })

    labels = [str(spec["window"]) for spec in specs]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate organization-window labels: {labels}")
    return specs

def daily_metrics_from_tcwv(
    q_halo: xr.DataArray,
    static: xr.Dataset,
    lon_range: Sequence[float],
    analysis_lat_range: Sequence[float],
    center_search_lat: Sequence[float],
    center_smooth_sigma_deg: float,
    window_specs: Sequence[dict[str, float | str]],
) -> xr.Dataset:
    """Calculate daily Hx and Agrad for fixed and center-following latitude windows.

    The daily ITCZ latitude is defined as the maximum of the ocean-weighted
    zonal-mean TCWV profile within ``center_search_lat``.  The profile is first
    smoothed in latitude and the maximum is refined by a three-point quadratic
    fit.  This center is used only to place the relative metric windows; the
    Hx and Agrad calculations themselves use the unsmoothed two-dimensional
    daily TCWV field.

    Hx uses all canonical ocean cells within each window.  Agrad excludes
    coastal ocean cells whose centered finite-difference stencil crosses land.
    Gradients are evaluated before applying the moving window, so a gradient at
    a window boundary remains a local property of the physical field rather
    than of the artificial analysis boundary.
    """
    q_halo = q_halo.transpose("time", "lat", "lon")
    lat = np.asarray(q_halo.lat.values, dtype=float)
    lon = np.asarray(q_halo.lon.values, dtype=float)
    Q = np.asarray(q_halo.values, dtype=float)
    if lat.size < 3 or lon.size < 3:
        raise ValueError("Loaded domain is too small for daily organization diagnostics")

    ocean_fraction = np.asarray(
        static.ocean_fraction.sel(lat=lat, lon=lon).values, dtype=float
    )
    ocean_mask = np.asarray(
        static.ocean_mask.sel(lat=lat, lon=lon).values, dtype=bool
    )

    # Daily zonal-mean TCWV used only to locate the ITCZ center.  The same
    # longitude range is used for the organization metrics and the center.
    lon_use = (lon >= min(lon_range)) & (lon <= max(lon_range))
    if lon_use.sum() < 3:
        raise ValueError("Longitude analysis range contains fewer than three grid points")
    Q_lon = Q[:, :, lon_use]
    of_lon = ocean_fraction[:, lon_use]
    om_lon = ocean_mask[:, lon_use]
    valid_lon = np.isfinite(Q_lon) & om_lon[None, :, :]
    wlon = np.where(valid_lon, of_lon[None, :, :], 0.0)
    zonal_num = np.sum(np.where(valid_lon, Q_lon, 0.0)*wlon, axis=2)
    zonal_den = np.sum(wlon, axis=2)
    q_zm = np.divide(
        zonal_num, zonal_den,
        out=np.full(zonal_num.shape, np.nan), where=zonal_den > 0,
    )

    center = np.full(Q.shape[0], np.nan)
    center_peak = np.full(Q.shape[0], np.nan)
    for it in range(Q.shape[0]):
        center[it], center_peak[it] = _quadratic_peak_latitude(
            lat, q_zm[it], center_search_lat, center_smooth_sigma_deg
        )

    # Physical gradients on the complete loaded domain.  The loaded latitude
    # range includes a one-cell halo beyond every possible metric window.
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    dq_dlambda = np.gradient(Q, lon_rad, axis=2)
    dq_dphi = np.gradient(Q, lat_rad, axis=1)
    coslat = np.cos(lat_rad)
    dqdx = dq_dlambda/(EARTH_RADIUS_M*coslat[None, :, None])
    dqdy = dq_dphi/EARTH_RADIUS_M

    gradient_ocean = np.zeros_like(ocean_mask, dtype=bool)
    gradient_ocean[1:-1, 1:-1] = (
        ocean_mask[1:-1, 1:-1]
        & ocean_mask[:-2, 1:-1]
        & ocean_mask[2:, 1:-1]
        & ocean_mask[1:-1, :-2]
        & ocean_mask[1:-1, 2:]
    )
    area_weight = ocean_fraction*np.cos(np.deg2rad(lat))[:, None]

    # Zonal departures are defined latitude by latitude over the selected
    # Atlantic longitude range and can therefore be reused for every latitude
    # window.
    Qm = Q[:, :, lon_use]
    dqdxm = dqdx[:, :, lon_use]
    dqdym = dqdy[:, :, lon_use]
    ofm = ocean_fraction[:, lon_use]
    om = ocean_mask[:, lon_use]
    gom = gradient_ocean[:, lon_use]
    aw = area_weight[:, lon_use]
    valid = np.isfinite(Qm) & om[None, :, :]
    wz = np.where(valid, ofm[None, :, :], 0.0)
    zn = np.sum(np.where(valid, Qm, 0.0)*wz, axis=2)
    zd = np.sum(wz, axis=2)
    q_zm_metric = np.divide(zn, zd, out=np.full(zn.shape, np.nan), where=zd > 0)
    qprime = Qm-q_zm_metric[:, :, None]

    labels = [str(spec["window"]) for spec in window_specs]
    ntime = Q.shape[0]
    nwin = len(window_specs)
    Hx = np.full((ntime, nwin), np.nan)
    hx_num = np.full_like(Hx, np.nan)
    hx_den = np.full_like(Hx, np.nan)
    Agrad = np.full_like(Hx, np.nan)
    Ex = np.full_like(Hx, np.nan)
    Ey = np.full_like(Hx, np.nan)
    VarX_PW = np.full_like(Hx, np.nan)
    VarY_PW = np.full_like(Hx, np.nan)
    AI_PW = np.full_like(Hx, np.nan)
    lat_min_daily = np.full_like(Hx, np.nan)
    lat_max_daily = np.full_like(Hx, np.nan)

    for iw, spec in enumerate(window_specs):
        if str(spec["kind"]) == "fixed":
            lo = np.full(ntime, float(spec["lat_min"]))
            hi = np.full(ntime, float(spec["lat_max"]))
        else:
            half = float(spec["half_width_deg"])
            lo = center-half
            hi = center+half
        lat_min_daily[:, iw] = lo
        lat_max_daily[:, iw] = hi
        band = (lat[None, :] >= lo[:, None]) & (lat[None, :] <= hi[:, None])
        ocean3 = om[None, :, :] & band[:, :, None]
        grad3 = gom[None, :, :] & band[:, :, None]
        weight3 = aw[None, :, :]

        hx_num[:, iw] = weighted_mean(qprime**2, weight3, ocean3, axes=(1, 2))
        q_domain = weighted_mean(Qm, weight3, ocean3, axes=(1, 2))
        hx_den[:, iw] = weighted_mean(
            (Qm-q_domain[:, None, None])**2, weight3, ocean3, axes=(1, 2)
        )
        Hx[:, iw] = np.divide(
            hx_num[:, iw], hx_den[:, iw],
            out=np.full(ntime, np.nan), where=hx_den[:, iw] > 0,
        )
        VarX_PW[:, iw] = hx_num[:, iw]
        q_meridional_mean = weighted_mean(Qm, weight3, ocean3, axes=(1,))
        VarY_PW[:, iw] = weighted_mean(
            (Qm-q_meridional_mean[:, None, :])**2,
            weight3, ocean3, axes=(1, 2),
        )
        AI_PW[:, iw] = np.divide(
            VarX_PW[:, iw]-VarY_PW[:, iw],
            VarX_PW[:, iw]+VarY_PW[:, iw],
            out=np.full(ntime, np.nan),
            where=(VarX_PW[:, iw]+VarY_PW[:, iw]) > 0,
        )
        Ex[:, iw] = weighted_mean(dqdxm**2, weight3, grad3, axes=(1, 2))
        Ey[:, iw] = weighted_mean(dqdym**2, weight3, grad3, axes=(1, 2))
        Agrad[:, iw] = np.divide(
            Ex[:, iw]-Ey[:, iw], Ex[:, iw]+Ey[:, iw],
            out=np.full(ntime, np.nan), where=(Ex[:, iw]+Ey[:, iw]) > 0,
        )

    out = xr.Dataset(
        {
            "Hx": (("time", "window"), Hx.astype("float32")),
            "Hx_numerator": (("time", "window"), hx_num.astype("float32")),
            "Hx_denominator": (("time", "window"), hx_den.astype("float32")),
            "Agrad": (("time", "window"), Agrad.astype("float32")),
            "Ex": (("time", "window"), Ex.astype("float32")),
            "Ey": (("time", "window"), Ey.astype("float32")),
            "VarX_PW": (("time", "window"), VarX_PW.astype("float32")),
            "VarY_PW": (("time", "window"), VarY_PW.astype("float32")),
            "AI_PW": (("time", "window"), AI_PW.astype("float32")),
            "window_lat_min": (("time", "window"), lat_min_daily.astype("float32")),
            "window_lat_max": (("time", "window"), lat_max_daily.astype("float32")),
            "itcz_center_lat": ("time", center.astype("float32")),
            "itcz_center_tcwv": ("time", center_peak.astype("float32")),
            "window_half_width_deg": (
                "window", np.asarray([float(spec["half_width_deg"]) for spec in window_specs], dtype="float32")
            ),
            "window_half_width_km": (
                "window", np.asarray([float(spec["half_width_km"]) for spec in window_specs], dtype="float32")
            ),
        },
        coords={"time": q_halo.time, "window": np.asarray(labels, dtype=str)},
    )
    out["Hx"].attrs.update({
        "long_name": "zonal heterogeneity fraction of daily TCWV",
        "definition": "area-mean((q-zonal_mean(q))^2) / area-mean((q-window_mean(q))^2)",
        "units": "1",
    })
    out["Hx_numerator"].attrs.update({"units": "(kg m-2)^2"})
    out["Hx_denominator"].attrs.update({"units": "(kg m-2)^2"})
    out["Agrad"].attrs.update({
        "long_name": "daily TCWV gradient anisotropy",
        "definition": "(Ex-Ey)/(Ex+Ey), Ex=<dq/dx squared>, Ey=<dq/dy squared>",
        "units": "1",
    })
    out["Ex"].attrs.update({"definition": "area-mean((dq/dx)^2)", "units": "kg2 m-6"})
    out["Ey"].attrs.update({"definition": "area-mean((dq/dy)^2)", "units": "kg2 m-6"})
    out["VarX_PW"].attrs.update({"definition": "area-weighted mean of (q-zonal_mean(q))^2", "units": "(kg m-2)^2"})
    out["VarY_PW"].attrs.update({"definition": "area-weighted mean of (q-meridional_mean(q))^2", "units": "(kg m-2)^2"})
    out["AI_PW"].attrs.update({"definition": "(VarX_PW-VarY_PW)/(VarX_PW+VarY_PW)", "units": "1"})
    out["itcz_center_lat"].attrs.update({
        "long_name": "daily Atlantic ITCZ latitude from zonal-mean TCWV peak",
        "units": "degrees_north",
        "search_range": f"{min(center_search_lat):g},{max(center_search_lat):g}",
        "smoothing_sigma_degrees": float(center_smooth_sigma_deg),
    })
    out["itcz_center_tcwv"].attrs.update({"units": "kg m-2"})
    out["window_lat_min"].attrs.update({"units": "degrees_north"})
    out["window_lat_max"].attrs.update({"units": "degrees_north"})
    out["window_half_width_deg"].attrs.update({"units": "degree"})
    out["window_half_width_km"].attrs.update({"units": "km"})
    return out

def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {stat: np.nan for stat in SUMMARY_STATS}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def organization_statistics(daily: xr.Dataset) -> pd.DataFrame:
    """Summarize daily organization metrics by year, period, and latitude window."""
    frame = daily[list(METRICS)].to_dataframe().reset_index()
    frame["year"] = frame.time.dt.year
    frame["month"] = frame.time.dt.month
    rows: list[dict] = []

    for (year, month, window), group in frame.groupby(["year", "month", "window"]):
        row = {
            "period_type": "monthly",
            "period": int(month),
            "year": int(year),
            "window": str(window),
            "n_days": int(len(group)),
        }
        for metric in METRICS:
            for stat, value in summarize(group[metric].to_numpy(float)).items():
                row[f"{metric}_{stat}"] = value
        rows.append(row)

    # DJF is assigned to the Jan-Feb year.  A seasonal row is retained only if
    # all three constituent calendar months are present.
    season_lookup = {m: season for season, months in SEASON_MONTHS.items() for m in months}
    frame["season"] = frame.month.map(season_lookup)
    frame["season_year"] = frame.year + ((frame.season == "DJF") & (frame.month == 12)).astype(int)
    for (year, season, window), group in frame.groupby(["season_year", "season", "window"]):
        expected = set(SEASON_MONTHS[str(season)])
        if set(group.month.unique()) != expected:
            continue
        row = {
            "period_type": "seasonal",
            "period": str(season),
            "year": int(year),
            "window": str(window),
            "n_days": int(len(group)),
        }
        for metric in METRICS:
            for stat, value in summarize(group[metric].to_numpy(float)).items():
                row[f"{metric}_{stat}"] = value
        rows.append(row)
    return pd.DataFrame(rows)

def detrend_vector(years: np.ndarray, values: np.ndarray) -> np.ndarray:
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    good = np.isfinite(years) & np.isfinite(values)
    if good.sum() < 3:
        return out
    yc = years[good]-np.mean(years[good])
    slope, intercept = np.polyfit(yc, values[good], 1)
    fit = slope*(years-np.mean(years[good]))+intercept
    out[good] = values[good]-fit[good]+np.mean(values[good])
    return out


def safe_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    good = np.isfinite(x) & np.isfinite(y)
    n = int(good.sum())
    if n < 3 or np.std(x[good]) <= 0 or np.std(y[good]) <= 0:
        return np.nan, np.nan, n
    if pearsonr is not None:
        result = pearsonr(x[good], y[good])
        return float(result.statistic), float(result.pvalue), n
    return float(np.corrcoef(x[good], y[good])[0, 1]), np.nan, n


def select_shape_targets(selection: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Return the canonical monthly/seasonal shape targets selected in Stage 2.

    Stage 4 does not classify or re-rank EOFs.  It consumes only rows marked
    ``selected_shape`` by Stage 2.  This keeps the mode choice identical in the
    SST-state, composite, and daily-organization analyses.
    """
    required = {
        "period_type", "period", "series_type", "selection_status",
        "selected_mode", "mode_type", "explained_variance_fraction",
    }
    missing = sorted(required.difference(selection.columns))
    if missing:
        raise KeyError(f"selected_shape_modes.csv is missing columns: {missing}")

    use = selection[
        selection["period_type"].isin(args.period_types)
        & selection["series_type"].isin(args.series_types)
        & (selection["selection_status"] == "selected_shape")
        & selection["selected_mode"].notna()
    ].copy()
    if use.empty:
        raise RuntimeError("No selected shape modes match the requested period/series types")

    rows: list[dict] = []
    for _, row in use.iterrows():
        period_type = str(row["period_type"])
        period = int(row["period"]) if period_type == "monthly" else str(row["period"])
        mode = int(row["selected_mode"])
        series_type = str(row["series_type"])
        rows.append({
            "target": f"{period_type}_{period}_{series_type}_selected_shape_eof{mode}",
            "period_type": period_type,
            "period": period,
            "series_type": series_type,
            "role": "selected_shape",
            "mode": mode,
            "mode_type": str(row["mode_type"]),
            "selection_status": str(row["selection_status"]),
            "explained_variance_fraction": float(row["explained_variance_fraction"]),
            "corr_integrated_tcwv_kgm2_deg": float(row.get("corr_integrated_tcwv_kgm2_deg", np.nan)),
            "corr_width_abs_48kgm2_deg": float(row.get("corr_width_abs_48kgm2_deg", np.nan)),
        })

    out = pd.DataFrame(rows)
    season_order = {"DJF": 1, "MAM": 2, "JJA": 3, "SON": 4}
    out["_period_order"] = [
        int(p) if t == "monthly" else season_order.get(str(p), 99)
        for t, p in zip(out.period_type, out.period)
    ]
    out["_type_order"] = out.period_type.map({"monthly": 0, "seasonal": 1})
    out["_series_order"] = out.series_type.map({"raw": 0, "linear_detrended": 1}).fillna(9)
    out = out.sort_values(["_type_order", "_period_order", "_series_order"]).drop(
        columns=["_period_order", "_type_order", "_series_order"]
    ).reset_index(drop=True)
    return out


def get_pc(target: pd.Series, modes_monthly: xr.Dataset, modes_seasonal: xr.Dataset) -> pd.DataFrame:
    source = modes_monthly if target.period_type == "monthly" else modes_seasonal
    selector = {"series_type": target.series_type, "mode": int(target["mode"])}
    selector["month" if target.period_type == "monthly" else "season"] = int(target.period) if target.period_type == "monthly" else str(target.period)
    pc = source.total_abs_pc.sel(**selector)
    return pd.DataFrame({"year": pc.year.values.astype(int), "pc": np.asarray(pc.values, dtype=float)})


def pc_relationships(
    stats: pd.DataFrame,
    targets: pd.DataFrame,
    modes_monthly: xr.Dataset,
    modes_seasonal: xr.Dataset,
) -> pd.DataFrame:
    """Relate one organization statistic per year to each selected shape PC."""
    rows: list[dict] = []
    stat_columns = [f"{metric}_{stat}" for metric in METRICS for stat in SUMMARY_STATS]
    for _, target in targets.iterrows():
        period_value = int(target.period) if target.period_type == "monthly" else str(target.period)
        subset_all = stats[
            (stats.period_type == target.period_type)
            & (stats.period.astype(str) == str(period_value))
        ].copy()
        pc = get_pc(target, modes_monthly, modes_seasonal)
        for window, subset in subset_all.groupby("window"):
            joined = pd.merge(pc, subset, on="year", how="inner")
            for column in stat_columns:
                values = joined[column].to_numpy(float)
                if target.series_type == "linear_detrended":
                    values = detrend_vector(joined.year.to_numpy(float), values)
                r, p, n = safe_corr(joined.pc.to_numpy(float), values)
                rows.append({
                    "target": target.target,
                    "period_type": target.period_type,
                    "period": target.period,
                    "series_type": target.series_type,
                    "role": target.role,
                    "mode": int(target["mode"]),
                    "explained_variance_fraction": float(target["explained_variance_fraction"]),
                    "window": str(window),
                    "organization_statistic": column,
                    "pearson_r": r,
                    "p_value": p,
                    "n_years": n,
                })
    return pd.DataFrame(rows)


def window_sensitivity_table(
    relationships: pd.DataFrame,
    window_specs: Sequence[dict[str, float | str]],
) -> pd.DataFrame:
    """Attach physical half-widths to PC--organization correlations.

    Only center-following windows are retained.  This compact table is intended
    for plots of correlation versus meridional half-width.  It includes the
    physically motivated 1536-km reference, the named sensitivity windows, and
    the dense sweep in one ordered table.
    """
    meta = pd.DataFrame([
        {
            "window": str(spec["window"]),
            "window_role": str(spec.get("role", "")),
            "window_kind": str(spec["kind"]),
            "half_width_deg": float(spec["half_width_deg"]),
            "half_width_km": float(spec["half_width_km"]),
        }
        for spec in window_specs
    ])
    out = relationships.merge(meta, on="window", how="left", validate="many_to_one")
    out = out[out.window_kind == "relative"].copy()
    if out.empty:
        return out
    split = out.organization_statistic.str.extract(r"^(Hx|Agrad|AI_PW)_(.+)$")
    out["organization_metric"] = split[0]
    out["summary_statistic"] = split[1]
    out["is_primary_reference"] = out.window_role.eq("primary_reference")
    order = [
        "target", "period_type", "period", "series_type", "role", "mode",
        "explained_variance_fraction", "organization_metric", "summary_statistic",
        "window", "window_role", "half_width_deg", "half_width_km",
        "is_primary_reference", "pearson_r", "p_value", "n_years",
    ]
    out = out[order].copy()
    season_order = {"DJF": 1, "MAM": 2, "JJA": 3, "SON": 4}
    out["_type_order"] = out.period_type.map({"monthly": 0, "seasonal": 1}).fillna(9)
    out["_period_order"] = [
        int(period) if ptype == "monthly" else season_order.get(str(period), 99)
        for ptype, period in zip(out.period_type, out.period)
    ]
    out["_series_order"] = out.series_type.map({"raw": 0, "linear_detrended": 1}).fillna(9)
    out = out.sort_values(
        ["_type_order", "_period_order", "_series_order", "organization_metric",
         "summary_statistic", "half_width_deg"]
    ).drop(columns=["_type_order", "_period_order", "_series_order"]).reset_index(drop=True)
    return out


def _daily_period_frame(daily: xr.Dataset) -> pd.DataFrame:
    """Return daily Hx/Agrad values with month, season, and analysis-year labels."""
    frame = daily[list(METRICS)].to_dataframe().reset_index()
    frame["time"] = pd.to_datetime(frame["time"])
    frame["date"] = frame.time.dt.normalize()
    frame["year"] = frame.time.dt.year.astype(int)
    frame["month"] = frame.time.dt.month.astype(int)
    season_lookup = {m: season for season, months in SEASON_MONTHS.items() for m in months}
    frame["season"] = frame.month.map(season_lookup)
    frame["season_year"] = frame.year + ((frame.season == "DJF") & (frame.month == 12)).astype(int)
    return frame


def _period_subset(frame: pd.DataFrame, period_type: str, period: int | str) -> pd.DataFrame:
    """Select one calendar month or complete meteorological season from daily data."""
    if period_type == "monthly":
        out = frame[frame.month == int(period)].copy()
        out["analysis_year"] = out.year.astype(int)
        return out
    season = str(period)
    out = frame[frame.season == season].copy()
    if out.empty:
        return out
    expected = set(SEASON_MONTHS[season])
    complete_years = []
    for year, group in out.groupby("season_year"):
        if set(group.month.unique()) == expected:
            complete_years.append(int(year))
    out = out[out.season_year.isin(complete_years)].copy()
    out["analysis_year"] = out.season_year.astype(int)
    return out


def _threshold_table_for_period_window(
    sub: pd.DataFrame,
    period_type: str,
    period: int | str,
    window: str,
    percentile: float,
) -> pd.DataFrame:
    """Build month-conditioned event thresholds for one period/window.

    Monthly targets have one baseline month.  Seasonal targets have one row for
    each constituent calendar month, using only days that belong to complete
    seasons as selected by ``_period_subset``.  The latter prevents the
    climatological Sep--Nov (or analogous) seasonal cycle from determining
    which days are classified as tail events.
    """
    low_percentile = 100.0 - float(percentile)
    rows: list[dict] = []
    if period_type == "monthly":
        groups = [(int(period), sub)]
        threshold_basis = "calendar_month"
    else:
        groups = [(int(month), group.copy()) for month, group in sub.groupby("month", sort=True)]
        threshold_basis = "constituent_calendar_month"

    for baseline_month, group in groups:
        hx = group.Hx.to_numpy(float)
        ag = group.Agrad.to_numpy(float)
        ai = group.AI_PW.to_numpy(float)
        hx_hi = float(np.nanpercentile(hx, percentile)) if np.isfinite(hx).any() else np.nan
        ag_hi = float(np.nanpercentile(ag, percentile)) if np.isfinite(ag).any() else np.nan
        ai_hi = float(np.nanpercentile(ai, percentile)) if np.isfinite(ai).any() else np.nan
        hx_lo = float(np.nanpercentile(hx, low_percentile)) if np.isfinite(hx).any() else np.nan
        ag_lo = float(np.nanpercentile(ag, low_percentile)) if np.isfinite(ag).any() else np.nan
        ai_lo = float(np.nanpercentile(ai, low_percentile)) if np.isfinite(ai).any() else np.nan
        rows.append({
            "period_type": period_type,
            "period": period,
            "baseline_month": int(baseline_month),
            "threshold_basis": threshold_basis,
            "window": str(window),
            "event_percentile": float(percentile),
            "high_percentile": float(percentile),
            "low_percentile": low_percentile,
            # Backward-compatible aliases for the upper thresholds.
            "Hx_threshold": hx_hi,
            "Agrad_threshold": ag_hi,
            "Hx_high_threshold": hx_hi,
            "Agrad_high_threshold": ag_hi,
            "Hx_low_threshold": hx_lo,
            "Agrad_low_threshold": ag_lo,
            "AI_PW_high_threshold": ai_hi,
            "AI_PW_low_threshold": ai_lo,
            "n_days_Hx": int(np.isfinite(hx).sum()),
            "n_days_Agrad": int(np.isfinite(ag).sum()),
            "n_days_AI_PW": int(np.isfinite(ai).sum()),
        })
    return pd.DataFrame(rows)


def _apply_event_thresholds(sub: pd.DataFrame, threshold_sub: pd.DataFrame) -> pd.DataFrame:
    """Attach the appropriate calendar-month threshold to each daily sample."""
    if sub.empty or threshold_sub.empty:
        return sub.iloc[:0].copy()
    threshold_cols = [
        "baseline_month", "threshold_basis",
        "Hx_high_threshold", "Agrad_high_threshold",
        "Hx_low_threshold", "Agrad_low_threshold",
        "AI_PW_high_threshold", "AI_PW_low_threshold",
    ]
    t = threshold_sub[threshold_cols].copy()
    t["month"] = t.baseline_month.astype(int)
    t = t.drop(columns="baseline_month")
    if t.month.duplicated().any():
        raise ValueError("Event-threshold table contains duplicate baseline months")
    out = sub.merge(t, on="month", how="left", validate="many_to_one")
    threshold_check_cols = [
        "Hx_high_threshold", "Agrad_high_threshold", "Hx_low_threshold", "Agrad_low_threshold",
        "AI_PW_high_threshold", "AI_PW_low_threshold",
    ]
    if out[threshold_check_cols].isna().all(axis=1).any():
        missing_months = sorted(out.loc[
            out[threshold_check_cols].isna().all(axis=1),
            "month",
        ].unique())
        raise ValueError(f"Missing event thresholds for calendar months {missing_months}")
    out["baseline_month"] = out.month.astype(int)
    out["is_high_Hx"] = np.isfinite(out.Hx) & (out.Hx >= out.Hx_high_threshold)
    out["is_high_Agrad"] = np.isfinite(out.Agrad) & (out.Agrad >= out.Agrad_high_threshold)
    out["is_joint_high"] = out.is_high_Hx & out.is_high_Agrad
    out["is_low_Hx"] = np.isfinite(out.Hx) & (out.Hx <= out.Hx_low_threshold)
    out["is_low_Agrad"] = np.isfinite(out.Agrad) & (out.Agrad <= out.Agrad_low_threshold)
    out["is_joint_low"] = out.is_low_Hx & out.is_low_Agrad
    out["is_high_AI_PW"] = np.isfinite(out.AI_PW) & (out.AI_PW >= out.AI_PW_high_threshold)
    out["is_low_AI_PW"] = np.isfinite(out.AI_PW) & (out.AI_PW <= out.AI_PW_low_threshold)
    return out


def event_thresholds_and_frequency(
    daily: xr.Dataset,
    targets: pd.DataFrame,
    event_windows: Sequence[str],
    percentile: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Define symmetric daily-tail states and their yearly occurrence frequency.

    Monthly event thresholds are pooled across all available years within that
    calendar month.  Seasonal event classification is *calendar-month
    conditioned*: each constituent month has its own pooled P90/P10 thresholds,
    after restricting to complete seasons, and those daily flags are then
    aggregated into one seasonal fraction for each year.  Thus, for SON,
    September is judged against September, October against October, and
    November against November rather than against a pooled SON baseline.
    """
    if not 50.0 < float(percentile) < 100.0:
        raise ValueError("event_percentile must lie strictly between 50 and 100")
    low_percentile = 100.0 - float(percentile)
    frame = _daily_period_frame(daily)
    available = set(frame.window.astype(str).unique())
    missing = [w for w in event_windows if w not in available]
    if missing:
        raise KeyError(f"Requested event windows are absent from daily metrics: {missing}")

    period_keys = targets[["period_type", "period"]].drop_duplicates()
    threshold_parts: list[pd.DataFrame] = []
    frequency_rows: list[dict] = []
    for _, key in period_keys.iterrows():
        period_type = str(key.period_type)
        period = int(key.period) if period_type == "monthly" else str(key.period)
        period_frame = _period_subset(frame, period_type, period)
        for window in event_windows:
            sub = period_frame[period_frame.window.astype(str) == str(window)].copy()
            if sub.empty:
                continue
            threshold_sub = _threshold_table_for_period_window(
                sub, period_type, period, str(window), percentile
            )
            threshold_parts.append(threshold_sub)
            classified = _apply_event_thresholds(sub, threshold_sub)

            for year, group in classified.groupby("analysis_year"):
                valid = (
                    np.isfinite(group.Hx.to_numpy(float))
                    & np.isfinite(group.Agrad.to_numpy(float))
                    & np.isfinite(group.AI_PW.to_numpy(float))
                )
                n = int(valid.sum())
                if n == 0:
                    counts = {name: 0 for name in (
                        "high_Hx", "high_Agrad", "joint_high",
                        "low_Hx", "low_Agrad", "joint_low",
                        "high_AI_PW", "low_AI_PW",
                    )}
                    fractions = {name: np.nan for name in counts}
                else:
                    counts = {
                        "high_Hx": int((group.is_high_Hx.to_numpy(bool) & valid).sum()),
                        "high_Agrad": int((group.is_high_Agrad.to_numpy(bool) & valid).sum()),
                        "joint_high": int((group.is_joint_high.to_numpy(bool) & valid).sum()),
                        "low_Hx": int((group.is_low_Hx.to_numpy(bool) & valid).sum()),
                        "low_Agrad": int((group.is_low_Agrad.to_numpy(bool) & valid).sum()),
                        "joint_low": int((group.is_joint_low.to_numpy(bool) & valid).sum()),
                        "high_AI_PW": int((group.is_high_AI_PW.to_numpy(bool) & valid).sum()),
                        "low_AI_PW": int((group.is_low_AI_PW.to_numpy(bool) & valid).sum()),
                    }
                    fractions = {name: counts[name]/n for name in counts}

                frequency_rows.append({
                    "period_type": period_type,
                    "period": period,
                    "year": int(year),
                    "window": str(window),
                    "event_percentile": float(percentile),
                    "high_percentile": float(percentile),
                    "low_percentile": low_percentile,
                    "threshold_basis": (
                        "calendar_month" if period_type == "monthly"
                        else "constituent_calendar_month"
                    ),
                    # There is no single scalar seasonal threshold after month
                    # conditioning.  Keep these aliases for monthly rows only.
                    "Hx_threshold": (
                        float(threshold_sub.iloc[0].Hx_high_threshold)
                        if period_type == "monthly" else np.nan
                    ),
                    "Agrad_threshold": (
                        float(threshold_sub.iloc[0].Agrad_high_threshold)
                        if period_type == "monthly" else np.nan
                    ),
                    "Hx_high_threshold": (
                        float(threshold_sub.iloc[0].Hx_high_threshold)
                        if period_type == "monthly" else np.nan
                    ),
                    "Agrad_high_threshold": (
                        float(threshold_sub.iloc[0].Agrad_high_threshold)
                        if period_type == "monthly" else np.nan
                    ),
                    "Hx_low_threshold": (
                        float(threshold_sub.iloc[0].Hx_low_threshold)
                        if period_type == "monthly" else np.nan
                    ),
                    "Agrad_low_threshold": (
                        float(threshold_sub.iloc[0].Agrad_low_threshold)
                        if period_type == "monthly" else np.nan
                    ),
                    "AI_PW_high_threshold": (
                        float(threshold_sub.iloc[0].AI_PW_high_threshold)
                        if period_type == "monthly" else np.nan
                    ),
                    "AI_PW_low_threshold": (
                        float(threshold_sub.iloc[0].AI_PW_low_threshold)
                        if period_type == "monthly" else np.nan
                    ),
                    "n_valid_days": n,
                    "n_high_Hx_days": counts["high_Hx"],
                    "n_high_Agrad_days": counts["high_Agrad"],
                    "n_joint_high_days": counts["joint_high"],
                    "high_Hx_fraction": fractions["high_Hx"],
                    "high_Agrad_fraction": fractions["high_Agrad"],
                    "joint_high_fraction": fractions["joint_high"],
                    "n_low_Hx_days": counts["low_Hx"],
                    "n_low_Agrad_days": counts["low_Agrad"],
                    "n_joint_low_days": counts["joint_low"],
                    "low_Hx_fraction": fractions["low_Hx"],
                    "low_Agrad_fraction": fractions["low_Agrad"],
                    "joint_low_fraction": fractions["joint_low"],
                    "n_high_AI_PW_days": counts["high_AI_PW"],
                    "high_AI_PW_fraction": fractions["high_AI_PW"],
                    "n_low_AI_PW_days": counts["low_AI_PW"],
                    "low_AI_PW_fraction": fractions["low_AI_PW"],
                })

    thresholds = pd.concat(threshold_parts, ignore_index=True) if threshold_parts else pd.DataFrame()
    return thresholds, pd.DataFrame(frequency_rows)

def pc_event_frequency_relationships(
    frequencies: pd.DataFrame,
    targets: pd.DataFrame,
    modes_monthly: xr.Dataset,
    modes_seasonal: xr.Dataset,
) -> pd.DataFrame:
    """Correlate annual event-day fractions with each selected shape PC."""
    rows: list[dict] = []
    for _, target in targets.iterrows():
        period_value = int(target.period) if target.period_type == "monthly" else str(target.period)
        subset_all = frequencies[
            (frequencies.period_type == target.period_type)
            & (frequencies.period.astype(str) == str(period_value))
        ].copy()
        pc = get_pc(target, modes_monthly, modes_seasonal)
        for window, subset in subset_all.groupby("window"):
            joined = pd.merge(pc, subset, on="year", how="inner")
            for column in EVENT_FRACTIONS:
                values = joined[column].to_numpy(float)
                if target.series_type == "linear_detrended":
                    values = detrend_vector(joined.year.to_numpy(float), values)
                r, p, n = safe_corr(joined.pc.to_numpy(float), values)
                rows.append({
                    "target": target.target,
                    "period_type": target.period_type,
                    "period": target.period,
                    "series_type": target.series_type,
                    "role": target.role,
                    "mode": int(target["mode"]),
                    "explained_variance_fraction": float(target["explained_variance_fraction"]),
                    "window": str(window),
                    "event_measure": column,
                    "pearson_r": r,
                    "p_value": p,
                    "n_years": n,
                })
    return pd.DataFrame(rows)


def raw_pc_daily_distribution_samples(
    daily: xr.Dataset,
    targets: pd.DataFrame,
    modes_monthly: xr.Dataset,
    modes_seasonal: xr.Dataset,
    thresholds: pd.DataFrame,
    event_windows: Sequence[str],
    top_n: int,
    bottom_n: int,
) -> pd.DataFrame:
    """Export actual daily metric samples for raw-PC top/bottom years.

    Every sample carries the threshold appropriate to its calendar month.
    Therefore seasonal daily distributions can be plotted either in raw metric
    space or with month-conditioned event flags without confusing the seasonal
    cycle with interannual organization variability.
    """
    frame = _daily_period_frame(daily)
    rows: list[pd.DataFrame] = []
    raw_targets = targets[targets.series_type == "raw"]
    for _, target in raw_targets.iterrows():
        pc = get_pc(target, modes_monthly, modes_seasonal).dropna().sort_values("pc")
        if len(pc) < max(top_n, bottom_n)+2:
            continue
        low_years = pc.head(bottom_n).year.astype(int).tolist()
        high_years = pc.tail(top_n).year.astype(int).tolist()
        period = int(target.period) if target.period_type == "monthly" else str(target.period)
        period_frame = _period_subset(frame, str(target.period_type), period)
        threshold_sub_all = thresholds[
            (thresholds.period_type == target.period_type)
            & (thresholds.period.astype(str) == str(period))
        ]
        for window in event_windows:
            sub = period_frame[period_frame.window.astype(str) == str(window)].copy()
            threshold_sub = threshold_sub_all[
                threshold_sub_all.window.astype(str) == str(window)
            ].copy()
            if sub.empty or threshold_sub.empty:
                continue
            classified = _apply_event_thresholds(sub, threshold_sub)
            for group, years in (("low", low_years), ("high", high_years)):
                sample = classified[classified.analysis_year.isin(years)].copy()
                if sample.empty:
                    continue
                sample["target"] = target.target
                sample["period_type"] = target.period_type
                sample["period"] = target.period
                sample["mode"] = int(target["mode"])
                sample["group"] = group
                rows.append(sample[[
                    "target", "period_type", "period", "mode", "group", "window",
                    "date", "analysis_year", "month", "baseline_month", "threshold_basis",
                    "Hx", "Agrad", "AI_PW",
                    "Hx_high_threshold", "Agrad_high_threshold",
                    "Hx_low_threshold", "Agrad_low_threshold",
                    "AI_PW_high_threshold", "AI_PW_low_threshold",
                    "is_high_Hx", "is_high_Agrad", "is_joint_high",
                    "is_low_Hx", "is_low_Agrad", "is_joint_low",
                    "is_high_AI_PW", "is_low_AI_PW",
                ]])
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out.date).dt.strftime("%Y-%m-%d")
    return out

def raw_pc_event_group_comparison(
    frequencies: pd.DataFrame,
    targets: pd.DataFrame,
    modes_monthly: xr.Dataset,
    modes_seasonal: xr.Dataset,
    top_n: int,
    bottom_n: int,
) -> pd.DataFrame:
    """Compare event-day fractions between raw-PC top and bottom years."""
    rows: list[dict] = []
    for _, target in targets[targets.series_type == "raw"].iterrows():
        pc = get_pc(target, modes_monthly, modes_seasonal).dropna().sort_values("pc")
        if len(pc) < max(top_n, bottom_n)+2:
            continue
        low_years = pc.head(bottom_n).year.astype(int).tolist()
        high_years = pc.tail(top_n).year.astype(int).tolist()
        period = int(target.period) if target.period_type == "monthly" else str(target.period)
        suball = frequencies[
            (frequencies.period_type == target.period_type)
            & (frequencies.period.astype(str) == str(period))
        ]
        for window, sub in suball.groupby("window"):
            for measure in EVENT_FRACTIONS:
                hi = sub[sub.year.isin(high_years)][measure].to_numpy(float)
                lo = sub[sub.year.isin(low_years)][measure].to_numpy(float)
                rows.append({
                    "target": target.target,
                    "period_type": target.period_type,
                    "period": target.period,
                    "mode": int(target["mode"]),
                    "window": str(window),
                    "event_measure": measure,
                    "high_mean": float(np.nanmean(hi)) if np.isfinite(hi).any() else np.nan,
                    "low_mean": float(np.nanmean(lo)) if np.isfinite(lo).any() else np.nan,
                    "high_minus_low": (
                        float(np.nanmean(hi)-np.nanmean(lo))
                        if np.isfinite(hi).any() and np.isfinite(lo).any() else np.nan
                    ),
                    "high_n_years": int(np.isfinite(hi).sum()),
                    "low_n_years": int(np.isfinite(lo).sum()),
                })
    return pd.DataFrame(rows)


def _select_separated_days(candidates: pd.DataFrame, n: int, min_separation_days: int) -> pd.DataFrame:
    """Take the highest-ranked dates while avoiding repeated days from one episode."""
    if candidates.empty or n <= 0:
        return candidates.iloc[:0].copy()
    chosen = []
    for idx, row in candidates.iterrows():
        date = pd.Timestamp(row.date).normalize()
        if all(abs((date - d).days) >= min_separation_days for d in chosen):
            chosen.append(date)
            if len(chosen) >= n:
                break
    if not chosen:
        return candidates.iloc[:0].copy()
    out = candidates[candidates.date.dt.normalize().isin(chosen)].copy()
    order = {d: i for i, d in enumerate(chosen, start=1)}
    out["rank"] = out.date.dt.normalize().map(order)
    return out.sort_values("rank").drop_duplicates("rank")


def _select_month_balanced_separated_days(
    candidates: pd.DataFrame,
    n: int,
    min_separation_days: int,
) -> pd.DataFrame:
    """Prefer one extreme day from each constituent month before repeats.

    Candidate ordering still defines extremeness.  The first pass takes the
    strongest available candidate from each month, and a second pass fills any
    remaining slots.  This is used only for seasonal representative examples;
    it avoids displaying three cases from one month merely because that month
    has a different raw metric baseline.
    """
    if candidates.empty or n <= 0:
        return candidates.iloc[:0].copy()
    work = candidates.copy()
    work["_candidate_order"] = np.arange(len(work), dtype=int)
    first_pass = work.groupby("month", sort=False, as_index=False).head(1)
    first_pass = first_pass.sort_values("_candidate_order")
    second_pass = work[~work.index.isin(first_pass.index)].sort_values("_candidate_order")

    chosen_rows = []
    chosen_dates: list[pd.Timestamp] = []
    for pool in (first_pass, second_pass):
        for idx, row in pool.iterrows():
            date = pd.Timestamp(row.date).normalize()
            if all(abs((date-d).days) >= min_separation_days for d in chosen_dates):
                chosen_rows.append(idx)
                chosen_dates.append(date)
                if len(chosen_rows) >= n:
                    break
        if len(chosen_rows) >= n:
            break
    if not chosen_rows:
        return candidates.iloc[:0].copy()
    out = work.loc[chosen_rows].copy()
    out["rank"] = np.arange(1, len(out)+1, dtype=int)
    return out.drop(columns="_candidate_order")


def select_representative_event_days(
    daily: xr.Dataset,
    targets: pd.DataFrame,
    modes_monthly: xr.Dataset,
    modes_seasonal: xr.Dataset,
    thresholds: pd.DataFrame,
    representative_window: str,
    n: int,
    min_separation_days: int,
) -> pd.DataFrame:
    """Select actual upper- and lower-tail days without compositing.

    Seasonal event eligibility and ranking are calendar-month conditioned.  A
    September day in SON is therefore compared with other September days, not
    with the raw SON distribution.  Representatives are ranked by pooled
    calendar-month percentile rank, which prevents the climatological seasonal
    cycle from forcing high cases toward the early season or low cases toward
    the late season.  A temporal-separation constraint avoids selecting several
    adjacent days from one persistent episode.
    """
    frame = _daily_period_frame(daily)
    if representative_window not in set(frame.window.astype(str).unique()):
        raise KeyError(f"Representative window {representative_window!r} is absent from daily metrics")
    rows: list[pd.DataFrame] = []
    raw_targets = targets[targets.series_type == "raw"].copy()
    for _, target in raw_targets.iterrows():
        period = int(target.period) if target.period_type == "monthly" else str(target.period)
        sub = _period_subset(frame, str(target.period_type), period)
        sub = sub[sub.window.astype(str) == representative_window].copy()
        if sub.empty:
            continue
        threshold_sub = thresholds[
            (thresholds.period_type == target.period_type)
            & (thresholds.period.astype(str) == str(period))
            & (thresholds.window.astype(str) == representative_window)
        ].copy()
        if threshold_sub.empty:
            continue
        sub = _apply_event_thresholds(sub, threshold_sub)

        # Percentile ranks are evaluated within each calendar month across all
        # complete years.  For monthly targets this is simply the whole target
        # sample; for seasonal targets this removes the intraseasonal baseline.
        sub["Hx_percentile_rank"] = sub.groupby("month")["Hx"].rank(pct=True, method="average")
        sub["Agrad_percentile_rank"] = sub.groupby("month")["Agrad"].rank(pct=True, method="average")
        sub["AI_PW_percentile_rank"] = sub.groupby("month")["AI_PW"].rank(pct=True, method="average")
        sub["joint_score"] = 0.5*(sub.Hx_percentile_rank + sub.Agrad_percentile_rank)
        pc = get_pc(target, modes_monthly, modes_seasonal).rename(
            columns={"year": "analysis_year", "pc": "pc_of_year"}
        )
        sub = sub.merge(pc, on="analysis_year", how="left")

        definitions = [
            (
                "high_Hx",
                sub[sub.is_high_Hx].sort_values(["Hx_percentile_rank", "Hx"], ascending=[False, False]),
                "Hx_percentile_rank",
            ),
            (
                "high_Agrad",
                sub[sub.is_high_Agrad].sort_values(["Agrad_percentile_rank", "Agrad"], ascending=[False, False]),
                "Agrad_percentile_rank",
            ),
            (
                "joint_high",
                sub[sub.is_joint_high].sort_values("joint_score", ascending=False),
                "joint_score",
            ),
            (
                "low_Hx",
                sub[sub.is_low_Hx].sort_values(["Hx_percentile_rank", "Hx"], ascending=[True, True]),
                "Hx_percentile_rank",
            ),
            (
                "low_Agrad",
                sub[sub.is_low_Agrad].sort_values(["Agrad_percentile_rank", "Agrad"], ascending=[True, True]),
                "Agrad_percentile_rank",
            ),
            (
                "joint_low",
                sub[sub.is_joint_low].sort_values("joint_score", ascending=True),
                "joint_score",
            ),
            (
                "high_AI_PW",
                sub[sub.is_high_AI_PW].sort_values(["AI_PW_percentile_rank", "AI_PW"], ascending=[False, False]),
                "AI_PW_percentile_rank",
            ),
            (
                "low_AI_PW",
                sub[sub.is_low_AI_PW].sort_values(["AI_PW_percentile_rank", "AI_PW"], ascending=[True, True]),
                "AI_PW_percentile_rank",
            ),
        ]
        for event_type, candidates, ranking_variable in definitions:
            if str(target.period_type) == "seasonal":
                selected = _select_month_balanced_separated_days(candidates, n, min_separation_days)
            else:
                selected = _select_separated_days(candidates, n, min_separation_days)
            if selected.empty:
                continue
            selected["target"] = target.target
            selected["period_type"] = target.period_type
            selected["period"] = target.period
            selected["mode"] = int(target["mode"])
            selected["representative_window"] = representative_window
            selected["event_type"] = event_type
            selected["ranking_variable"] = ranking_variable
            rows.append(selected[[
                "target", "period_type", "period", "mode", "representative_window",
                "event_type", "rank", "date", "analysis_year", "month", "baseline_month",
                "threshold_basis", "pc_of_year", "Hx", "Agrad", "AI_PW",
                "Hx_percentile_rank", "Agrad_percentile_rank", "AI_PW_percentile_rank", "joint_score",
                "Hx_high_threshold", "Agrad_high_threshold",
                "Hx_low_threshold", "Agrad_low_threshold",
                "AI_PW_high_threshold", "AI_PW_low_threshold", "ranking_variable",
            ]])
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out.date).dt.normalize()
    return out.sort_values(["period_type", "period", "event_type", "rank"]).reset_index(drop=True)

def extract_representative_day_maps(
    prepare: Path,
    selections: pd.DataFrame,
    static: xr.Dataset,
    lon_range: Sequence[float],
    lat_range: Sequence[float],
) -> xr.Dataset:
    """Extract unshifted actual maps for selected organization-event days.

    Maps are never aligned in longitude or averaged.  Each sample preserves its
    observed longitudinal phase.  The zonal TCWV anomaly is included because it
    is the field entering the Hx numerator and is therefore useful for checking
    visually that a high-Hx day is genuinely zonally heterogeneous.
    """
    if selections.empty:
        return xr.Dataset()
    lon_min, lon_max = sorted(map(float, lon_range))
    lat_min, lat_max = sorted(map(float, lat_range))
    static_map = static.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    of = np.asarray(static_map.ocean_fraction.values, dtype=float)
    om = np.asarray(static_map.ocean_mask.values, dtype=bool)
    sample_q = []
    sample_qprime = []
    sample_tp = []
    found_rows = []
    file_cache: dict[int, xr.Dataset] = {}
    try:
        for _, row in selections.iterrows():
            date = pd.Timestamp(row.date).normalize()
            year = int(date.year)
            if year not in file_cache:
                path = prepare/"daily"/f"era5_daily_{year}.nc"
                if not path.exists():
                    raise FileNotFoundError(f"Missing daily file for representative date: {path}")
                file_cache[year] = xr.open_dataset(path)
            ds = file_cache[year]
            dates = pd.DatetimeIndex(ds.time.values).normalize()
            idx = np.where(dates == date)[0]
            if idx.size != 1:
                raise KeyError(f"Expected exactly one daily record for {date.date()}, found {idx.size}")
            day = ds.isel(time=int(idx[0])).sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
            q = np.asarray(day.tcwv.values, dtype=float)
            valid = np.isfinite(q) & om
            w = np.where(valid, of, 0.0)
            zn = np.sum(np.where(valid, q, 0.0)*w, axis=1)
            zd = np.sum(w, axis=1)
            qzm = np.divide(zn, zd, out=np.full(zn.shape, np.nan), where=zd > 0)
            qprime = np.where(om, q-qzm[:, None], np.nan)
            tp = np.asarray(day.tp.values, dtype=float) if "tp" in day else np.full_like(q, np.nan)
            sample_q.append(q)
            sample_qprime.append(qprime)
            sample_tp.append(tp)
            found_rows.append(row)
    finally:
        for ds in file_cache.values():
            ds.close()

    meta = pd.DataFrame(found_rows).reset_index(drop=True)
    ns = len(meta)
    out = xr.Dataset(
        {
            "tcwv": (("sample", "lat", "lon"), np.asarray(sample_q, dtype="float32")),
            "tcwv_zonal_anomaly": (("sample", "lat", "lon"), np.asarray(sample_qprime, dtype="float32")),
            "tp": (("sample", "lat", "lon"), np.asarray(sample_tp, dtype="float32")),
            "ocean_fraction": (("lat", "lon"), of.astype("float32")),
            "ocean_mask": (("lat", "lon"), om.astype("int8")),
            "analysis_year": ("sample", meta.analysis_year.to_numpy(int)),
            "rank": ("sample", meta["rank"].to_numpy(int)),
            "pc_of_year": ("sample", meta.pc_of_year.to_numpy(float).astype("float32")),
            "Hx": ("sample", meta.Hx.to_numpy(float).astype("float32")),
            "Agrad": ("sample", meta.Agrad.to_numpy(float).astype("float32")),
            "AI_PW": ("sample", meta.AI_PW.to_numpy(float).astype("float32")),
            "joint_score": ("sample", meta.joint_score.to_numpy(float).astype("float32")),
        },
        coords={
            "sample": np.arange(ns, dtype=int),
            "time": ("sample", pd.to_datetime(meta.date).to_numpy(dtype="datetime64[ns]")),
            "lat": static_map.lat.values,
            "lon": static_map.lon.values,
            "target": ("sample", meta.target.astype(str).to_numpy()),
            "period_type": ("sample", meta.period_type.astype(str).to_numpy()),
            "period": ("sample", meta.period.astype(str).to_numpy()),
            "event_type": ("sample", meta.event_type.astype(str).to_numpy()),
            "representative_window": ("sample", meta.representative_window.astype(str).to_numpy()),
        },
    )
    out["tcwv"].attrs.update({"units": "kg m-2", "long_name": "actual daily total column water vapor"})
    out["tcwv_zonal_anomaly"].attrs.update({
        "units": "kg m-2",
        "long_name": "daily TCWV departure from ocean-weighted zonal mean",
        "note": "not longitudinally phase-aligned and not composited",
    })
    if sample_tp:
        out["tp"].attrs.update({"long_name": "actual daily precipitation from Stage-1 canonical file"})
    out.attrs.update({
        "representative_day_method": "actual separated extreme days; no longitudinal alignment and no event-day composite",
        "map_lon_range": f"{lon_min:g},{lon_max:g}",
        "map_lat_range": f"{lat_min:g},{lat_max:g}",
    })
    return out

def group_summaries(
    daily: xr.Dataset,
    targets: pd.DataFrame,
    modes_monthly: xr.Dataset,
    modes_seasonal: xr.Dataset,
    top_n: int,
    bottom_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize pooled daily distributions for raw-PC high and low years.

    The pooled daily values are descriptive only.  Because consecutive days are
    serially correlated, no independent-day significance test is applied.  The
    same high/low year selection is applied to every latitude window, allowing
    direct comparison of the fixed-domain and center-following definitions.
    """
    frame = daily[list(METRICS)].to_dataframe().reset_index()
    frame["year"] = frame.time.dt.year
    frame["month"] = frame.time.dt.month
    frame["season"] = frame.month.map({m: s for s, months in SEASON_MONTHS.items() for m in months})
    frame["season_year"] = frame.year + ((frame.season == "DJF") & (frame.month == 12)).astype(int)
    rows: list[dict] = []
    year_rows: list[dict] = []

    for _, target in targets[targets.series_type == "raw"].iterrows():
        pc = get_pc(target, modes_monthly, modes_seasonal).dropna().sort_values("pc")
        if len(pc) < max(top_n, bottom_n)+2:
            continue
        low_years = pc.head(bottom_n).year.astype(int).tolist()
        high_years = pc.tail(top_n).year.astype(int).tolist()
        for group_name, years in [("low", low_years), ("high", high_years)]:
            ordered = low_years if group_name == "low" else list(reversed(high_years))
            for rank, year in enumerate(ordered, start=1):
                year_rows.append({
                    "target": target.target,
                    "period_type": target.period_type,
                    "period": target.period,
                    "mode": int(target["mode"]),
                    "group": group_name,
                    "rank": rank,
                    "year": int(year),
                })

            if target.period_type == "monthly":
                selected_all = frame[(frame.month == int(target.period)) & (frame.year.isin(years))]
            else:
                selected_all = frame[(frame.season == str(target.period)) & (frame.season_year.isin(years))]

            for window, selected in selected_all.groupby("window"):
                for metric in METRICS:
                    s = summarize(selected[metric].to_numpy(float))
                    rows.append({
                        "target": target.target,
                        "period_type": target.period_type,
                        "period": target.period,
                        "series_type": target.series_type,
                        "mode": int(target["mode"]),
                        "window": str(window),
                        "group": group_name,
                        "metric": metric,
                        "n_years": len(years),
                        "n_days": int(np.isfinite(selected[metric]).sum()),
                        **s,
                    })
    return pd.DataFrame(rows), pd.DataFrame(year_rows)

def group_comparisons(groups: pd.DataFrame) -> pd.DataFrame:
    """Return high-minus-low descriptive differences for daily metrics."""
    rows: list[dict] = []
    if groups.empty:
        return pd.DataFrame(rows)
    keys = ["target", "period_type", "period", "series_type", "mode", "window", "metric"]
    for key, sub in groups.groupby(keys, dropna=False):
        high = sub[sub.group == "high"]
        low = sub[sub.group == "low"]
        if high.empty or low.empty:
            continue
        h = high.iloc[0]
        l = low.iloc[0]
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row.update({
            "high_n_days": int(h.n_days),
            "low_n_days": int(l.n_days),
        })
        for stat in SUMMARY_STATS:
            row[f"high_{stat}"] = float(h[stat])
            row[f"low_{stat}"] = float(l[stat])
            row[f"high_minus_low_{stat}"] = float(h[stat]-l[stat])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    prepare = Path(args.prepare_dir)
    modes_dir = Path(args.modes_dir)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    expected_outputs = [
        outdir/"daily_organization_metrics.nc",
        outdir/"organization_statistics.csv",
        outdir/"organization_targets.csv",
        outdir/"pc_organization_relationships.csv",
        outdir/"organization_window_sensitivity.csv",
        outdir/"pc_organization_group_summary.csv",
        outdir/"pc_organization_group_years.csv",
        outdir/"pc_organization_group_comparison.csv",
        outdir/"organization_event_thresholds.csv",
        outdir/"organization_event_frequency.csv",
        outdir/"pc_event_frequency_relationships.csv",
        outdir/"pc_event_frequency_group_comparison.csv",
        outdir/"pc_organization_daily_samples.csv",
        outdir/"representative_event_days.csv",
        outdir/"run_metadata.json",
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Stage-4 outputs already exist ({names}). Use --overwrite to replace them."
        )

    selected_path = Path(args.selected_shape_modes) if args.selected_shape_modes else modes_dir/"selected_shape_modes.csv"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing Stage-2 shape-selection table: {selected_path}")
    selection = pd.read_csv(selected_path)
    targets = select_shape_targets(selection, args)
    targets.to_csv(outdir/"organization_targets.csv", index=False)
    print(f"[INFO] Using {len(targets)} selected shape-mode targets from {selected_path}")

    static = xr.open_dataset(prepare/"static_fields.nc")
    daily_files = sorted((prepare/"daily").glob("era5_daily_*.nc"))
    if not daily_files:
        raise FileNotFoundError(f"No canonical daily files under {prepare/'daily'}")

    sweep_spec = None if args.no_width_sweep else args.sweep_half_width_deg
    window_specs = _window_specifications(
        args.analysis_lat_range,
        args.relative_half_width_km,
        args.relative_half_width_deg,
        sweep_spec,
    )
    print("[INFO] Organization latitude windows:")
    for spec in window_specs:
        if spec["kind"] == "fixed":
            print(f"[INFO]   {spec['window']}: fixed {spec['lat_min']:g} to {spec['lat_max']:g} deg")
        else:
            print(
                f"[INFO]   {spec['window']}: daily TCWV center +/- "
                f"{float(spec['half_width_deg']):.3f} deg "
                f"({float(spec['half_width_km']):.1f} km), role={spec.get('role', '')}"
            )

    # Load enough latitude to contain the largest center-following window plus
    # one neighboring grid cell for centered finite differences.  Stage 1 was
    # intentionally generated on a wider canonical domain for this purpose.
    lon_resolution = float(np.median(np.diff(static.lon.values)))
    lat_resolution = float(np.median(np.diff(static.lat.values)))
    relative_halves = [
        float(spec["half_width_deg"])
        for spec in window_specs
        if spec["kind"] == "relative"
    ]
    max_half = max(relative_halves) if relative_halves else 0.0
    metric_lat_min = min(float(min(args.analysis_lat_range)), float(min(args.center_search_lat))-max_half)
    metric_lat_max = max(float(max(args.analysis_lat_range)), float(max(args.center_search_lat))+max_half)
    halo_lon = [
        float(min(args.analysis_lon_range))-abs(lon_resolution),
        float(max(args.analysis_lon_range))+abs(lon_resolution),
    ]
    halo_lat = [metric_lat_min-abs(lat_resolution), metric_lat_max+abs(lat_resolution)]
    if halo_lon[0] < float(static.lon.min()) or halo_lon[1] > float(static.lon.max()):
        raise ValueError("Stage-1 canonical longitude domain is too narrow for the requested gradient halo")
    if halo_lat[0] < float(static.lat.min()) or halo_lat[1] > float(static.lat.max()):
        raise ValueError(
            "Stage-1 canonical latitude domain is too narrow for the requested center-following window and gradient halo"
        )

    parts: list[xr.Dataset] = []
    for i, path in enumerate(daily_files, start=1):
        print(f"[INFO] {i}/{len(daily_files)}: {path.name}")
        ds = xr.open_dataset(path)
        q = ds.tcwv.sel(
            lon=slice(halo_lon[0], halo_lon[1]),
            lat=slice(halo_lat[0], halo_lat[1]),
        )
        part = daily_metrics_from_tcwv(
            q, static,
            args.analysis_lon_range,
            args.analysis_lat_range,
            args.center_search_lat,
            args.center_smooth_sigma_deg,
            window_specs,
        )
        parts.append(part)
        ds.close()
    daily = xr.concat(parts, dim="time").sortby("time")
    daily.attrs.update({
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "analysis_lon_range": f"{args.analysis_lon_range[0]:g},{args.analysis_lon_range[1]:g}",
        "fixed_analysis_lat_range": f"{args.analysis_lat_range[0]:g},{args.analysis_lat_range[1]:g}",
        "daily_itcz_center_definition": "peak of ocean-weighted zonal-mean TCWV within center_search_lat after latitude smoothing and quadratic peak refinement",
        "center_search_lat": f"{args.center_search_lat[0]:g},{args.center_search_lat[1]:g}",
        "center_smooth_sigma_deg": float(args.center_smooth_sigma_deg),
        "primary_relative_window": "rel_1536km when present; this is half of the 3072-km idealized meridional domain",
        "ocean_weighting": "ocean_fraction * cos(latitude), restricted to fixed ocean_mask (lsm < 0.5)",
        "Hx_ocean_mask": "all ocean cells inside each fixed or daily-center-following latitude window",
        "AI_PW_ocean_mask": "all ocean cells inside each fixed or daily-center-following latitude window",
        "Agrad_ocean_mask": "ocean cells inside each window whose four cardinal finite-difference neighbors are also ocean",
        "gradient_boundary_processing": "physical gradients are evaluated before application of the moving latitude window, using a one-grid-cell halo",
        "additional_spatial_smoothing": "none for Hx/Agrad/AI_PW; smoothing is used only to locate the daily TCWV peak latitude",
    })
    write_netcdf(daily, outdir/"daily_organization_metrics.nc")

    stats = organization_statistics(daily)
    stats.to_csv(outdir/"organization_statistics.csv", index=False)

    modes_monthly = xr.open_dataset(modes_dir/"tcwv_modes_monthly.nc")
    modes_seasonal = xr.open_dataset(modes_dir/"tcwv_modes_seasonal.nc")
    relationships = pc_relationships(stats, targets, modes_monthly, modes_seasonal)
    relationships.to_csv(outdir/"pc_organization_relationships.csv", index=False)
    sensitivity = window_sensitivity_table(relationships, window_specs)
    sensitivity.to_csv(outdir/"organization_window_sensitivity.csv", index=False)

    groups, group_years = group_summaries(
        daily, targets, modes_monthly, modes_seasonal, args.top_n, args.bottom_n
    )
    groups.to_csv(outdir/"pc_organization_group_summary.csv", index=False)
    group_years.to_csv(outdir/"pc_organization_group_years.csv", index=False)
    group_comparisons(groups).to_csv(outdir/"pc_organization_group_comparison.csv", index=False)

    # Event-like daily states are diagnosed only for explicitly named windows,
    # not for every member of the dense width sweep. Monthly targets use one
    # calendar-month baseline; seasonal targets classify each constituent month
    # against its own climatological daily distribution before aggregating the
    # flags to a seasonal fraction. This removes intraseasonal baseline drift.
    available_windows = set(str(v) for v in daily.window.values)
    missing_event_windows = [w for w in args.event_windows if w not in available_windows]
    if missing_event_windows:
        raise KeyError(f"Requested --event-windows are unavailable: {missing_event_windows}")
    if args.representative_window not in available_windows:
        raise KeyError(f"Requested --representative-window is unavailable: {args.representative_window}")

    event_thresholds, event_frequency = event_thresholds_and_frequency(
        daily, targets, args.event_windows, args.event_percentile
    )
    event_thresholds.to_csv(outdir/"organization_event_thresholds.csv", index=False)
    event_frequency.to_csv(outdir/"organization_event_frequency.csv", index=False)
    event_relationships = pc_event_frequency_relationships(
        event_frequency, targets, modes_monthly, modes_seasonal
    )
    event_relationships.to_csv(outdir/"pc_event_frequency_relationships.csv", index=False)
    raw_pc_event_group_comparison(
        event_frequency, targets, modes_monthly, modes_seasonal, args.top_n, args.bottom_n
    ).to_csv(outdir/"pc_event_frequency_group_comparison.csv", index=False)

    daily_samples = raw_pc_daily_distribution_samples(
        daily, targets, modes_monthly, modes_seasonal, event_thresholds,
        args.event_windows, args.top_n, args.bottom_n
    )
    daily_samples.to_csv(outdir/"pc_organization_daily_samples.csv", index=False)

    representative_days = select_representative_event_days(
        daily, targets, modes_monthly, modes_seasonal, event_thresholds,
        args.representative_window, args.representative_n,
        args.representative_min_separation_days,
    )
    representative_days_out = representative_days.copy()
    if not representative_days_out.empty:
        representative_days_out["date"] = pd.to_datetime(representative_days_out.date).dt.strftime("%Y-%m-%d")
    representative_days_out.to_csv(outdir/"representative_event_days.csv", index=False)
    if not args.no_representative_maps and not representative_days.empty:
        rep_maps = extract_representative_day_maps(
            prepare, representative_days, static, args.analysis_lon_range, args.analysis_lat_range
        )
        write_netcdf(rep_maps, outdir/"representative_event_day_maps.nc")

    metadata = {
        "pipeline_stage": "04_daily_organization",
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "script_name": Path(__file__).name,
        "script_sha256": sha256_file(Path(__file__)),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "prepare_dir": str(prepare),
        "modes_dir": str(modes_dir),
        "selected_shape_modes_file": str(selected_path),
        "selected_shape_modes_sha256": sha256_file(selected_path),
        "shape_mode_selection": "consumed directly from Stage-2 selected_shape_modes.csv; Stage 4 does not re-rank EOFs",
        "analysis_lon_range": args.analysis_lon_range,
        "fixed_analysis_lat_range": args.analysis_lat_range,
        "center_search_lat": args.center_search_lat,
        "center_smooth_sigma_deg": args.center_smooth_sigma_deg,
        "relative_half_width_km": args.relative_half_width_km,
        "relative_half_width_deg": args.relative_half_width_deg,
        "sweep_half_width_deg": None if args.no_width_sweep else args.sweep_half_width_deg,
        "width_sweep_interpretation": "dense sensitivity curve of PC--organization correlation versus center-following meridional half-width; not an optimization criterion",
        "organization_windows": [
            {
                "window": str(spec["window"]),
                "kind": str(spec["kind"]),
                "role": str(spec.get("role", "")),
                "half_width_deg": None if not np.isfinite(float(spec["half_width_deg"])) else float(spec["half_width_deg"]),
                "half_width_km": None if not np.isfinite(float(spec["half_width_km"])) else float(spec["half_width_km"]),
                "lat_min": None if not np.isfinite(float(spec["lat_min"])) else float(spec["lat_min"]),
                "lat_max": None if not np.isfinite(float(spec["lat_max"])) else float(spec["lat_max"]),
            }
            for spec in window_specs
        ],
        "daily_itcz_center_definition": "smoothed ocean-weighted zonal-mean TCWV peak within center_search_lat, with three-point quadratic refinement",
        "primary_window_interpretation": "rel_1536km corresponds to half of the 3072-km meridional domain used in the idealized experiment; named and densely swept windows diagnose scale sensitivity",
        "period_types": args.period_types,
        "series_types": args.series_types,
        "top_n": args.top_n,
        "bottom_n": args.bottom_n,
        "Hx_definition": "fraction of area-weighted horizontal TCWV variance carried by zonal departures from the daily zonal mean",
        "Agrad_definition": "(Ex-Ey)/(Ex+Ey), with physical-distance gradients and coastal gradient stencils excluded",
        "AI_PW_definition": "(VarX_PW-VarY_PW)/(VarX_PW+VarY_PW), where VarX_PW is the area-weighted mean zonal TCWV variance and VarY_PW is the area-weighted mean meridional TCWV variance inside the same analysis window",
        "AI_PW_interpretation": "larger AI_PW indicates relatively stronger zonal variance and/or weaker meridional variance; values near -1 represent a strongly zonally extended narrow rainbelt",
        "event_percentile": args.event_percentile,
        "event_windows": args.event_windows,
        "event_definition": "symmetric upper/lower daily tails: monthly targets use pooled calendar-month thresholds; seasonal targets use separate thresholds for each constituent calendar month before daily flags are aggregated to a seasonal fraction; AI_PW high/low events use its own P90/P10 tails; legacy joint_high/joint_low require both Hx and Agrad; annual values are fractions of valid days, not counts of independent episodes",
        "low_event_percentile": 100.0 - float(args.event_percentile),
        "event_threshold_reference_period": "monthly targets: all available years within the same calendar month; seasonal targets: each constituent calendar month pooled across complete seasons, then month-specific flags aggregated within the season",
        "seasonal_event_baseline": "calendar-month-conditioned to remove the climatological intraseasonal baseline before seasonal event-frequency calculation",
        "representative_window": args.representative_window,
        "representative_n": args.representative_n,
        "representative_min_separation_days": args.representative_min_separation_days,
        "representative_map_method": "actual separated upper- and lower-tail days are saved without longitudinal phase alignment or compositing; seasonal representatives are ranked using within-calendar-month percentile ranks and preferentially sample distinct constituent months; AI_PW, Hx, and Agrad representative cases are retained",
        "statistical_note": "PC correlations use one month/season organization statistic or event-day fraction per year; pooled daily high/low distributions are descriptive and receive no independent-day p-values",
    }
    write_json(outdir/"run_metadata.json", metadata)
    print(f"[INFO] Wrote daily-organization products to {outdir}")


if __name__ == "__main__":
    main()
