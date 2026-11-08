#!/usr/bin/env python3
"""Plot Fig. 1: monthly TCWV variability and seasonal EOF patterns.

Panels (a,b) show monthly climatology and interannual standard deviation of
zonal-mean TCWV. Panels (c-f) show raw EOF1/EOF2 for DJF, MAM, JJA, and SON.
EOF amplitudes correspond to a +1-standard-deviation PC excursion. Inputs are
the Stage 2 fields and mode-summary table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.ticker import MaxNLocator

plt.rcParams["font.size"] = 12

SCRIPT_VERSION = "1.7.0"
SEASONS = ("DJF", "MAM", "JJA", "SON")
MONTH_LABELS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)

EOF1_COLOR = "#0072B2"   # blue
EOF2_COLOR = "#D55E00"   # vermilion


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--modes-dir",
        type=Path,
        default=Path("out_atlantic_itcz/02_modes"),
        help="Stage-2 output directory.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("out_atlantic_itcz/figures"),
        help="Directory for manuscript figures.",
    )
    p.add_argument(
        "--series-type",
        choices=("raw", "linear_detrended"),
        default="raw",
        help="EOF/PC series used in panels c-f. Main-text default is raw.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution.",
    )
    return p.parse_args()



def _as_str_list(a: np.ndarray) -> list[str]:
    out: list[str] = []
    for x in np.asarray(a).ravel():
        if isinstance(x, (bytes, np.bytes_)):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out



def load_stage2(modes_dir: Path) -> dict[str, np.ndarray | list[str] | pd.DataFrame]:
    fmon = modes_dir / "tcwv_modes_monthly.nc"
    fsea = modes_dir / "tcwv_modes_seasonal.nc"
    fsum = modes_dir / "tcwv_mode_summary.csv"

    for f in (fmon, fsea, fsum):
        if not f.exists():
            raise FileNotFoundError(f"Required Stage-2 file not found: {f}")

    with xr.open_dataset(fmon) as ds:
        ds = ds.load()
        mon = {
            "lat": ds["lat"].values.astype(float),
            "month": ds["month"].values.astype(int),
            "tcwv_climatology": ds["tcwv_climatology"].values.astype(float),
            "tcwv_zonal_mean": ds["tcwv_zonal_mean"].values.astype(float),
        }

    with xr.open_dataset(fsea) as ds:
        ds = ds.load()
        sea = {
            "lat": ds["lat"].values.astype(float),
            "season": _as_str_list(ds["season"].values),
            "series_type": _as_str_list(ds["series_type"].values),
            "tcwv_climatology": ds["tcwv_climatology"].values.astype(float),
            "total_abs_eof": ds["total_abs_eof"].values.astype(float),
            "total_abs_eigenvalue": ds["total_abs_eigenvalue"].values.astype(float),
            "total_abs_evf": ds["total_abs_evf"].values.astype(float),
        }

    if not np.allclose(mon["lat"], sea["lat"]):
        raise ValueError("Monthly and seasonal Stage-2 latitude grids differ.")

    summary = pd.read_csv(fsum)
    return {"monthly": mon, "seasonal": sea, "summary": summary}



def nice_levels(data: np.ndarray, n: int = 12, start_at_zero: bool = False) -> np.ndarray:
    finite = np.asarray(data)[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("No finite values available for contour levels.")
    lo = 0.0 if start_at_zero else float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    locator = MaxNLocator(nbins=n)
    lev = locator.tick_values(lo, hi)
    if start_at_zero:
        lev = lev[lev >= 0]
        if lev.size == 0 or lev[0] > 0:
            lev = np.r_[0.0, lev]
    return lev



def eof_axis_limit(x: np.ndarray) -> float:
    m = float(np.nanmax(np.abs(x)))
    if not np.isfinite(m) or m <= 0:
        return 1.0
    return max(0.5, np.ceil(1.10 * m * 2.0) / 2.0)



def main() -> None:
    args = parse_args()
    d = load_stage2(args.modes_dir)
    mon = d["monthly"]
    sea = d["seasonal"]
    summary: pd.DataFrame = d["summary"]  # type: ignore[assignment]

    lat = np.asarray(mon["lat"], dtype=float)
    month = np.asarray(mon["month"], dtype=int)
    mon_clim = np.asarray(mon["tcwv_climatology"], dtype=float)
    mon_zm = np.asarray(mon["tcwv_zonal_mean"], dtype=float)
    mon_sd = np.nanstd(mon_zm, axis=0, ddof=1)
    mon_peak_lat = lat[np.nanargmax(mon_clim, axis=1)]

    season_names: list[str] = list(sea["season"])  # type: ignore[arg-type]
    series_names: list[str] = list(sea["series_type"])  # type: ignore[arg-type]
    if args.series_type not in series_names:
        raise ValueError(f"Series {args.series_type!r} not present in {series_names}")
    iser = series_names.index(args.series_type)

    eof = np.asarray(sea["total_abs_eof"], dtype=float)[iser, :, :2, :]
    eig = np.asarray(sea["total_abs_eigenvalue"], dtype=float)[iser, :, :2]
    eof_1sigma = eof * np.sqrt(eig)[..., None]

    sea_clim = np.asarray(sea["tcwv_climatology"], dtype=float)
    sea_peak_lat = lat[np.nanargmax(sea_clim, axis=1)]

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=(1.02, 1.0),
        hspace=0.66,
        wspace=0.38,
    )

    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4], sharey=ax_a)

    # Panel (a): climatology.
    # Use 5 kg m-2 spacing for the climatological TCWV shading.
    finite_clim = mon_clim[np.isfinite(mon_clim)]
    clim_lo = 5.0 * np.floor(np.nanmin(finite_clim) / 5.0)
    clim_hi = 5.0 * np.ceil(np.nanmax(finite_clim) / 5.0)
    lev_clim = np.arange(clim_lo, clim_hi + 2.5, 5.0)
    cf_a = ax_a.contourf(
        month,
        lat,
        mon_clim.T,
        levels=lev_clim,
        cmap="BrBG",
        extend="both",
    )
    # Characteristic moist-tropics threshold following Mapes et al. (2018).
    c48 = ax_a.contour(
        month,
        lat,
        mon_clim.T,
        levels=[48.0],
        colors="k",
        linewidths=1.0,
        linestyles="--",
    )
    ax_a.plot(month, mon_peak_lat, color="k", lw=1.0)
    ax_a.axhline(0.0, color="0.45", lw=0.6, ls=":")
    ax_a.set_ylabel("Latitude / (deg)")
    ax_a.set_title("(a) Climatological zonal-mean TCWV")
    cb_a = fig.colorbar(cf_a, ax=ax_a, pad=0.02, fraction=0.045)
    cb_a.set_label(r"TCWV / (kg m$^{-2}$)")


    # Panel (b): interannual variability.
    lev_sd = nice_levels(mon_sd, n=12, start_at_zero=True)
    cf_b = ax_b.contourf(
        month,
        lat,
        mon_sd.T,
        levels=lev_sd,
        cmap="viridis",
        extend="max",
    )
    ax_b.plot(month, mon_peak_lat, color="w", lw=1.0)
    ax_b.axhline(0.0, color="w", alpha=0.65, lw=0.6, ls=":")
    ax_b.set_title("(b) Interannual variability")
    plt.setp(ax_b.get_yticklabels(), visible=False)
    cb_b = fig.colorbar(cf_b, ax=ax_b, pad=0.02, fraction=0.045)
    cb_b.set_label(r"STD(TCWV) / (kg m$^{-2}$)")


    for ax in (ax_a, ax_b):
        ax.set_xlim(1, 12)
        ax.set_ylim(float(lat.min()), float(lat.max()))
        tick_months = np.arange(1, 13, 2)
        ax.set_xticks(tick_months)
        ax.set_xticklabels([MONTH_LABELS[i - 1] for i in tick_months])
        ax.set_xlabel("Month")
        ax.tick_params(axis="y")

    # Panels (c-f): seasonal EOF structures.
    for j, season in enumerate(SEASONS):
        ax = fig.add_subplot(gs[1, j], sharey=ax_a)
        ax2 = ax.twiny()

        if season not in season_names:
            raise ValueError(f"Season {season} absent from seasonal NetCDF")
        isea = season_names.index(season)

        lim1 = eof_axis_limit(eof_1sigma[isea, 0])
        lim2 = eof_axis_limit(eof_1sigma[isea, 1])

        ax.axvline(0.0, color="0.5", lw=0.7, ls=":")
        ax.axhline(sea_peak_lat[isea], color="0.6", lw=0.7, ls="--")
        ax.plot(eof_1sigma[isea, 0], lat, color=EOF1_COLOR, lw=2.0)
        ax2.plot(eof_1sigma[isea, 1], lat, color=EOF2_COLOR, lw=2.0, ls="--")

        ax.set_xlim(-lim1, lim1)
        ax2.set_xlim(-lim2, lim2)
        ax.set_ylim(float(lat.min()), float(lat.max()))
        ax2.set_ylim(float(lat.min()), float(lat.max()))

        ax.tick_params(axis="x", colors=EOF1_COLOR)
        ax2.tick_params(axis="x", colors=EOF2_COLOR, pad=1)
        ax.tick_params(axis="y")
        ax.spines["bottom"].set_color(EOF1_COLOR)
        ax2.spines["top"].set_color(EOF2_COLOR)

        q = summary[
            (summary["period_type"] == "seasonal")
            & (summary["period"] == season)
            & (summary["series_type"] == args.series_type)
            & (summary["mode_family"] == "total_abs")
            & (summary["mode"].isin([1, 2]))
        ].sort_values("mode")
        if len(q) != 2:
            raise ValueError(f"Expected two seasonal total_abs rows for {season}; found {len(q)}")

        txt_lines: list[str] = []
        for _, r in q.iterrows():
            mode = int(r["mode"])
            evf = 100.0 * float(r["explained_variance_fraction"])
            mode_type = str(r["mode_type"])
            txt_lines.append(f"EOF{mode}: {evf:.0f}%")
            txt_lines.append(f"({mode_type})")
            txt_lines.append("")

        ax.text(
            0.04,
            0.88,
            "\n".join(txt_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            linespacing=1.05,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=1.8),
        )

        panel_label = chr(ord("c") + j)
        ax.set_title(f"({panel_label}) {season}", pad=10)

        ax.set_xlabel(
            r"EOF1 $\Delta$TCWV (+1$\sigma$ PC) / (kg m$^{-2}$)",
            fontsize=10,
            color=EOF1_COLOR,
            labelpad=2,
        )
        ax2.set_xlabel(
            r"EOF2 $\Delta$TCWV (+1$\sigma$ PC) / (kg m$^{-2}$)",
            fontsize=10,
            color=EOF2_COLOR,
            labelpad=4,
        )
        ax2.xaxis.set_label_coords(0.5, 1.16)

        if j == 0:
            ax.set_ylabel("Latitude / (deg)")
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        ax2.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)

    args.outdir.mkdir(parents=True, exist_ok=True)
    suffix = "raw" if args.series_type == "raw" else "detrended"
    base = args.outdir / f"fig1_tcwv_seasonal_modes_{suffix}_v1_7"
    fig.savefig(base.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {base.with_suffix('.png')}")
    print(f"Wrote {base.with_suffix('.pdf')}")
    print(f"plot_atlantic_itcz_fig1_v1_7.py version {SCRIPT_VERSION}")


if __name__ == "__main__":
    main()
