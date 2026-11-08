#!/usr/bin/env python3
"""Plot the seasonal TCWV diagnostic heatmap (Fig. S1) and monthly EOFs (Fig. S2).

Legacy output stems use the reverse S1/S2 numbering. reproduce_figures.py
assigns the manuscript filenames.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm

SCRIPT_VERSION = "1.1.0"
MONTH_LABELS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
SEASONS = ("DJF", "MAM", "JJA", "SON")
EOF1_COLOR = "#0072B2"
EOF2_COLOR = "#D55E00"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modes-dir", type=Path, default=Path("out_atlantic_itcz/02_modes"))
    p.add_argument("--outdir", type=Path, default=Path("out_atlantic_itcz/figures/supporting"))
    p.add_argument("--series-type", choices=("raw", "linear_detrended"), default="raw")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def _strings(a) -> list[str]:
    out = []
    for x in np.asarray(a).ravel():
        out.append(x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x))
    return out


def axis_limit(x: np.ndarray) -> float:
    m = float(np.nanmax(np.abs(x)))
    return max(0.5, np.ceil(1.10 * m * 2.0) / 2.0)


def load_inputs(modes_dir: Path):
    fmon = modes_dir / "tcwv_modes_monthly.nc"
    fsum = modes_dir / "tcwv_mode_summary.csv"
    for f in (fmon, fsum):
        if not f.exists():
            raise FileNotFoundError(f)

    with xr.open_dataset(fmon) as ds:
        ds = ds.load()
        data = {
            "lat": ds.lat.values.astype(float),
            "month": ds.month.values.astype(int),
            "series_type": _strings(ds.series_type.values),
            "eof": ds.total_abs_eof.values.astype(float),
            "eig": ds.total_abs_eigenvalue.values.astype(float),
            "clim": ds.tcwv_climatology.values.astype(float),
        }
    summary = pd.read_csv(fsum)
    return data, summary


def plot_monthly_modes(data, summary: pd.DataFrame, series_type: str, outdir: Path, dpi: int):
    lat = data["lat"]
    series_names = data["series_type"]
    if series_type not in series_names:
        raise ValueError(f"{series_type} not found in {series_names}")
    iser = series_names.index(series_type)

    eof = data["eof"][iser, :, :2, :]
    eig = data["eig"][iser, :, :2]
    eof1s = eof * np.sqrt(eig)[..., None]
    clim = data["clim"]
    peak_lat = lat[np.nanargmax(clim, axis=1)]

    fig, axes = plt.subplots(3, 4, figsize=(9.2, 8.8), sharey=True)
    axes = axes.ravel()

    for im, ax in enumerate(axes, start=1):
        ax2 = ax.twiny()
        q1 = eof1s[im - 1, 0]
        q2 = eof1s[im - 1, 1]
        ax.set_xlim(-axis_limit(q1), axis_limit(q1))
        ax2.set_xlim(-axis_limit(q2), axis_limit(q2))
        ax.set_ylim(lat.min(), lat.max())
        ax2.set_ylim(lat.min(), lat.max())

        ax.axvline(0, color="0.55", lw=0.6, ls=":")
        ax.axhline(peak_lat[im - 1], color="0.65", lw=0.6, ls="--")
        ax.plot(q1, lat, color=EOF1_COLOR, lw=1.8)
        ax2.plot(q2, lat, color=EOF2_COLOR, lw=1.8, ls="--")

        ax.tick_params(axis="x", colors=EOF1_COLOR, labelsize=6)
        ax2.tick_params(axis="x", colors=EOF2_COLOR, labelsize=6, pad=1)
        ax.tick_params(axis="y", labelsize=6)
        ax.spines["bottom"].set_color(EOF1_COLOR)
        ax2.spines["top"].set_color(EOF2_COLOR)
        ax2.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)

        q = summary[
            (summary.period_type == "monthly")
            & (summary.period.astype(str) == str(im))
            & (summary.series_type == series_type)
            & (summary.mode_family == "total_abs")
            & (summary["mode"].isin([1, 2]))
        ].sort_values("mode")
        lines = []
        for _, r in q.iterrows():
            lines.append(
                f"EOF{int(r['mode'])}: {100*r['explained_variance_fraction']:.0f}% "
                f"({r['mode_type']})"
            )
        ax.text(0.03, 0.96, "\n".join(lines), transform=ax.transAxes,
                ha="left", va="top", fontsize=5.8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=1.2))

        ax.set_title(f"({chr(96 + im)}) {MONTH_LABELS[im-1]}", fontsize=8.2, pad=7)
        if im > 8:
            ax.set_xlabel(r"EOF1 $\Delta$TCWV / (kg m$^{-2}$)", color=EOF1_COLOR, fontsize=6)
        if im <= 4:
            ax2.set_xlabel(r"EOF2 $\Delta$TCWV / (kg m$^{-2}$)", color=EOF2_COLOR, fontsize=6)
            ax2.xaxis.set_label_coords(0.5, 1.09)
        if (im - 1) % 4 == 0:
            ax.set_ylabel("Latitude / (deg)", fontsize=7)

    fig.suptitle(r"Monthly TCWV EOFs; amplitude for a +1$\sigma$ PC excursion", fontsize=9.5, y=0.975)
    fig.subplots_adjust(hspace=0.52, wspace=0.35, top=0.87)

    suffix = "raw" if series_type == "raw" else "detrended"
    base = outdir / f"figS1_tcwv_monthly_modes_{suffix}_v1_1"
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_seasonal_diagnostics(summary: pd.DataFrame, outdir: Path, dpi: int):
    metrics = [
        ("corr_center_deg", "Center latitude"),
        ("corr_integrated_tcwv_kgm2_deg", "Integrated TCWV"),
        ("corr_width_frac_0p8_deg", r"Width $q\geq0.8q_{max}$"),
        ("corr_width_abs_48kgm2_deg", r"Width $q\geq48$"),
    ]

    rows = []
    vals = []
    for series_type, prefix in (("raw", "Raw"), ("linear_detrended", "Detrended")):
        for season in SEASONS:
            q = summary[
                (summary.period_type == "seasonal")
                & (summary.period == season)
                & (summary.series_type == series_type)
                & (summary.mode_family == "total_abs")
                & (summary["mode"].isin([1, 2]))
            ].sort_values("mode")
            for _, r in q.iterrows():
                rows.append(
                    f"{prefix} {season} EOF{int(r['mode'])} "
                    f"{r['mode_type']} ({100*r['explained_variance_fraction']:.0f}%)"
                )
                vals.append([float(r[c]) for c, _ in metrics])

    arr = np.asarray(vals)
    bounds = np.arange(-1.0, 1.0001, 0.2)
    cmap = plt.get_cmap("RdBu_r", len(bounds) - 1)
    norm = BoundaryNorm(bounds, cmap.N, clip=True)

    fig, ax = plt.subplots(figsize=(7.0, 6.6))
    im = ax.imshow(arr, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([lab for _, lab in metrics], rotation=25, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=6.5)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i,j]:+.2f}", ha="center", va="center", fontsize=5.8,
                    color="white" if abs(arr[i,j]) >= 0.6 else "black")

    ax.axhline(7.5, color="k", lw=0.8)
    ax.set_title("Seasonal EOF correlations with physical TCWV metrics", fontsize=9)
    cb = fig.colorbar(im, ax=ax, pad=0.02, boundaries=bounds, ticks=bounds)
    cb.set_label("Pearson correlation", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)

    base = outdir / "figS2_tcwv_seasonal_mode_diagnostics_v1_1"
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    data, summary = load_inputs(args.modes_dir)
    plot_monthly_modes(data, summary, args.series_type, args.outdir, args.dpi)
    plot_seasonal_diagnostics(summary, args.outdir, args.dpi)
    print(f"plot_atlantic_itcz_fig1_support_v1_1.py version {SCRIPT_VERSION}")
    print(f"Wrote supporting figures to {args.outdir}")


if __name__ == "__main__":
    main()
