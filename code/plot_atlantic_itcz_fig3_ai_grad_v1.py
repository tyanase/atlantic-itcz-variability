#!/usr/bin/env python3
"""Plot Fig. 3: daily TCWV maps at monthly AI_grad extremes.

Stage 4 stores gradient anisotropy as Agrad; labels use the manuscript name
AI_grad. For each SON month, the low and high rows use the minimum and maximum
over all available years at the specified center-following window (rel_5deg
by default). Window width is fixed here; sensitivity is shown separately.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D

plt.rcParams["font.size"] = 12

DEFAULT_WINDOW = "rel_5deg"
MONTHS = (9, 10, 11)
MONTH_LABELS = {9: "Sep", 10: "Oct", 11: "Nov"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare-dir", type=Path, default=Path("out_atlantic_itcz/01_prepare"))
    p.add_argument("--stage4-dir", type=Path, required=True)
    p.add_argument("--metrics-file", default="daily_organization_metrics.nc")
    p.add_argument("--static-file", default="static_fields.nc")
    p.add_argument("--window", default=DEFAULT_WINDOW)
    p.add_argument("--months", type=int, nargs="+", default=MONTHS)
    p.add_argument("--precip-levels", type=float, nargs="+", default=(5.0, 20.0))
    p.add_argument("--tcwv-min", type=float, default=20.0)
    p.add_argument("--tcwv-max", type=float, default=65.0)
    p.add_argument("--tcwv-step", type=float, default=5.0)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--output-stem", default="fig3_daily_ai_grad_v1")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def _as_str(values):
    out = []
    for x in np.asarray(values):
        out.append(x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x))
    return np.asarray(out)


def _find_metric_name(ds, candidates, label):
    for name in candidates:
        if name in ds:
            return name
    raise KeyError(f"Could not find {label}. Tried: {candidates}")


def read_daily_metrics(path: Path, window: str) -> pd.DataFrame:
    ds = xr.open_dataset(path)
    windows = _as_str(ds["window"].values)
    matches = np.where(windows == window)[0]
    if len(matches) != 1:
        raise ValueError(
            f"Window {window!r} not found uniquely in {path}. "
            f"Available windows: {list(windows)}"
        )
    iwin = int(matches[0])

    ai_grad_name = _find_metric_name(ds, ["Agrad", "agrad", "A_grad", "AI_grad"], "AI_grad")
    time = pd.to_datetime(ds["time"].values)
    df = pd.DataFrame({
        "time": time,
        "year": time.year,
        "month": time.month,
        "AI_grad": np.asarray(ds[ai_grad_name].isel(window=iwin).values, float),
    }).replace([np.inf, -np.inf], np.nan)
    ds.close()
    return df


def select_days(df, months):
    son = df[df["month"].isin(months) & np.isfinite(df["AI_grad"])].copy()
    low, high = [], []
    for month in months:
        sub = son[son["month"] == month]
        if sub.empty:
            raise ValueError(f"No valid samples for month={month}")
        lo = sub.loc[sub["AI_grad"].idxmin()].copy()
        hi = sub.loc[sub["AI_grad"].idxmax()].copy()
        lo["selection_rule"] = "monthly_minimum_AI_grad"
        hi["selection_rule"] = "monthly_maximum_AI_grad"
        low.append(lo)
        high.append(hi)
    return (
        pd.DataFrame(low).sort_values("month").reset_index(drop=True),
        pd.DataFrame(high).sort_values("month").reset_index(drop=True),
    )


def load_map_fields(prepare_daily_dir, static_file, times):
    static = xr.open_dataset(static_file)
    ocean_mask = np.asarray(static["ocean_mask"].values, bool)
    lat = np.asarray(static["lat"].values, float)
    lon = np.asarray(static["lon"].values, float)
    static.close()

    year_cache = {}
    fields = {}
    try:
        for t in times:
            year = int(t.year)
            if year not in year_cache:
                f = prepare_daily_dir / f"era5_daily_{year}.nc"
                if not f.exists():
                    raise FileNotFoundError(f)
                year_cache[year] = xr.open_dataset(f)
            sel = year_cache[year].sel(time=np.datetime64(t))
            fields[pd.Timestamp(t)] = {
                "tcwv": np.where(ocean_mask, np.asarray(sel["tcwv"].values, float), np.nan),
                "tp": np.where(ocean_mask, np.asarray(sel["tp"].values, float), np.nan),
            }
    finally:
        for ds in year_cache.values():
            ds.close()
    return fields, lon, lat


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = read_daily_metrics(args.stage4_dir / args.metrics_file, args.window)
    low_df, high_df = select_days(df, list(args.months))

    times = [pd.Timestamp(x) for x in pd.concat([low_df["time"], high_df["time"]])]
    fields, lon, lat = load_map_fields(
        args.prepare_dir / "daily",
        args.prepare_dir / args.static_file,
        times,
    )

    tcwv_levels = np.arange(args.tcwv_min, args.tcwv_max + 0.5 * args.tcwv_step, args.tcwv_step)
    plevs = np.asarray(sorted(args.precip_levels), float)
    pwidths = np.linspace(0.6, 1.3, len(plevs))

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.88, bottom=0.18, top=0.91, wspace=0.12, hspace=0.28)

    rows = [low_df, high_df]
    row_labels = [r"Low $\mathrm{AI}_{\mathrm{grad}}$", r"High $\mathrm{AI}_{\mathrm{grad}}$"]
    cf = None
    panel = 0

    for ir, sdf in enumerate(rows):
        for ic, (_, row) in enumerate(sdf.iterrows()):
            ax = axes[ir, ic]
            panel += 1
            t = pd.Timestamp(row["time"])
            q = fields[t]["tcwv"]
            p = fields[t]["tp"]

            cf = ax.contourf(lon, lat, q, levels=tcwv_levels, cmap="BrBG", extend="both")
            if len(plevs):
                ax.contour(lon, lat, p, levels=plevs, colors="k", linewidths=pwidths)

            ax.set_title(f"({chr(96 + panel)}) {t.strftime('%d %b %Y')}", pad=6)
            ax.text(
                0.03, 0.96,
                rf"$\mathrm{{AI}}_{{\mathrm{{grad}}}}={row['AI_grad']:.2f}$",
                transform=ax.transAxes, ha="left", va="top", fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.2),
            )
            ax.set_xlim(-50.0, -20.0)
            ax.set_ylim(-15.0, 30.0)

    for ax in axes[-1, :]:
        ax.set_xlabel("Longitude / deg")
    for ax in axes[:, 0]:
        ax.set_ylabel("Latitude / deg")

    fig.text(0.015, 0.72, row_labels[0], rotation=90, ha="center", va="center", fontsize=11)
    fig.text(0.015, 0.30, row_labels[1], rotation=90, ha="center", va="center", fontsize=11)

    cax = fig.add_axes([0.895, 0.17, 0.02, 0.62])
    cbar = fig.colorbar(cf, cax=cax)
    cbar.set_label(r"TCWV / (kg m$^{-2}$)")

    if len(plevs):
        handles = [
            Line2D([0], [0], color="k", lw=lw, label=rf"{lev:g} mm day$^{{-1}}$")
            for lev, lw in zip(plevs, pwidths)
        ]
        fig.legend(
            handles=handles, title="Precipitation", loc="lower center",
            bbox_to_anchor=(0.50, 0.006), ncol=len(handles), frameon=False,
        )

    png = args.outdir / f"{args.output_stem}.png"
    pdf = args.outdir / f"{args.output_stem}.pdf"
    fig.savefig(png, dpi=args.dpi)
    fig.savefig(pdf)
    plt.close(fig)

    summary = pd.concat(
        [low_df.assign(selection="low"), high_df.assign(selection="high")],
        ignore_index=True,
    )
    summary.to_csv(args.outdir / f"{args.output_stem}_selected_days.csv", index=False)

    print("[INFO] Selected days:")
    for _, row in summary.sort_values(["selection", "month"]).iterrows():
        print(
            f"  {row['selection']:>4s} {MONTH_LABELS.get(int(row['month']), row['month'])}: "
            f"{pd.Timestamp(row['time']).strftime('%Y-%m-%d')} "
            f"AI_grad={row['AI_grad']:.3f}"
        )
    print(f"[INFO] Wrote {png}")
    print(f"[INFO] Wrote {pdf}")


if __name__ == "__main__":
    main()
