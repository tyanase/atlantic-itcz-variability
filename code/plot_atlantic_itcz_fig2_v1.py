#!/usr/bin/env python3
"""Plot Fig. 2 and seasonal SST-PC scale relationships (Fig. S3).

Inputs are seasonal_pc12_sst_space_metrics.csv and
seasonal_pc12_sst_space_scale_sensitivity.csv from Stage 3.

Fig. 2 compares MAM EOF1 with SST asymmetry and SON EOF1 with SST level, along
with their cross-predictor controls and width sensitivity. Each control uses
the same SST half-width as its corresponding main relationship.

The coordinates are T_level = (T_N+T_S)/2 and T_asym = (T_N-T_S)/2, where T_N
and T_S are SST means in bands north and south of the fixed climatological
TCWV-peak latitude. Fig. S3 shows both predictors for all four seasons and
EOF1/EOF2. The direct CLI also writes auxiliary SST diagnostics; the package
wrapper calls only the Fig. 2 and S3 plotting functions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_VERSION = "1.12.0"
SEASONS = ["DJF", "MAM", "JJA", "SON"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--state-dir", type=Path, required=True,
                   help="Stage-3 output directory")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--series-type", default="raw",
                   choices=["raw", "linear_detrended"])
    p.add_argument("--mam-scale-deg", type=float, default=None,
                   help="Optional MAM scatter half-width. If omitted, use the scale maximizing R^2_asym.")
    p.add_argument("--son-scale-deg", type=float, default=None,
                   help="Optional SON scatter half-width. If omitted, use the scale maximizing R^2_level.")
    p.add_argument("--dpi", type=int, default=240)
    return p.parse_args()


def require_columns(df: pd.DataFrame, cols: list[str], fname: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{fname} is missing columns required by this plotting script: {missing}\n"
            "Re-run Stage 3 with atlantic_itcz_state_analysis_v1_8.py or later."
        )


def save(fig, outdir: Path, stem: str, dpi: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def nearest_scale(df: pd.DataFrame, target: float) -> float:
    scales = np.sort(df["scale_deg"].dropna().unique().astype(float))
    if scales.size == 0:
        raise ValueError("No finite scale_deg values found.")
    return float(scales[np.argmin(np.abs(scales - target))])


def standardize_pc(values: np.ndarray) -> np.ndarray:
    """Return a PC standardized by its interannual sample standard deviation."""
    x = np.asarray(values, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    good = np.isfinite(x)
    if good.sum() < 2:
        return out
    sd = np.std(x[good], ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return out
    out[good] = (x[good] - np.mean(x[good])) / sd
    return out


def objective_scale(rel: pd.DataFrame, season: str, mode: int,
                    series_type: str, metric_col: str) -> float:
    """Select the predefined sweep scale that maximizes the target single-predictor R^2."""
    d = subset_rel(rel, season, mode, series_type)
    good = d[np.isfinite(d[metric_col])].copy()
    if good.empty:
        raise ValueError(f"No finite {metric_col} values for {season} EOF{mode}.")
    return float(good.loc[good[metric_col].idxmax(), "scale_deg"])


def subset_rel(rel, season, mode, series_type, sst_space="tcwv_centered_ns"):
    x = rel[
        (rel["season"] == season)
        & (rel["mode"].astype(int) == int(mode))
        & (rel["series_type"] == series_type)
        & (rel["sst_space"] == sst_space)
    ].copy()
    return x.sort_values("scale_deg")


def subset_metrics(metrics, season, mode, series_type, scale, sst_space="tcwv_centered_ns"):
    x = metrics[
        (metrics["season"] == season)
        & (metrics["mode"].astype(int) == int(mode))
        & (metrics["series_type"] == series_type)
        & (metrics["sst_space"] == sst_space)
    ].copy()
    use_scale = nearest_scale(x, scale)
    x = x[np.isclose(x["scale_deg"].astype(float), use_scale)].copy()
    return x.sort_values("year"), use_scale


def linfit(ax, x, y, color=None):
    good = np.isfinite(x) & np.isfinite(y)
    xx = np.asarray(x)[good]
    yy = np.asarray(y)[good]
    if len(xx) < 3:
        return np.nan
    p = np.polyfit(xx, yy, 1)
    gx = np.linspace(xx.min(), xx.max(), 100)
    ax.plot(gx, p[0] * gx + p[1], lw=1.4, color=color)
    r = np.corrcoef(xx, yy)[0, 1]
    return float(r * r)


def _overlap_area(a, b):
    """Overlap area of two boxes (xmin, xmax, ymin, ymax) in axes coordinates."""
    dx = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    dy = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    return dx * dy


def scatter_all_years(ax, x: np.ndarray, y: np.ndarray) -> None:
    """Plot all years with the same neutral marker style."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if not good.any():
        return

    ax.scatter(
        x[good], y[good],
        s=20,
        facecolors="0.65",
        edgecolors="none",
        alpha=0.75,
        zorder=2,
    )

def panel_label(ax, label):
    ax.text(-0.15, 1.06, f"({label})", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=11, fontweight="bold")


def plot_main(metrics, rel, args):
    """Plot MAM/SON SST-PC relationships, controls, and width sensitivity.

    Each control uses the half-width selected for its corresponding main pair;
    the control does not have a separately optimized width.
    """
    mam_target = (args.mam_scale_deg if args.mam_scale_deg is not None
                  else objective_scale(rel, "MAM", 1, args.series_type, "r2_asym"))
    son_target = (args.son_scale_deg if args.son_scale_deg is not None
                  else objective_scale(rel, "SON", 1, args.series_type, "r2_level"))
    mam, mam_scale = subset_metrics(metrics, "MAM", 1, args.series_type, mam_target)
    son, son_scale = subset_metrics(metrics, "SON", 1, args.series_type, son_target)
    for frame, season in [(mam, "MAM"), (son, "SON")]:
        if frame.empty:
            raise ValueError(f"No {season} EOF1 data available")
        if frame.year.duplicated().any():
            raise ValueError(f"Duplicate {season} EOF1 years at chosen scale")

    fig, axs = plt.subplots(3, 2, figsize=(7.6, 8.8))
    fig.subplots_adjust(left=0.125, right=0.98, bottom=0.07, top=0.97,
                        wspace=0.32, hspace=0.50)

    specs = [
        # Main SST-PC relationships.
        (axs[0, 0], "a", "MAM", mam, mam_scale, "sst_asym_K", "asym", "C0"),
        (axs[0, 1], "b", "SON", son, son_scale, "sst_level_K", "level", "C1"),
        # The cross-comparisons use the same year samples and d values.
        (axs[1, 0], "c", "MAM", mam, mam_scale, "sst_level_K", "level", "C1"),
        (axs[1, 1], "d", "SON", son, son_scale, "sst_asym_K", "asym", "C0"),
    ]
    for ax, label, season, frame, scale, predictor, name, color in specs:
        xx = frame[predictor].to_numpy(float)
        yy = standardize_pc(frame["pc"].to_numpy(float))
        if (np.isfinite(xx) & np.isfinite(yy)).sum() < 3:
            raise ValueError(f"Not enough valid {season} paired values for {name}")
        scatter_all_years(ax, xx, yy)
        r2 = linfit(ax, xx, yy, color=color)
        ax.axhline(0, lw=0.7, ls=":", color="0.5")
        ax.set_xlabel(rf"SST {('asymmetry' if name == 'asym' else 'level')}, "
                      + (r"$T_{\rm asym}$ / K" if name == "asym" else r"$T_{\rm level}$ / K"))
        ax.set_ylabel(f"Standardized {season} EOF1 PC")
        mode_desc = "shift" if season == "MAM" else "shape"
        ax.set_title(f"({label}) {season} EOF1 ({mode_desc}), d = {scale:g}°", fontsize=10)
        ax.text(0.04, 0.94, rf"$R^2={r2:.2f}$", transform=ax.transAxes,
                va="top", ha="left", fontsize=10)
        if season == "SON" and name == "level":
            ax.set_xticks([300, 300.5, 301, 301.5])
        print(f"[CHECK] Fig2{label}: {season} EOF1 vs SST {name}, d={scale:g}: R2={r2:.5f}")

    for ax, season, label, title in [
        (axs[2, 0], "MAM", "e", "MAM EOF1 (shift)"),
        (axs[2, 1], "SON", "f", "SON EOF1 (shape)"),
    ]:
        selected = subset_rel(rel, season, 1, args.series_type)
        if selected.empty:
            raise ValueError(f"Missing SST-width sensitivity for {season} EOF1")
        ax.plot(selected["scale_deg"], selected["r2_asym"], marker="o", ms=3,
                color="C0", label=r"SST asymmetry $T_{\rm asym}$")
        ax.plot(selected["scale_deg"], selected["r2_level"], marker="o", ms=3,
                color="C1", label=r"SST level $T_{\rm level}$")
        ax.set_xlabel(r"SST half-width, $d$ / deg")
        ax.set_ylabel(r"$R^2$")
        ax.set_ylim(bottom=0)
        ax.set_title(f"({label}) {title}", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="best")

    save(fig, args.outdir, "fig2_sst_shift_shape_v1", args.dpi)


def plot_all_seasons(rel, args):
    fig, axs = plt.subplots(4, 2, figsize=(7.3, 9.0), sharex=True, sharey=True)
    for i, season in enumerate(SEASONS):
        for j, mode in enumerate([1, 2]):
            ax = axs[i, j]
            d = subset_rel(rel, season, mode, args.series_type)
            if d.empty:
                ax.set_visible(False)
                continue
            mode_type = str(d.iloc[0].get("mode_type", ""))
            evf = float(d.iloc[0].get("explained_variance_fraction", np.nan))
            ax.plot(d["scale_deg"], d["r2_asym"], marker="o", ms=2.5,
                    label=r"$T_{\rm asym}$")
            ax.plot(d["scale_deg"], d["r2_level"], marker="o", ms=2.5,
                    label=r"$T_{\rm level}$")
            ax.set_ylim(0, 1)
            panel_idx = i * 2 + j
            panel_letter = chr(ord("a") + panel_idx)
            ax.set_title(
                f"({panel_letter}) {season} EOF{mode} {mode_type} ({100*evf:.0f}%)",
                fontsize=9,
            )
            if j == 0:
                ax.set_ylabel(r"$R^2$")
            if i == 3:
                ax.set_xlabel(r"SST half-width, $d$ / deg")
    axs[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("SST level and asymmetry relationships for all seasonal EOF1/EOF2 modes",
                 y=0.995, fontsize=11)
    fig.tight_layout()
    save(fig, args.outdir, "figS_sst_all_seasons_pc12_v1", args.dpi)


def plot_joint_skill(rel, args):
    fig, axs = plt.subplots(1, 2, figsize=(7.3, 3.2), sharey=True)
    for ax, season, label, title in [
        (axs[0], "MAM", "a", "MAM EOF1 (shift)"),
        (axs[1], "SON", "b", "SON EOF1 (shape)"),
    ]:
        d = subset_rel(rel, season, 1, args.series_type)
        ax.plot(d["scale_deg"], d["r2_asym"], marker="o", ms=3,
                label=r"$T_{\rm asym}$ only")
        ax.plot(d["scale_deg"], d["r2_level"], marker="o", ms=3,
                label=r"$T_{\rm level}$ only")
        ax.plot(d["scale_deg"], d["r2_joint"], marker="o", ms=3,
                label="Joint")
        ax.set_xlabel(r"SST half-width, $d$ / deg")
        ax.set_title(f"({label}) {title}")
        ax.set_ylim(0, 1)
    axs[0].set_ylabel(r"$R^2$")
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, args.outdir, "figS_sst_joint_skill_v1", args.dpi)


def plot_peak_contrast(rel, args):
    d = subset_rel(rel, "SON", 1, args.series_type, sst_space="peak_contrast")
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.3))
    # Compare single-predictor and joint R2.
    ax.plot(d["scale_deg"], d["r2_peak"], marker="o", ms=3,
            label="Exact SST peak")
    if "r2_contrast" in d.columns:
        ax.plot(d["scale_deg"], d["r2_contrast"], marker="o", ms=3,
                label="Peak-to-flank contrast")
    ax.plot(d["scale_deg"], d["r2_joint"], marker="o", ms=3,
            label="Joint")
    ax.set_xlabel("Flank distance / deg")
    ax.set_ylabel(r"$R^2$")
    ax.set_ylim(0, 1)
    ax.set_title("SON EOF1 (shape): SST peak and peak-to-flank contrast")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, args.outdir, "figS_sst_peak_contrast_son_v1", args.dpi)


def plot_peak_level_bridge(rel, args):
    d = subset_rel(rel, "SON", 1, args.series_type)
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    ax.plot(d["scale_deg"], d["r2_peak_level"], marker="o", ms=3,
            label=r"$T_{\rm level}$ vs exact SST peak")
    ax.plot(d["scale_deg"], d["r2_pc_peak_same_sample"], marker="o", ms=3,
            label="SON EOF1 PC vs exact SST peak")
    ax.plot(d["scale_deg"], d["r2_level"], marker="o", ms=3,
            label=r"SON EOF1 PC vs $T_{\rm level}$")
    ax.set_xlabel(r"SST half-width, $d$ / deg")
    ax.set_ylabel(r"$R^2$")
    ax.set_ylim(0, 1.05)
    ax.set_title("SON warm-state correspondence")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, args.outdir, "figS_sst_peak_level_bridge_son_v1", args.dpi)


def main():
    args = parse_args()
    metrics_file = args.state_dir / "seasonal_pc12_sst_space_metrics.csv"
    rel_file = args.state_dir / "seasonal_pc12_sst_space_scale_sensitivity.csv"

    if not metrics_file.exists() or not rel_file.exists():
        raise FileNotFoundError(
            "Required Stage-3 files not found:\n"
            f"  {metrics_file}\n"
            f"  {rel_file}\n"
            "Run atlantic_itcz_state_analysis_v1_8.py (or later) first."
        )

    metrics = pd.read_csv(metrics_file)
    rel = pd.read_csv(rel_file)

    require_columns(
        metrics,
        ["season", "mode", "series_type", "sst_space", "scale_deg", "year",
         "pc", "sst_level_K", "sst_asym_K", "peak_sst_K", "contrast_K"],
        metrics_file.name,
    )
    require_columns(
        rel,
        ["season", "mode", "series_type", "sst_space", "scale_deg",
         "r2_asym", "r2_level", "r2_joint",
         "r2_peak_level", "r2_pc_peak_same_sample"],
        rel_file.name,
    )

    plot_main(metrics, rel, args)
    plot_all_seasons(rel, args)
    plot_joint_skill(rel, args)
    plot_peak_contrast(rel, args)
    plot_peak_level_bridge(rel, args)

    print(f"plot_atlantic_itcz_fig2_v1_12.py version {SCRIPT_VERSION}")
    print(f"Output directory: {args.outdir}")


if __name__ == "__main__":
    main()
