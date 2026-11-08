#!/usr/bin/env python3
"""Relate SON SST, rainbelt shape, and daily moisture organization.

The reference SST predictor is T_level at a 12.5-degree half-width, selected
from the Stage 3 SST-PC relationship rather than SST-organization correlations.
The reference organization half-width is 5 degrees. Outputs also include
T_asym, SST-width sensitivity, detrended correlations, and descriptive
PC-adjusted partial correlations. Neither correlations nor partial
correlations establish causal direction.

Inputs are the Stage 3 seasonal_pc12_sst_space_metrics.csv and the Stage 4
organization_statistics.csv and organization_event_frequency.csv. Stage 1/2
fields and daily NetCDF files are not reprocessed.

Example:
    python -B code/atlantic_itcz_sst_organization.py \
        --state-dir /path/to/stage3 --stage4-dir /path/to/stage4 \
        --outdir /path/to/new/stage5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

SCRIPT_VERSION = "1.0.0"
STATS = ("Agrad_mean", "AI_PW_mean", "Hx_mean")
EVENTS = (
    "high_Agrad_fraction", "low_Agrad_fraction",
    "high_AI_PW_fraction", "low_AI_PW_fraction",
    "high_Hx_fraction", "low_Hx_fraction",
)
RESPONSES = STATS + EVENTS
SST_COLS = ("sst_level_K", "sst_asym_K")
LABELS = {
    "Agrad_mean": r"SON-mean $\mathrm{AI}_{\mathrm{grad}}$",
    "AI_PW_mean": r"SON-mean $\mathrm{AI}_{\mathrm{var}}$",
    "Hx_mean": r"SON-mean $H_x$",
    "high_Agrad_fraction": r"High-$\mathrm{AI}_{\mathrm{grad}}$ days / %",
    "low_Agrad_fraction": r"Low-$\mathrm{AI}_{\mathrm{grad}}$ days / %",
    "high_AI_PW_fraction": r"High-$\mathrm{AI}_{\mathrm{var}}$ days / %",
    "low_AI_PW_fraction": r"Low-$\mathrm{AI}_{\mathrm{var}}$ days / %",
    "high_Hx_fraction": r"High-$H_x$ days / %",
    "low_Hx_fraction": r"Low-$H_x$ days / %",
}
PANEL_TITLES = (
    r"Seasonal-mean $\mathrm{AI}_{\mathrm{grad}}$",
    r"Seasonal-mean $H_x$",
    r"High-$\mathrm{AI}_{\mathrm{grad}}$ days",
    r"Low-$\mathrm{AI}_{\mathrm{grad}}$ days",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state-dir", type=Path,
                        default=Path("out_atlantic_itcz/03_state_latm15_30"))
    parser.add_argument("--stage4-dir", type=Path,
                        default=Path("out_atlantic_itcz/04_daily_organization_width_sensitivity_events"))
    parser.add_argument("--outdir", type=Path,
                        default=Path("out_atlantic_itcz/05_sst_organization"))
    parser.add_argument("--sst-half-width", type=float, default=12.5,
                        help="Predeclared SST half-width in deg, matching Fig. 2b")
    parser.add_argument("--organization-window", default="rel_5deg")
    parser.add_argument("--season", default="SON", help="Default manuscript season: SON")
    parser.add_argument("--mode", type=int, default=1, choices=(1, 2),
                        help="SON shape mode is EOF1 in the current manuscript")
    parser.add_argument("--series-type", default="raw")
    parser.add_argument("--dpi", type=int, default=250)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def require_columns(df: pd.DataFrame, cols: tuple[str, ...] | list[str], filename: str) -> None:
    missing = sorted(set(cols).difference(df.columns))
    if missing:
        raise ValueError(f"{filename} is missing columns {missing}. Present: {list(df.columns)}")


def assert_unique_year(df: pd.DataFrame, source: str) -> None:
    duplicates = df.loc[df["year"].duplicated(keep=False), "year"].unique().tolist()
    if duplicates:
        raise ValueError(f"Duplicate year rows in {source}: {duplicates[:12]}. "
                         "Inspect filters and source CSV rather than silently averaging them.")


def load_stage3(args: argparse.Namespace) -> pd.DataFrame:
    filename = args.state_dir / "seasonal_pc12_sst_space_metrics.csv"
    s = read_csv(filename)
    require_columns(s, ["year", "season", "mode", "series_type", "sst_space",
                        "scale_deg", "pc", *SST_COLS], str(filename))
    cond = (
        s["season"].astype(str).eq(args.season)
        & pd.to_numeric(s["mode"], errors="coerce").eq(args.mode)
        & s["series_type"].astype(str).eq(args.series_type)
        & s["sst_space"].astype(str).eq("tcwv_centered_ns")
    )
    if "reference" in s:
        cond &= s["reference"].astype(str).eq("tcwv_clim_peak")
    s = s.loc[cond].copy()
    if s.empty:
        raise ValueError(f"No matching Stage-3 rows for season={args.season}, "
                         f"mode={args.mode}, series_type={args.series_type}, "
                         "sst_space=tcwv_centered_ns")
    if "mode_type" in s:
        modes = s["mode_type"].dropna().unique().tolist()
        if modes != ["shape"]:
            print(f"[WARNING] Expected SON EOF1 'shape'; Stage-3 mode_type values={modes}")
    for col in ["year", "scale_deg", "pc", *SST_COLS]:
        s[col] = pd.to_numeric(s[col], errors="coerce")
    if s["year"].isna().any():
        raise ValueError("Stage-3 year contains nonnumeric values")
    s["year"] = s["year"].astype(int)
    for scale, subset in s.groupby("scale_deg"):
        assert_unique_year(subset, f"Stage-3 SST d={scale}")
    return s[["year", "scale_deg", "pc", *SST_COLS]].copy()


def load_stage4(args: argparse.Namespace) -> pd.DataFrame:
    fstats = args.stage4_dir / "organization_statistics.csv"
    fevents = args.stage4_dir / "organization_event_frequency.csv"
    stats = read_csv(fstats)
    events = read_csv(fevents)
    filters = ["year", "period_type", "period", "window"]
    require_columns(stats, filters + list(STATS), str(fstats))
    require_columns(events, filters + list(EVENTS), str(fevents))
    for source, df in ((str(fstats), stats), (str(fevents), events)):
        matches = df.loc[
            df["period_type"].astype(str).eq("seasonal")
            & df["period"].astype(str).eq(args.season)
            & df["window"].astype(str).eq(args.organization_window)
        ].copy()
        if matches.empty:
            available = sorted(df.loc[df["period"].astype(str).eq(args.season), "window"].astype(str).unique())
            raise ValueError(f"No season={args.season}, window={args.organization_window} "
                             f"rows in {source}. Available windows: {available}")
        matches["year"] = pd.to_numeric(matches["year"], errors="raise").astype(int)
        assert_unique_year(matches, source)
        if df is stats:
            filtered_stats = matches[["year", *STATS]].copy()
        else:
            filtered_events = matches[["year", *EVENTS]].copy()
    out = filtered_stats.merge(filtered_events, on="year", how="inner", validate="one_to_one")
    if out.empty:
        raise ValueError("No matching years in Stage-4 statistics and event-frequency CSVs")
    return out


def get_yearly(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray]:
    sst = load_stage3(args)
    org = load_stage4(args)
    joined = sst.merge(org, on="year", how="inner", validate="many_to_one")
    if joined.empty:
        raise ValueError("No common years between Stage-3 SST and Stage-4 organization")
    # Require the same PC at every SST averaging width.
    nunique = joined.groupby("year")["pc"].agg(lambda v: v.max() - v.min())
    if bool((nunique.abs() > 1e-6).any()):
        raise ValueError("Stage-3 PC changes across SST widths for the same year")
    scales = np.sort(joined["scale_deg"].dropna().unique())
    if not np.any(np.isclose(scales, args.sst_half_width, atol=1e-8, rtol=0)):
        raise ValueError(f"SST half-width={args.sst_half_width:g} deg not present. "
                         f"Available: {scales.tolist()}")
    return joined, scales


def finite_triplet(x, y, z):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return x[valid], y[valid], z[valid]


def detrend_vector(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    # Subtract fitted linear trend; retain mean; don't rely on uniform year spacing.
    values = np.asarray(values, float)
    years = np.asarray(years, float)
    result = np.full(values.size, np.nan)
    ok = np.isfinite(values) & np.isfinite(years)
    if ok.sum() >= 3:
        t = years[ok] - years[ok].mean()
        slope = float(np.dot(t, values[ok] - values[ok].mean()) / np.dot(t, t))
        result[ok] = values[ok] - slope * t
    return result


def correlations(x, y, years, pc) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    years = np.asarray(years, dtype=float)
    pc = np.asarray(pc, dtype=float)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(years) & np.isfinite(pc)
    x, y, years, pc = x[good], y[good], years[good], pc[good]
    out = {"n": int(len(x)), "r": np.nan, "p": np.nan, "r2": np.nan,
           "spearman_r": np.nan, "spearman_p": np.nan,
           "detrended_r": np.nan, "detrended_p": np.nan,
           "pc_adjusted_partial_r": np.nan, "pc_adjusted_partial_p": np.nan}
    if len(x) < 4 or np.std(x) <= 1e-10 or np.std(y) <= 1e-10:
        return out
    pr = pearsonr(x, y)
    sr = spearmanr(x, y)
    out.update(r=float(pr.statistic), p=float(pr.pvalue), r2=float(pr.statistic ** 2),
               spearman_r=float(sr.statistic), spearman_p=float(sr.pvalue))
    xd, yd = detrend_vector(x, years), detrend_vector(y, years)
    if np.std(xd) > 1e-10 and np.std(yd) > 1e-10:
        detr = pearsonr(xd, yd)
        out.update(detrended_r=float(detr.statistic), detrended_p=float(detr.pvalue))
    # Diagnostic only: conditioning on the shape PC may block a physical pathway.
    if np.isfinite(pc).all() and np.std(pc) > 1e-10:
        ac = np.column_stack((np.ones(len(pc)), pc))
        residual_x = x - ac @ np.linalg.lstsq(ac, x, rcond=None)[0]
        residual_y = y - ac @ np.linalg.lstsq(ac, y, rcond=None)[0]
        if np.std(residual_x) > 1e-10 and np.std(residual_y) > 1e-10:
            partial = pearsonr(residual_x, residual_y)
            out["pc_adjusted_partial_r"] = float(partial.statistic)
            # Partial correlation controlling 1 variable has df=n-3.
            from scipy.stats import t as student_t
            r = float(partial.statistic)
            df = len(pc) - 3
            statistic = abs(r) * np.sqrt(df / max(1e-15, 1 - r*r))
            out["pc_adjusted_partial_p"] = float(2 * student_t.sf(statistic, df))
    return out


def yearly_reference(full: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    d = full[np.isclose(full["scale_deg"].to_numpy(float), args.sst_half_width, atol=1e-8, rtol=0)].copy()
    d = d.sort_values("year").reset_index(drop=True)
    assert_unique_year(d, "reference SST half-width")
    pc = d["pc"].to_numpy(float)
    ok = np.isfinite(pc)
    if ok.sum() < 2:
        raise ValueError("Insufficient SON PC samples")
    sd = float(np.std(pc[ok], ddof=1))
    d["shape_pc_sd"] = (pc - float(np.mean(pc[ok]))) / sd
    d.rename(columns={"pc": "shape_pc_raw"}, inplace=True)
    print(f"[INFO] Reference sample: SST half-width={args.sst_half_width:g}°, "
          f"organization window={args.organization_window}, "
          f"n={len(d)}, years={d.year.min()}–{d.year.max()}")
    return d


def scale_correlations(full: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    records: list[dict] = []
    for scale, d in full.groupby("scale_deg", sort=True):
        for predictor in SST_COLS:
            for response in RESPONSES:
                stats = correlations(d[predictor], d[response], d["year"], d["pc"])
                records.append({"sst_half_width_deg": scale, "sst_predictor": predictor,
                                "organization_measure": response, "organization_window": args.organization_window,
                                **stats})
    return pd.DataFrame(records)


def three_way_correlations(d: pd.DataFrame) -> pd.DataFrame:
    out = []
    triples = [("sst_level_K", "shape_pc_sd"), ("sst_asym_K", "shape_pc_sd")]
    triples += [("shape_pc_sd", response) for response in RESPONSES]
    triples += [("sst_level_K", response) for response in RESPONSES]
    for x, y in triples:
        stat = correlations(d[x], d[y], d["year"], d["shape_pc_sd"])
        out.append({"x": x, "y": y, "n": stat["n"], "r": stat["r"], "p": stat["p"],
                    "detrended_r": stat["detrended_r"], "detrended_p": stat["detrended_p"]})
    return pd.DataFrame(out)


def save_plot(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_scatter(d: pd.DataFrame, args: argparse.Namespace) -> None:
    specs = ["Agrad_mean", "Hx_mean", "high_Agrad_fraction", "low_Agrad_fraction"]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.3), constrained_layout=True)
    x = d["sst_level_K"].to_numpy(float)
    for i, (ax, variable) in enumerate(zip(axes.flat, specs)):
        y = d[variable].to_numpy(float) * (100 if variable.endswith("_fraction") else 1)
        good = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[good], y[good], s=25, alpha=0.8, color="0.55", edgecolors="none")
        if good.sum() >= 3 and np.std(x[good]) > 1e-10 and np.std(y[good]) > 1e-10:
            slope, intercept = np.polyfit(x[good], y[good], deg=1)
            xfit = np.linspace(np.nanmin(x[good]), np.nanmax(x[good]), 100)
            ax.plot(xfit, slope*xfit + intercept, linewidth=1.4)
        stats = correlations(x, y, d.year, d.shape_pc_sd)
        r, rd = stats["r"], stats["detrended_r"]
        ax.text(0.04, 0.96, f"r = {r:+.2f} (n={stats['n']})\n"
                             f"detrended r = {rd:+.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9)
        ax.set_title(f"({chr(97+i)}) {PANEL_TITLES[i]}")
        ax.set_xlabel(r"SON SST level, $T_{\rm level}$ / K " + f"($d={args.sst_half_width:g}°$)")
        ax.set_ylabel(LABELS[variable])
        ax.grid(alpha=0.2, linewidth=0.6)
    save_plot(fig, args.outdir / "sst_organization_reference_scatter", args.dpi)


def plot_scale(summary: pd.DataFrame, args: argparse.Namespace) -> None:
    specs = ["Agrad_mean", "Hx_mean", "high_Agrad_fraction", "low_Agrad_fraction"]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.3), constrained_layout=True)
    for i, (ax, target) in enumerate(zip(axes.flat, specs)):
        for predictor, label, style in [
            ("sst_level_K", r"$T_{\rm level}$", "-"),
            ("sst_asym_K", r"$T_{\rm asym}$", "--"),
        ]:
            sub = summary[(summary["sst_predictor"] == predictor)
                          & (summary["organization_measure"] == target)]
            ax.plot(sub["sst_half_width_deg"], sub["r"], linestyle=style,
                    marker="o", ms=3, label=label)
        ax.axvline(args.sst_half_width, color="0.4", linestyle=":", linewidth=0.9)
        ax.axhline(0, color="0.6", linewidth=0.7)
        ax.set_ylim(-1, 1)
        ax.set_title(f"({chr(97+i)}) {PANEL_TITLES[i]}")
        ax.set_xlabel("SST half-width, d / deg")
        ax.set_ylabel("Pearson r")
        ax.grid(alpha=0.2, linewidth=0.6)
    axes[0, 0].legend(frameon=False, fontsize=9)
    save_plot(fig, args.outdir / "sst_organization_sst_scale_sensitivity", args.dpi)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    full, scales = get_yearly(args)
    ref = yearly_reference(full, args)
    summary = scale_correlations(full, args)
    triad = three_way_correlations(ref)
    full.sort_values(["scale_deg", "year"]).to_csv(args.outdir / "yearly_joined_all_sst_scales.csv", index=False)
    ref.to_csv(args.outdir / "yearly_joined_reference.csv", index=False)
    summary.to_csv(args.outdir / "correlations_by_sst_scale.csv", index=False)
    triad.to_csv(args.outdir / "three_way_correlations_reference.csv", index=False)
    plot_scatter(ref, args)
    plot_scale(summary, args)

    print("\n[RESULT] Reference SST/organization correlations (unadjusted; observational):")
    for variable in ("Agrad_mean", "Hx_mean", "high_Agrad_fraction", "low_Agrad_fraction"):
        row = summary[(summary.sst_predictor == "sst_level_K")
                      & (summary.organization_measure == variable)
                      & np.isclose(summary.sst_half_width_deg, args.sst_half_width)].iloc[0]
        print(f"  T_level–{variable:<24s} n={row['n']:2.0f}, "
              f"r={row['r']:+.3f}, p={row['p']:.3g}, "
              f"detrended_r={row['detrended_r']:+.3f}, "
              f"partial_r|PC={row['pc_adjusted_partial_r']:+.3f}")
    print(f"[INFO] SST width samples: {scales.tolist()}")
    print(f"[INFO] Outputs: {args.outdir}")
    print("[NOTE] Values are correlations, not causal mediation estimates; "
          "PC-adjusted correlations may remove an actual mediated SST–organization pathway.")


if __name__ == "__main__":
    main()
