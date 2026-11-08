#!/usr/bin/env python3
"""Plot Fig. 4 and Figs. S4-S7 from annual and daily organization data.

Fig. 4 shows daily AI_grad distributions in low/high SON shape-PC years and
SON-mean AI_grad versus the standardized shape PC and SST level. Fig. S4 shows
high/low event frequencies; S5 compares AI_grad, AI_var, and H_x; S6 varies the
organization half-width; S7 varies the SST half-width for level and asymmetry.

Stage 4 inputs are pc_organization_daily_samples.csv,
pc_organization_group_years.csv, pc_event_frequency_group_comparison.csv,
organization_statistics.csv, and organization_event_frequency.csv. Stage 5
provides yearly_joined_reference.csv and yearly_joined_all_sst_scales.csv for
46 SON years (1979-2024).

The reference SST half-width is 12.5 degrees, selected from the SST-PC
relationship in Fig. 2b. The reference organization half-width is 5 degrees.
Neither is optimized against SST-organization correlations.

Example:
    python -B code/plot_atlantic_itcz_fig4_support_reorganized_v3_1.py \
        --stage4-dir /path/to/stage4 --analysis-dir /path/to/stage5 \
        --outdir /path/to/new/figures
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

SCRIPT_VERSION = "3.1.0"
plt.rcParams["font.size"] = 11

DEFAULT_TARGET = "seasonal_SON_raw_selected_shape_eof1"
DEFAULT_WINDOW = "rel_5deg"
EXPECTED_METRICS = ("Agrad", "AI_PW", "Hx")
MEAN_VARS = ("Agrad_mean", "AI_PW_mean", "Hx_mean")
HIGH_VARS = ("high_Agrad_fraction", "high_AI_PW_fraction", "high_Hx_fraction")
LOW_VARS = ("low_Agrad_fraction", "low_AI_PW_fraction", "low_Hx_fraction")
METRIC_LABELS = (
    r"$\mathrm{AI}_{\mathrm{grad}}$",
    r"$\mathrm{AI}_{\mathrm{var}}$",
    r"$H_x$",
)
STAT_CATEGORIES = (
    ("mean", "Seasonal mean", MEAN_VARS),
    ("high", "High-event frequency", HIGH_VARS),
    ("low", "Low-event frequency", LOW_VARS),
)
EVENT_HIGH_CANDIDATES = ("high_Agrad_fraction", "high_AI_grad_fraction")
EVENT_LOW_CANDIDATES = ("low_Agrad_fraction", "low_AI_grad_fraction")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage4-dir", type=Path, default=Path("out_atlantic_itcz/04_daily_organization_width_sensitivity_events"))
    p.add_argument("--analysis-dir", type=Path, default=Path("out_atlantic_itcz/05_sst_organization"))
    p.add_argument("--outdir", type=Path, default=Path("out_atlantic_itcz/figures"))
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--window", default=DEFAULT_WINDOW)
    p.add_argument("--sst-half-width", type=float, default=12.5)
    p.add_argument("--high-years", type=int, default=5)
    p.add_argument("--low-years", type=int, default=5)
    p.add_argument("--hist-bins", type=int, default=18)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--output-prefix", default="", help="Optional prefix for output file names")
    p.add_argument("--s7-only", action="store_true",
                   help="Plot only Fig. S7: requires Stage-5 CSVs, not Stage-4 inputs")
    return p.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required input missing: {path}")
    return pd.read_csv(path)


def require_columns(frame: pd.DataFrame, required, source: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing columns {missing}; available: {list(frame.columns)}")


def unique_keys(frame: pd.DataFrame, keys: list[str], source: str) -> None:
    bad = frame.duplicated(keys, keep=False)
    if bad.any():
        raise ValueError(f"{source} has duplicate keys {keys}; example: {frame.loc[bad, keys].head(5).to_dict('records')}")


def pearson_values(x, y) -> tuple[float, int]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        raise ValueError(f"Need at least 3 finite pairs, only {int(ok.sum())}")
    if np.ptp(x[ok]) <= 1e-12 or np.ptp(y[ok]) <= 1e-12:
        raise ValueError("Predictor or outcome constant over matched years")
    return float(pearsonr(x[ok], y[ok]).statistic), int(ok.sum())


def detrended_correlation(year, x, y) -> float:
    yr, x, y = np.asarray(year, float), np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(yr) & np.isfinite(x) & np.isfinite(y)
    yr, x, y = yr[ok], x[ok], y[ok]
    if len(x) < 4:
        raise ValueError("Insufficient years to detrend")
    t = yr - yr.mean()
    if np.dot(t, t) <= 0:
        raise ValueError("Cannot detrend constant year")
    xd = x - (np.dot(t, x - x.mean()) / np.dot(t, t)) * t
    yd = y - (np.dot(t, y - y.mean()) / np.dot(t, t)) * t
    return pearson_values(xd, yd)[0]


def load_reference(args) -> pd.DataFrame:
    f = args.analysis_dir / "yearly_joined_reference.csv"
    d = read_csv(f)
    required = ("year", "scale_deg", "shape_pc_sd", "sst_level_K", *MEAN_VARS, *HIGH_VARS, *LOW_VARS)
    require_columns(d, required, str(f))
    d = d.copy()
    for col in required:
        d[col] = pd.to_numeric(d[col], errors="raise")
    d["year"] = d["year"].astype(int)
    unique_keys(d, ["year"], str(f))
    widths = d["scale_deg"].unique()
    if len(widths) != 1 or not np.isclose(widths[0], args.sst_half_width, atol=1e-8, rtol=0):
        raise ValueError(f"Expected d={args.sst_half_width:g} deg, but {f} has d={widths.tolist()}")
    d = d.sort_values("year").reset_index(drop=True)
    if len(d) != 46 or d.year.min() != 1979 or d.year.max() != 2024:
        raise ValueError(f"Manuscript sample mismatch: expected 46 years (1979--2024); "
                         f"got {len(d)} rows ({d.year.min()}--{d.year.max()}).")
    if d[list(required)].isna().any().any():
        raise ValueError("Missing reference values: inspect yearly_joined_reference.csv")
    for col in (*HIGH_VARS, *LOW_VARS):
        vals = d[col].to_numpy(float)
        if not (np.nanmin(vals) >= -1e-8 and np.nanmax(vals) <= 1 + 1e-8):
            raise ValueError(f"{col} expected fractions 0..1, not percentages")
    print(f"[INFO] Reference sample: {len(d)} SON years; T_level SST d={args.sst_half_width:g} deg")
    return d


# ---------------------------------------------------------------------------
# Daily AI_grad histograms for low/high shape-PC years.
# ---------------------------------------------------------------------------
def choose_years(frame: pd.DataFrame, target: str, group: str, n_keep: int):
    out = frame[(frame["target"] == target) & (frame["group"] == group)].sort_values("rank").head(n_keep)
    if len(out) < n_keep:
        raise ValueError(f"Only {len(out)} {group} years found for target={target}")
    return out["year"].astype(int).tolist()


def find_event_row(sub: pd.DataFrame, candidates):
    for name in candidates:
        row = sub[sub["event_measure"] == name]
        if not row.empty:
            return row.iloc[0], name
    raise ValueError(f"Could not find event measure among {list(candidates)}")


def get_event_stats(group_cmp: pd.DataFrame, target: str, window: str):
    sub = group_cmp[(group_cmp["target"] == target) & (group_cmp["window"] == window)].copy()
    hirow, hiname = find_event_row(sub, EVENT_HIGH_CANDIDATES)
    lorow, loname = find_event_row(sub, EVENT_LOW_CANDIDATES)
    return {
        "high_event_name": hiname,
        "low_event_name": loname,
        "high_event_high_mean": float(hirow["high_mean"]),
        "high_event_low_mean": float(hirow["low_mean"]),
        "low_event_high_mean": float(lorow["high_mean"]),
        "low_event_low_mean": float(lorow["low_mean"]),
    }


def find_ai_grad_column(frame: pd.DataFrame) -> str:
    for col in ("AI_grad", "Agrad", "agrad", "A_grad"):
        if col in frame.columns:
            return col
    raise KeyError("Could not find AI_grad/Agrad column in Stage-4 daily samples")


def load_histogram(args, d: pd.DataFrame):
    samples = read_csv(args.stage4_dir / "pc_organization_daily_samples.csv")
    group_years = read_csv(args.stage4_dir / "pc_organization_group_years.csv")
    group_cmp = read_csv(args.stage4_dir / "pc_event_frequency_group_comparison.csv")
    require_columns(group_years, ["target", "group", "rank", "year"], "pc_organization_group_years.csv")
    require_columns(samples, ["target", "window", "analysis_year"], "pc_organization_daily_samples.csv")
    require_columns(group_cmp, ["target", "window", "event_measure", "high_mean", "low_mean"],
                    "pc_event_frequency_group_comparison.csv")
    high_years = choose_years(group_years, args.target, "high", args.high_years)
    low_years = choose_years(group_years, args.target, "low", args.low_years)
    if set(high_years) & set(low_years):
        raise ValueError("Selected high-PC and low-PC groups overlap")
    if (set(high_years) | set(low_years)) - set(d.year):
        raise ValueError("Histogram group years are absent from reference yearly data")
    # Check that the Stage 4 groups match the Stage 5 PC rankings.
    expected_high = set(d.nlargest(args.high_years, "shape_pc_sd").year)
    expected_low = set(d.nsmallest(args.low_years, "shape_pc_sd").year)
    if set(high_years) != expected_high or set(low_years) != expected_low:
        raise ValueError(
            "Histogram high/low shape-PC years disagree with the Stage-5 "
            "yearly_joined_reference.csv shape-PC ranks. Inspect revisions "
            "of Stage 2, Stage 4, and Stage 5 before plotting."
        )
    sub = samples[(samples["target"] == args.target) & (samples["window"] == args.window)].copy()
    if sub.empty:
        available = sorted(samples["window"].astype(str).unique().tolist())
        raise ValueError(f"No daily samples for target={args.target}, window={args.window}. Available: {available}")
    ai_col = find_ai_grad_column(sub)
    sub = sub.rename(columns={ai_col: "AI_grad"})
    sub["pc_group"] = np.where(
        sub["analysis_year"].isin(high_years), "high",
        np.where(sub["analysis_year"].isin(low_years), "low", "other")
    )
    sub = sub[sub["pc_group"].isin(["high", "low"])].copy()
    high = sub.loc[sub["pc_group"] == "high", "AI_grad"].to_numpy(float)
    low = sub.loc[sub["pc_group"] == "low", "AI_grad"].to_numpy(float)
    if not np.isfinite(np.r_[high, low]).any():
        raise ValueError("No finite daily AI_grad values")
    stats = get_event_stats(group_cmp, args.target, args.window)
    allvals = np.r_[high, low]
    lo, hi = np.nanmin(allvals), np.nanmax(allvals)
    pad = 0.03 * (hi - lo) if hi > lo else 0.1
    edges = np.linspace(lo - pad, hi + pad, args.hist_bins + 1)
    # The original grouped-event summaries must also match the reference data.
    for key, col, subset in (
        ("high_event_high_mean", "high_Agrad_fraction", high_years),
        ("high_event_low_mean", "high_Agrad_fraction", low_years),
        ("low_event_high_mean", "low_Agrad_fraction", high_years),
        ("low_event_low_mean", "low_Agrad_fraction", low_years),
    ):
        expected = float(d.loc[d.year.isin(subset), col].mean())
        if not np.isclose(stats[key], expected, rtol=1e-5, atol=1e-6):
            raise ValueError(f"Stage-4 group-event statistic {key}={stats[key]:.6g} "
                             f"does not match reference-yearly mean {expected:.6g}; "
                             "check Stage-4 and Stage-5 revisions")
    print(f"[INFO] Low/high shape-PC years: {low_years} / {high_years}")
    print("[INFO] Fig4a group event percentages: "
          f"high-PC(high/low)={100*stats['high_event_high_mean']:.1f}/{100*stats['low_event_high_mean']:.1f}; "
          f"low-PC(high/low)={100*stats['high_event_low_mean']:.1f}/{100*stats['low_event_low_mean']:.1f}")
    return low, high, edges, stats


def draw_original_histogram(ax, low, high, edges, stats):
    # Use the same histogram bins and density normalization for both groups.
    ax.hist(low, bins=edges, density=True, histtype="stepfilled",
            alpha=0.30, color="tab:blue", label="Low shape-PC years")
    ax.hist(high, bins=edges, density=True, histtype="stepfilled",
            alpha=0.30, color="tab:orange", label="High shape-PC years")
    ax.hist(low, bins=edges, density=True, histtype="step",
            linewidth=1.8, color="tab:blue")
    ax.hist(high, bins=edges, density=True, histtype="step",
            linewidth=1.8, color="tab:orange")
    ax.axvline(np.nanmedian(low), color="tab:blue", linestyle="--", linewidth=1.4)
    ax.axvline(np.nanmedian(high), color="tab:orange", linestyle="--", linewidth=1.4)
    ax.set_xlabel(r"Daily $\mathrm{AI}_{\mathrm{grad}}$")
    ax.set_ylabel("Probability density")
    ax.set_title(r"Daily $\mathrm{AI}_{\mathrm{grad}}$ distribution")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(frameon=True, framealpha=0.9, loc="center right")
    txt = "\n".join([
        rf"High-PC years: high-$\mathrm{{AI}}_{{\mathrm{{grad}}}}$ = "
        rf"{100*stats['high_event_high_mean']:.1f}%, low-$\mathrm{{AI}}_{{\mathrm{{grad}}}}$ = "
        rf"{100*stats['low_event_high_mean']:.1f}%",
        rf"Low-PC years: high-$\mathrm{{AI}}_{{\mathrm{{grad}}}}$ = "
        rf"{100*stats['high_event_low_mean']:.1f}%, low-$\mathrm{{AI}}_{{\mathrm{{grad}}}}$ = "
        rf"{100*stats['low_event_low_mean']:.1f}%",
    ])
    ax.text(
        0.98, 0.90, txt, transform=ax.transAxes,
        ha="right", va="top", fontsize=10,
        bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.92,
                  boxstyle="round,pad=0.4"),
    )


def draw_scatter(ax, year, x, y, xlabel: str, ylabel: str,
                 show_detrended: bool = True, annotation_side: str = "left") -> tuple[float, float]:
    year = np.asarray(year, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    valid = np.isfinite(year) & np.isfinite(x) & np.isfinite(y)
    xx, yy = x[valid], y[valid]
    r, n = pearson_values(xx, yy)
    rd = detrended_correlation(year[valid], xx, yy)
    ax.scatter(xx, yy, s=23, facecolors="0.55", edgecolors="none", alpha=0.8)
    fit = np.polyfit(xx, yy, 1)
    gx = np.linspace(np.min(xx), np.max(xx), 100)
    ax.plot(gx, np.polyval(fit, gx), lw=1.3, color="tab:blue")
    label = rf"$r={r:+.2f}$"
    if show_detrended:
        label += "\n" + rf"detrended $r={rd:+.2f}$"
    xloc = 0.96 if annotation_side == "right" else 0.04
    ax.text(xloc, 0.95, label, transform=ax.transAxes,
            ha="right" if annotation_side == "right" else "left", va="top", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.20, linewidth=0.6)
    return r, rd


def save_figure(fig, outdir: Path, stem: str, dpi: int):
    pdf, png = outdir / f"{stem}.pdf", outdir / f"{stem}.png"
    fig.savefig(pdf)  # Keep the specified physical size; do not trim the bounding box.
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    print(f"[INFO] Wrote {pdf} and {png}")


def plot_fig4(d, low, high, edges, stats, args):
    # Histogram-axis dimensions in inches:
    # x=0.12*7.2; width=0.86*7.2; bottom=0.15*4.6; height=0.73*4.6.
    page_width, original_height, page_height = 7.2, 4.6, 8.55
    origin = page_height - original_height
    fig = plt.figure(figsize=(page_width, page_height))
    old_fontsize = plt.rcParams["font.size"]
    plt.rcParams["font.size"] = 12  # Use 12-point fonts for Fig. 4.
    ax_a = fig.add_axes([0.12,
                         (origin + 0.15*original_height)/page_height,
                         0.86,
                         (0.88-0.15)*original_height/page_height])
    draw_original_histogram(ax_a, low, high, edges, stats)
    plt.rcParams["font.size"] = old_fontsize

    ax_b = fig.add_axes([0.12, 0.090, 0.355, 0.315])
    ax_c = fig.add_axes([0.600, 0.090, 0.355, 0.315])
    xlabel_pc = "Standardized SON shape PC"
    xlabel_sst = rf"SON $T_{{\rm level}}$ / K ($d={args.sst_half_width:g}^\circ$)"
    ylabel = r"SON-mean $\mathrm{AI}_{\mathrm{grad}}$"
    r_pc, rd_pc = draw_scatter(ax_b, d["year"], d["shape_pc_sd"], d["Agrad_mean"], xlabel_pc, ylabel)
    r_sst, rd_sst = draw_scatter(ax_c, d["year"], d["sst_level_K"], d["Agrad_mean"], xlabel_sst, ylabel)
    ax_b.set_title("Broadening and zonal heterogeneity", fontsize=10.5)
    ax_c.set_title("SST level and zonal heterogeneity", fontsize=10.5)
    for ax in (ax_b, ax_c):
        ax.tick_params(labelsize=10)

    fig.text(0.035, 0.978, "(a)", ha="left", va="top", fontsize=12)
    fig.text(0.035, 0.427, "(b)", ha="left", va="top", fontsize=12)
    fig.text(0.513, 0.427, "(c)", ha="left", va="top", fontsize=12)
    save_figure(fig, args.outdir, f"{args.output_prefix}fig4_ai_grad_shape_pc_sst_level_v3", args.dpi)
    print(f"[CHECK] Fig4b shape-PC--AI_grad r={r_pc:+.4f}, detrended={rd_pc:+.4f}")
    print(f"[CHECK] Fig4c SST-level--AI_grad r={r_sst:+.4f}, detrended={rd_sst:+.4f}")
    return (r_pc, r_sst)


def plot_s4(d, args):
    specs = [("shape_pc_sd", "Standardized SON shape PC"),
             ("sst_level_K", rf"SON $T_{{\rm level}}$ / K ($d={args.sst_half_width:g}^\circ$)")]
    targets = [("high_Agrad_fraction", "High-event frequency"),
               ("low_Agrad_fraction", "Low-event frequency")]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.3))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.11, top=0.935, wspace=0.30, hspace=0.44)
    for i, (predictor, xlabel) in enumerate(specs):
        for j, (event, title) in enumerate(targets):
            ax = axes[i, j]
            panel = chr(97 + 2*i + j)
            r, _ = draw_scatter(ax, d["year"], d[predictor], 100 * d[event],
                                xlabel,
                                (r"High-$\mathrm{AI}_{\mathrm{grad}}$ days / %" if j == 0
                                 else r"Low-$\mathrm{AI}_{\mathrm{grad}}$ days / %"),
                                show_detrended=False, annotation_side="right" if j == 1 else "left")
            ax.set_title(f"({panel}) {title}", fontsize=11)
            print(f"[CHECK] FigS4{panel} {predictor} vs {event}: r={r:+.4f}")
    save_figure(fig, args.outdir, f"{args.output_prefix}figS4_organization_event_frequency_v3", args.dpi)


def plot_s5(d, args):
    predictors = (("shape_pc_sd", "SON shape PC"), ("sst_level_K", r"SON $T_{\rm level}$"))
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.095, top=0.94, wspace=0.20, hspace=0.35)
    summaries = []
    xpos = np.arange(3)
    for i, (predictor, pred_label) in enumerate(predictors):
        for j, (cat, title, cols) in enumerate(STAT_CATEGORIES):
            ax = axes[i, j]
            vals = []
            for metric, response in zip(EXPECTED_METRICS, cols):
                r, n = pearson_values(d[predictor], d[response])
                vals.append(r)
                summaries.append({"predictor": predictor, "statistic": cat, "metric": metric,
                                  "response": response, "r": r, "n": n})
            # One bar per organization metric.
            ax.bar(xpos, vals, width=0.62, color="tab:blue")
            ax.axhline(0, lw=0.8, color="0.4")
            ax.set_xticks(xpos)
            ax.set_xticklabels(METRIC_LABELS)
            ax.set_ylim(-1, 1)
            ax.set_title(f"({chr(97+i*3+j)}) {title}", fontsize=11)
            ax.grid(True, axis="y", alpha=0.23, lw=0.6)
            for xi, val in zip(xpos, vals):
                offset = 0.035 if val >= 0 else -0.035
                ax.text(xi, val + offset, f"{val:+.2f}", ha="center",
                        va="bottom" if val >= 0 else "top", fontsize=9)
        axes[i, 0].set_ylabel(f"Pearson $r$ with {pred_label}", fontsize=10)
    out = args.outdir / f"{args.output_prefix}figS5_organization_metric_robustness_v3"
    pd.DataFrame(summaries).to_csv(out.with_name(out.name + "_data.csv"), index=False)
    save_figure(fig, args.outdir, out.name, args.dpi)
    return pd.DataFrame(summaries)


def window_half_width(window: str) -> float:
    match = re.fullmatch(r"rel_(\d+)(?:p(\d+))?deg", str(window))
    if not match:
        return np.nan
    return float(match.group(1)) + (float("0." + match.group(2)) if match.group(2) else 0.0)


def load_stage4_width_data(args, d):
    sf = args.stage4_dir / "organization_statistics.csv"
    ef = args.stage4_dir / "organization_event_frequency.csv"
    stats, events = read_csv(sf), read_csv(ef)
    keys = ["period_type", "period", "year", "window"]
    require_columns(stats, [*keys, *MEAN_VARS], str(sf))
    require_columns(events, [*keys, *HIGH_VARS, *LOW_VARS], str(ef))
    frames = []
    for frame, source, columns in ((stats, str(sf), MEAN_VARS),
                                   (events, str(ef), (*HIGH_VARS, *LOW_VARS))):
        selected = frame.loc[(frame["period_type"].astype(str) == "seasonal")
                             & (frame["period"].astype(str) == "SON")].copy()
        selected["half_width_deg"] = selected["window"].map(window_half_width)
        selected = selected[np.isfinite(selected["half_width_deg"])].copy()
        if selected.empty:
            raise ValueError(f"No SON relative-degree windows in {source}")
        selected["year"] = pd.to_numeric(selected["year"], errors="raise").astype(int)
        unique_keys(selected, ["year", "window"], source)
        frames.append(selected[["year", "window", "half_width_deg", *columns]])

    stats, events = frames
    widths1, widths2 = set(stats["window"]), set(events["window"])
    if widths1 != widths2:
        raise ValueError("Stage-4 statistics/event-frequency windows do not match: "
                         f"statistics-only={sorted(widths1-widths2)}, event-only={sorted(widths2-widths1)}. "
                         "Use 04_daily_organization_width_sensitivity_events output.")
    combined = stats.merge(events, on=["year", "window", "half_width_deg"], validate="one_to_one")
    dsmall = d[["year", "shape_pc_sd", "sst_level_K"]].copy()
    combined = combined.merge(dsmall, on="year", validate="many_to_one")
    widths = sorted(combined["half_width_deg"].unique().tolist())
    if not 5.0 in widths:
        raise ValueError("Stage-4 width series lacks ±5° reference")
    if len(widths) < 3:
        raise ValueError(f"Need >=3 organization half-widths, got {widths}")
    for width, group in combined.groupby("half_width_deg"):
        unique_keys(group, ["year"], f"width={width}")
        if set(group.year) != set(d.year):
            missing = sorted(set(d.year) - set(group.year))
            raise ValueError(f"Missing years at organization half-width={width}: {missing[:10]}")
    for event_col in (*HIGH_VARS, *LOW_VARS):
        q = combined[event_col].to_numpy(float)
        if np.nanmin(q) < -1e-8 or np.nanmax(q) > 1+1e-8:
            raise ValueError(f"Stage-4 {event_col} must be fractions, not percentages")

    # Require the Stage 4 and Stage 5 statistics to share the reference window.
    reference = combined[np.isclose(combined.half_width_deg, window_half_width(args.window))].copy()
    reference = reference.sort_values("year").reset_index(drop=True)
    df = d.sort_values("year").reset_index(drop=True)
    for col in (*MEAN_VARS, *HIGH_VARS, *LOW_VARS):
        if not np.allclose(reference[col], df[col], rtol=1e-6, atol=1e-7, equal_nan=True):
            maxdiff = np.max(np.abs(reference[col].to_numpy(float)-df[col].to_numpy(float)))
            raise ValueError(f"Stage 4 vs Stage 5 reference mismatch in {col}: max diff={maxdiff:g}. "
                             "Regenerate Stage 5 from the same Stage 4 directory before plotting.")
    print(f"[INFO] Organization-window half-widths: {widths}; {len(d)} matched years each")
    return combined, widths


def plot_s6(wide, widths, s5, args):
    predictors = (("shape_pc_sd", "SON shape PC"), ("sst_level_K", r"SON $T_{\rm level}$"))
    records = []
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 7.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.105, top=0.94, wspace=0.19, hspace=0.31)
    # Fixed colors across S5 and S6 are unnecessary; use the three distinct
    # Matplotlib default colors consistently across the six panels.
    colors = ("tab:blue", "tab:orange", "tab:green")
    for i, (predictor, pred_label) in enumerate(predictors):
        for j, (cat, title, cols) in enumerate(STAT_CATEGORIES):
            ax = axes[i, j]
            for k, (metric, response) in enumerate(zip(EXPECTED_METRICS, cols)):
                rr = []
                for width in widths:
                    selected = wide[np.isclose(wide["half_width_deg"], width)].sort_values("year")
                    r, n = pearson_values(selected[predictor], selected[response])
                    rr.append(r)
                    records.append({"predictor": predictor, "statistic": cat, "metric": metric,
                                    "response": response, "organization_half_width_deg": width,
                                    "r": r, "n": n})
                ax.plot(widths, rr, marker="o", ms=3.5, lw=1.4,
                        color=colors[k], label=METRIC_LABELS[k])
            ax.axhline(0, lw=0.8, color="0.4")
            ax.axvline(window_half_width(args.window), lw=0.9, ls="--", color="0.5")
            ax.set_ylim(-1, 1)
            ax.set_title(f"({chr(97+i*3+j)}) {title}", fontsize=11)
            ax.grid(True, alpha=0.23, lw=0.6)
            if i == 1:
                ax.set_xlabel("Organization half-width / deg")
        axes[i, 0].set_ylabel(f"Pearson $r$ with {pred_label}", fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="lower right")
    stem = f"{args.output_prefix}figS6_organization_window_sensitivity_v3"
    series = pd.DataFrame(records)
    series.to_csv(args.outdir / f"{stem}_data.csv", index=False)
    save_figure(fig, args.outdir, stem, args.dpi)

    # The half-width=5° points in S6 must numerically reproduce the bars in S5.
    matched = series[np.isclose(series.organization_half_width_deg, window_half_width(args.window))]
    check = matched.merge(s5, on=["predictor", "statistic", "metric", "response"],
                          suffixes=("_s6", "_s5"), validate="one_to_one")
    if len(check) != 18 or not np.allclose(check["r_s6"], check["r_s5"], atol=1e-10, rtol=0):
        raise AssertionError("Reference-window S6 correlations do not match S5 bar values")
    print("[PASS] S5 bar correlations == S6 correlations at 5° (all 18 entries)")
    return series


def load_sst_width_data(args, reference: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    """Read one row per SST half-width and year with fixed organization responses.

    Require complete annual samples and agreement with the reference data used by
    Fig. 4 and Fig. S5.
    """
    f = args.analysis_dir / "yearly_joined_all_sst_scales.csv"
    data = read_csv(f)
    need = ["year", "scale_deg", "pc", "sst_level_K", "sst_asym_K",
            *MEAN_VARS, *HIGH_VARS, *LOW_VARS]
    require_columns(data, need, str(f))
    data = data.copy()
    for col in need:
        data[col] = pd.to_numeric(data[col], errors="raise")
    data["year"] = data["year"].astype(int)
    unique_keys(data, ["year", "scale_deg"], str(f))
    widths = sorted(float(x) for x in data["scale_deg"].unique())
    if len(widths) < 3:
        raise ValueError(f"Expected SST half-width sweep, found only {widths}")
    if not any(np.isclose(widths, args.sst_half_width, rtol=0, atol=1e-8)):
        raise ValueError(f"SST reference d={args.sst_half_width:g} is missing from sweep")
    ref = reference.set_index("year").sort_index()
    for width, sub in data.groupby("scale_deg"):
        sub = sub.set_index("year").sort_index()
        if set(sub.index) != set(ref.index):
            raise ValueError(f"Incomplete sample at SST half-width {width:g}: "
                             f"missing {sorted(set(ref.index)-set(sub.index))}")
        # Stage 5 is expected to repeat these season-year organization values
        # over the different choices of SST averaging width.
        for col in (*MEAN_VARS, *HIGH_VARS, *LOW_VARS):
            if not np.allclose(sub[col], ref[col], rtol=1e-9, atol=1e-10, equal_nan=True):
                raise ValueError(f"Organization response {col} differs from reference "
                                 f"at SST half-width d={width:g}; do not use these revisions together")
        if "shape_pc_raw" in ref.columns and not np.allclose(
            sub["pc"], ref["shape_pc_raw"], rtol=1e-9, atol=1e-10, equal_nan=True
        ):
            raise ValueError(f"Shape PC differs between yearly data and SST-width sweep at d={width:g}")
        if np.isclose(width, args.sst_half_width, atol=1e-8, rtol=0):
            for key in ("sst_level_K", "sst_asym_K"):
                if not np.allclose(sub[key], ref[key], rtol=1e-10, atol=1e-10):
                    raise ValueError(f"Reference SST {key} differs from d={width:g} SST sweep")
    if data[need].isna().any().any():
        raise ValueError("Incomplete SST-width analysis values")
    print(f"[INFO] SST half-widths (Fig.S7): {widths} ({len(ref)} years per width)")
    return data, widths


def plot_s7(data, widths, reference, args):
    """Fig.S7: two SST predictors x three response statistics x three metrics."""
    predictors = (("sst_level_K", r"SON $T_{\rm level}$"),
                  ("sst_asym_K", r"SON $T_{\rm asym}$"))
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 7.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.105, top=0.94,
                        wspace=0.19, hspace=0.31)
    colors = ("tab:blue", "tab:orange", "tab:green")
    records = []
    for i, (predictor, label) in enumerate(predictors):
        for j, (cat, title, cols) in enumerate(STAT_CATEGORIES):
            ax = axes[i, j]
            for k, (metric, response) in enumerate(zip(EXPECTED_METRICS, cols)):
                rs = []
                for width in widths:
                    sub = data[np.isclose(data["scale_deg"], width, atol=1e-8, rtol=0)].sort_values("year")
                    r, n = pearson_values(sub[predictor], sub[response])
                    rs.append(r)
                    records.append({"sst_predictor": predictor,
                                    "sst_half_width_deg": width,
                                    "statistic": cat, "metric": metric,
                                    "organization_measure": response,
                                    "r": r, "n": n,
                                    "organization_window": args.window})
                ax.plot(widths, rs, marker="o", ms=3.5, lw=1.4,
                        color=colors[k], label=METRIC_LABELS[k])
            ax.axhline(0, lw=0.8, color="0.4")
            ax.axvline(args.sst_half_width, lw=0.9, ls="--", color="0.5")
            ax.set_ylim(-1, 1)
            ax.set_title(f"({chr(97+i*3+j)}) {title}", fontsize=11)
            ax.grid(True, alpha=0.23, lw=0.6)
            if i == 1:
                ax.set_xlabel(r"SST half-width, $d$ / deg")
        axes[i, 0].set_ylabel(f"Pearson $r$ with {label}", fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="lower right")
    stem = f"{args.output_prefix}figS7_sst_width_sensitivity_v3_1"
    out = pd.DataFrame(records)
    out.to_csv(args.outdir / f"{stem}_data.csv", index=False)
    save_figure(fig, args.outdir, stem, args.dpi)

    # At d=12.5, the nine T_level correlations must match the lower row of S5.
    match = out[(out["sst_predictor"] == "sst_level_K") &
                np.isclose(out["sst_half_width_deg"], args.sst_half_width, atol=1e-8, rtol=0)]
    for _, row in match.iterrows():
        v = pearson_values(reference["sst_level_K"], reference[row["organization_measure"]])[0]
        if not np.isclose(row["r"], v, atol=1e-12, rtol=0):
            raise AssertionError(f"Fig.S7 d={args.sst_half_width:g} != reference for {row['organization_measure']}")
    if len(match) != 9:
        raise AssertionError(f"Expected 9 SST-level reference correlations, got {len(match)}")
    print("[PASS] Fig.S7 SST d=12.5 correlations == Fig.S5 SST-level bars (all 9 entries)")

    # Optional cross-check against independent Stage-5 precomputed correlations.
    check_file = args.analysis_dir / "correlations_by_sst_scale.csv"
    if check_file.is_file():
        q = read_csv(check_file)
        require_columns(q, ["sst_half_width_deg", "sst_predictor", "organization_measure",
                            "organization_window", "r", "n"], str(check_file))
        q = q[q["organization_window"].astype(str) == str(args.window)]
        joined = out.merge(q, on=["sst_half_width_deg", "sst_predictor", "organization_measure", "organization_window"],
                           validate="one_to_one", suffixes=("_s7", "_precomputed"))
        if len(joined) != len(out) or not np.allclose(joined["r_s7"], joined["r_precomputed"], atol=1e-10, rtol=0):
            raise ValueError("Fig.S7 values disagree with correlations_by_sst_scale.csv")
        if not (joined["n_s7"] == joined["n_precomputed"]).all():
            raise ValueError("Fig.S7 sample sizes disagree with precomputed correlations")
        print("[PASS] Fig.S7 all 180 correlations match independent Stage-5 correlations CSV")
    return out


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    d = load_reference(args)
    # SST-width sensitivity uses only Stage-5 yearly CSVs and can run alone.
    sst_wide, sst_widths = load_sst_width_data(args, d)
    if args.s7_only:
        plot_s7(sst_wide, sst_widths, d, args)
        print("[DONE] Fig. S7 generated")
        return
    low, high, edges, stats = load_histogram(args, d)
    # Check all inputs before writing figures.
    wide, widths = load_stage4_width_data(args, d)
    plot_fig4(d, low, high, edges, stats, args)
    plot_s4(d, args)
    s5 = plot_s5(d, args)
    plot_s6(wide, widths, s5, args)
    plot_s7(sst_wide, sst_widths, d, args)
    print("[DONE] Reorganized Fig. 4 and Figs. S4--S7 generated without PDF input")


if __name__ == "__main__":
    main()
