#!/usr/bin/env python3
"""Export Table S1 from raw seasonal Stage 2 diagnostic correlations.

The exporter uses only the Python standard library.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SEASONS = ("DJF", "MAM", "JJA", "SON")
FRACTIONAL = (0.7, 0.8, 0.9)
ABSOLUTE = (45, 48, 50)
MARGIN = 0.15
ATOL = 1e-12
RTOL = 1e-10
CENTER = "center_deg"
INTEGRAL = "integrated_tcwv_kgm2_deg"
FRAC_KEYS = {value: "width_frac_" + str(value).replace(".", "p") + "_deg"
             for value in FRACTIONAL}
ABS_KEYS = {value: "width_abs_{}kgm2_deg".format(value) for value in ABSOLUTE}
DIAGNOSTICS = (CENTER, INTEGRAL, *FRAC_KEYS.values(), *ABS_KEYS.values())
SELECTORS = ("period_type", "period", "series_type", "mode_family", "mode")
REQUIRED = (*SELECTORS, "n_years",
            *("corr_" + key for key in DIAGNOSTICS),
            *("n_" + key for key in DIAGNOSTICS))
TABLE_FIELDS = ("season", "mode", "series_type", "shift_score", "shape_score",
                "score_difference", "classification", "n_years")
SENSITIVITY_FIELDS = (*TABLE_FIELDS, "fractional_threshold",
                      "absolute_threshold_kg_m2", "classification_margin",
                      "n_center", "n_integrated", "n_fractional", "n_absolute")


class ValidationError(ValueError):
    """Invalid exporter input or output destination."""


def finite_number(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValidationError("{} must be a finite number".format(label)) from None
    if not math.isfinite(result):
        raise ValidationError("{} must be a finite number".format(label))
    return result


def integer(value, label, minimum=1):
    result = finite_number(value, label)
    if not result.is_integer() or result < minimum:
        raise ValidationError("{} must be an integer >= {}".format(label, minimum))
    return int(result)


def compute_scores(center, integrated, fractional, absolute, margin=MARGIN):
    """Use absolute correlations before averaging the three shape diagnostics."""
    values = [finite_number(v, "correlation")
              for v in (center, integrated, fractional, absolute)]
    if any(abs(v) > 1 for v in values):
        raise ValidationError("correlations must lie in [-1, 1]")
    margin = finite_number(margin, "margin")
    if margin < 0:
        raise ValidationError("margin must be nonnegative")
    shift = abs(values[0])
    shape = math.fsum(abs(v) for v in values[1:]) / 3
    # Match classify_and_orient's inclusive comparison, including at boundaries.
    if shift >= shape + margin:
        classification = "shift"
    elif shape >= shift + margin:
        classification = "shape"
    else:
        classification = "mixed"
    return dict(shift_score=shift, shape_score=shape,
                score_difference=abs(shift - shape), classification=classification)


def load_summary(input_path):
    """Validate and select exactly eight raw seasonal total_abs modes."""
    path = Path(input_path).resolve(strict=True)
    content = path.read_bytes()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
    headers = reader.fieldnames or []
    if len(headers) != len(set(headers)):
        raise ValidationError("duplicate CSV column names")
    missing = set(REQUIRED) - set(headers)
    if missing:
        raise ValidationError("missing columns: " + ", ".join(sorted(missing)))
    selected = {}
    seen = set()
    row_count = 0
    checked_scores = 0
    for line, row in enumerate(reader, 2):
        row_count += 1
        if None in row or any(value is None for value in row.values()):
            raise ValidationError("line {}: wrong number of CSV fields".format(line))
        period_type = row["period_type"]
        period = row["period"]
        if period_type not in ("monthly", "seasonal"):
            raise ValidationError("line {}: invalid period_type".format(line))
        if period_type == "seasonal" and period not in SEASONS:
            raise ValidationError("line {}: invalid season".format(line))
        if period_type == "monthly" and period not in tuple(str(m) for m in range(1, 13)):
            raise ValidationError("line {}: invalid month".format(line))
        if row["series_type"] not in ("raw", "linear_detrended"):
            raise ValidationError("line {}: invalid series_type".format(line))
        if row["mode_family"] not in ("total_abs", "shape_relative"):
            raise ValidationError("line {}: invalid mode_family".format(line))
        mode = integer(row["mode"], "mode")
        if mode not in (1, 2, 3):
            raise ValidationError("line {}: unsupported EOF number".format(line))
        key = (period_type, period, row["series_type"], row["mode_family"], mode)
        if key in seen:
            raise ValidationError("duplicate selector: {!r}".format(key))
        seen.add(key)
        if (period_type != "seasonal" or row["series_type"] != "raw"
                or row["mode_family"] != "total_abs" or mode not in (1, 2)):
            continue
        n_years = integer(row["n_years"], "n_years", 3)
        correlations = {}
        counts = {}
        for diagnostic in DIAGNOSTICS:
            value = finite_number(row["corr_" + diagnostic], diagnostic)
            if abs(value) > 1:
                raise ValidationError("{}: correlation outside [-1, 1]".format(diagnostic))
            count = integer(row["n_" + diagnostic], "n_" + diagnostic, 3)
            if count > n_years:
                raise ValidationError("{}: sample size exceeds n_years".format(diagnostic))
            correlations[diagnostic] = value
            counts[diagnostic] = count
        scores = compute_scores(*(correlations[k] for k in
                                  (CENTER, INTEGRAL, FRAC_KEYS[0.8], ABS_KEYS[48])))
        for field in ("shift_score", "shape_score"):
            if field in headers:
                stored = finite_number(row[field], "stored " + field)
                if not math.isclose(scores[field], stored, abs_tol=ATOL, rel_tol=RTOL):
                    raise ValidationError("{} EOF{}: stored {} mismatch".format(period, mode, field))
                checked_scores += 1
        if "mode_type" in headers and row["mode_type"] != scores["classification"]:
            raise ValidationError("{} EOF{}: stored classification mismatch".format(period, mode))
        selected[(period, mode)] = dict(season=period, mode=mode, n_years=n_years,
                                        correlations=correlations, counts=counts)
    expected = {(season, mode) for season in SEASONS for mode in (1, 2)}
    if set(selected) != expected:
        raise ValidationError("missing raw seasonal total_abs modes: {!r}".format(
            sorted(expected - set(selected))))
    metadata = dict(path=str(path), sha256=hashlib.sha256(content).hexdigest(),
                    rows_read=row_count, rows_selected=8,
                    stored_score_values_checked=checked_scores,
                    stored_classifications_checked=8 if "mode_type" in headers else 0)
    return [selected[(season, mode)] for season in SEASONS for mode in (1, 2)], metadata


def build_tables(modes):
    table, sensitivity, margin_checks = [], [], []
    for mode in modes:
        corr, counts = mode["correlations"], mode["counts"]
        common = {key: mode[key] for key in ("season", "mode", "n_years")}
        common["series_type"] = "raw"
        args = [corr[key] for key in (CENTER, INTEGRAL, FRAC_KEYS[0.8], ABS_KEYS[48])]
        baseline = dict(common, **compute_scores(*args))
        table.append(baseline)
        endpoint_classes = [compute_scores(*args, margin=m)["classification"]
                            for m in (0.0, 0.30)]
        margin_checks.append(dict(season=mode["season"], mode=mode["mode"],
                                  classification_at_margin_0=endpoint_classes[0],
                                  classification_at_margin_0p30=endpoint_classes[1],
                                  unchanged_through_0p30=all(
                                      c == baseline["classification"] for c in endpoint_classes)))
        for fractional in FRACTIONAL:
            for absolute in ABSOLUTE:
                fkey, akey = FRAC_KEYS[fractional], ABS_KEYS[absolute]
                scores = compute_scores(corr[CENTER], corr[INTEGRAL], corr[fkey], corr[akey])
                sensitivity.append(dict(common, **scores, fractional_threshold=fractional,
                                        absolute_threshold_kg_m2=absolute,
                                        classification_margin=MARGIN,
                                        n_center=counts[CENTER], n_integrated=counts[INTEGRAL],
                                        n_fractional=counts[fkey], n_absolute=counts[akey]))
    baseline_classes = {(row["season"], row["mode"]): row["classification"] for row in table}
    checks = dict(
        baseline_width_margin_sensitivity=dict(
            fractional_threshold=0.8, absolute_threshold_kg_m2=48,
            margin_interval=[0.0, 0.30], per_mode=margin_checks,
            all_unchanged=all(row["unchanged_through_0p30"] for row in margin_checks),
            minimum_score_difference=min(row["score_difference"] for row in table)),
        width_threshold_sensitivity=dict(
            classification_margin=MARGIN, combinations=9, rows=len(sensitivity),
            all_match_baseline_classification=all(
                row["classification"] == baseline_classes[(row["season"], row["mode"])]
                for row in sensitivity),
            minimum_score_difference=min(row["score_difference"] for row in sensitivity)),
        interpretation="The margin interval applies only to baseline widths. "
                       "The nine width combinations are tested at margin 0.15 only.")
    return table, sensitivity, checks


def checked_destination(output_path):
    """Reject existing destinations and aliases into protected repository trees."""
    original = Path(output_path).expanduser()
    if os.path.lexists(original):
        raise ValidationError("output destination already exists: {}".format(original))
    destination = original.resolve()
    protected = {ROOT / name for name in
                 ("audit", "draft", "script", "src", "config", "doc", "tests",
                  "notebook", "old", ".git")}
    protected.update(p for p in ROOT.iterdir()
                     if p.name.startswith(("out_", "fig", "log", "csv")))
    protected = {p.resolve() for p in protected}
    inside_root = ROOT in destination.parents
    reserved_name = (inside_root and destination.relative_to(ROOT).parts[0].startswith(
        ("out_", "fig", "log", "csv")))
    if (destination == ROOT or reserved_name
            or any(p == destination or p in destination.parents for p in protected)):
        raise ValidationError("output destination is in a protected repository tree")
    return destination


def csv_text(rows, fields):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def markdown_text(table, checks):
    lines = ["# Table S1", "", "Raw seasonal total_abs EOFs. Baseline widths: fractional 0.8, "
             "absolute 48 kg m^-2; classification margin: 0.15.", "",
             "| Season | EOF | Shift score | Shape score | Score difference | Classification |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in table:
        lines.append("| {season} | {mode} | {shift_score:.3f} | {shape_score:.3f} | "
                     "{score_difference:.3f} | {classification} |".format(**row))
    lines.extend(["", "CSV files retain unrounded floating-point values.", "",
                  "Baseline-width classification unchanged for margins 0 to 0.30: **{}**.".format(
                      checks["baseline_width_margin_sensitivity"]["all_unchanged"]), "",
                  "Classification unchanged across nine width combinations at margin 0.15: **{}**.".format(
                      checks["width_threshold_sensitivity"]["all_match_baseline_classification"]), "",
                  checks["interpretation"], ""])
    return "\n".join(lines)


def export_table(input_path, output_path):
    destination = checked_destination(output_path)
    modes, metadata = load_summary(input_path)
    table, sensitivity, checks = build_tables(modes)
    summary = dict(schema_version=1, input=metadata,
                   exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   selection=dict(period_type="seasonal", series_type="raw",
                                  mode_family="total_abs", seasons=list(SEASONS), modes=[1, 2]),
                   settings=dict(baseline_fractional_threshold=0.8,
                                 baseline_absolute_threshold_kg_m2=48,
                                 fractional_thresholds=list(FRACTIONAL),
                                 absolute_thresholds_kg_m2=list(ABSOLUTE),
                                 classification_margin=MARGIN,
                                 stored_score_check_atol=ATOL, stored_score_check_rtol=RTOL),
                   output_rows=dict(table_s1=8, threshold_sensitivity=72), validation=checks,
                   limitations=["Saved diagnostic correlations are inputs; PCs, EOF fields and "
                                "diagnostic time series are not recomputed.",
                                "All three shape correlations must be finite; no missing-component fallback.",
                                "Golden values are used only by the separate regression tests."])
    files = {"table_s1.csv": csv_text(table, TABLE_FIELDS),
             "table_s1.md": markdown_text(table, checks),
             "threshold_sensitivity.csv": csv_text(sensitivity, SENSITIVITY_FIELDS),
             "validation_summary.json": json.dumps(summary, indent=2, sort_keys=True,
                                                    allow_nan=False) + "\n"}
    # Validate and render before creating anything. No overwrite mode is provided.
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    for name, content in files.items():
        with (destination / name).open("x", encoding="utf-8", newline="") as handle:
            handle.write(content)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Stage 2 tcwv_mode_summary.csv")
    parser.add_argument("--outdir", type=Path, required=True, help="New output directory; must not exist")
    args = parser.parse_args(argv)
    try:
        export_table(args.input, args.outdir)
    except (OSError, UnicodeError, csv.Error, ValidationError) as error:
        parser.exit(2, "error: {}\n".format(error))
    print("Created Table S1 and sensitivity outputs in {}".format(args.outdir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
