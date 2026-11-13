# Atlantic ITCZ variability

Code and processed data for studying meridional shifts, changes in width, and
daily moisture organization of the Atlantic Intertropical Convergence Zone
(ITCZ), and their relationships with sea surface temperature (SST). The
analysis uses ERA5 total column water vapor (TCWV) for 1979–2024.

## Reproduce the figures

The bundled data are sufficient to generate Figs. 1–4, S1–S7, and Table S1.
No ERA5 download or analysis rerun is needed for this step. Use Python 3.10 or
later; the tested version is 3.10.16. Required packages are NumPy, SciPy,
pandas, xarray, Matplotlib, and netCDF4, with versions in
`code/requirements.txt`.

From the package directory:

```sh
python -m pip install -r code/requirements.txt
python -B code/reproduce_figures.py --outdir figures
```

Choose a new output directory. Existing directories are not overwritten, and
`code/`, `data/`, and `config/` cannot be used as output locations. The wrapper
runs each plotting script separately and generates Table S1 in an external
temporary directory before moving it to the output directory.

Outputs are `Fig1.png/.pdf` through `Fig4.png/.pdf`, `FigS1.png/.pdf` through
`FigS7.png/.pdf`, `TableS1.csv`, `TableS1.md`, and
`TableS1_threshold_sensitivity.csv`. The wrapper also saves the dates selected
for Fig. 3 and source-data CSVs for S5–S7. It assigns the manuscript S1/S2
names to files whose original numbering was reversed.

## Bundled data

| Directory | Contents | Figures |
|---|---|---|
| `data/modes/` | TCWV climatology, EOF1/2, and mode diagnostics | 1, S1, S2, Table S1 |
| `data/sst/` | Annual SST/PC values and seasonal width sensitivity | 2, S3 |
| `data/maps/` | Six daily TCWV/precipitation maps and the ocean mask | 3 |
| `data/organization/` | SON AI_grad, PC-group samples, annual means, and event frequencies | 3, 4, S6 |
| `data/relationships/` | Annual SST/PC/organization records at reference and sensitivity widths | 4, S4–S7 |

`data/manifest.csv` lists each file's source, selection, size, and SHA256.
Subsets retain the original values, missing masks, and CSV numeric precision;
NetCDF compression is lossless. No interpolation, averaging, or requantization
was used to prepare the figure package.

For Fig. 3, all 4,186 SON AI_grad samples from 1979–2024 at the ±5° window are
retained for selecting each month's minimum and maximum. Map fields are
limited to the resulting six dates and the visible region, with a one-cell
halo for contour clipping. Each `era5_daily_YEAR.nc` file contains only the
required day, **not a complete year**. These data are plot inputs, not complete
Stage 1–5 outputs.

## Run the analysis from ERA5

Stage 1–5 and their shared module are included in `code/`. Filenames such as
`fix1` and `v1` identify the implementations used for reproduction; they are
retained for compatibility. `atlantic_itcz_prepare_era5_fix1.py` is Stage 1
version 1.0.1.

### Input archive

Stage 1 requires the **DKRZ E5 daily `1D` (P1D) GRIB products for 1979–2024**
(16,802 days), plus invariant land-sea mask (LSM) fields. The reader does not
convert arbitrary CDS hourly or NetCDF downloads into these products. Matching
variable names alone do not establish matching aggregation, grid, or timing.

Directories below are relative to `ERA5_ROOT`; `an` and `fc` denote analysis
and forecast. Both streams were present where listed. A file being discovered
does not mean its values are retained after duplicate times are removed.

| Variable | paramId | Source units | Directories |
|---|---|---|---|
| TCWV | 137 | kg m-2 | `sf/an/1D/137/`, `sf/fc/1D/137/` |
| Total precipitation | 228 | m per daily accumulation | `sf/fc/1D/228/` |
| SST | 34 | K | `sf/an/1D/034/`, `sf/fc/1D/034/` |
| 10 m zonal/meridional wind | 165, 166 | m s-1 | `sf/an/1D/165/`, `sf/an/1D/166/` |
| 2 m temperature | 167 | K | `sf/an/1D/167/` |
| Boundary-layer height | 159 | m | `sf/an/1D/159/`, `sf/fc/1D/159/` |
| Mean sea level pressure | 151 | Pa | `sf/an/1D/151/` |
| Temperature at **700 and 1000 hPa** | 130 | K | `pl/an/1D/130/` |
| Invariant LSM | 172 | fraction | `sf/an/IV/172/`, `sf/fc/IV/172/` |

Monthly filenames include `E5sf00_1D_YYYY-MM_137.grb` (analysis),
`E5sf12_1D_YYYY-MM_228.grb` (forecast precipitation), and
`E5pl00_1D_YYYY-MM_130.grb` (pressure-level temperature). SST uses the
zero-padded token `034`. Each monthly file must contain its daily records;
temperature files must include both required levels. The reader accepts
`.grb` and `.grib`. Static examples are `E5sf00_IV_2000-01-01_172.grb` and
`E5sf00_IV_INVARIANT_172.grb`; the nominal date does not make LSM year-specific.
LSM and pressure temperatures are loaded separately from `--params`.

The surface grid used for reproduction was reduced Gaussian N320 (542,080
points). Stage 1 averages finite native points within 1° cells over
65°W–10°W, 25°S–35°N. Preserve the source grid, product, time metadata, and
directory layout when reproducing this analysis.

**SST source order matters.** Filenames are sorted lexically and the first
record at each exact valid time is retained. In this layout, analysis SST
precedes forecast SST; the values can differ. The overlapping July–August
2023 forecast monthly/daily records were identical to each other, but analysis
SST was used. After exact-time deduplication, times are floored to calendar
days and distinct retained times on the same day are averaged. LSM uses the
first successfully read candidate. Changing this order can change results.

P1D precipitation is a daily accumulation in meters, converted to mm/day by
multiplying by 1000. Bounded hourly checks supported the sum from D 01–23 UTC
plus D+1 00 UTC. No extra correction is applied for a `0–13 hours` step label.
The reader disables cfgrib index files with `indexpath=""`.

### Environments and execution

Stage 1 needs cfgrib, the ecCodes Python package, and the native ecCodes
library. `code/requirements-era5-recorded.txt` records the Python 3.13.11 GRIB
environment; `code/requirements.txt` records the Python 3.10.16 analysis/plot
environment. Use these separately; they are not a single platform-independent
environment lock. Installing the plotting requirements alone does not supply
GRIB support.

Set the archive path, a **new output root outside the package**, and the
Python executables. Run stages in order and stop on failure. Some science
scripts can reuse or overwrite outputs, so do not reuse a previous `RUN` or
enable overwrite, incomplete-day, or skip-LTS options for reproduction.

```sh
export ERA5_ROOT=/path/to/authorized/ERA5/E5
export RUN=/path/to/new/full_analysis
export PY_STAGE1=/path/to/grib-environment/bin/python
export PY_ANALYSIS=/path/to/analysis-environment/bin/python
export MPLBACKEND=Agg
```

The commands below use the settings in `config/stage_commands.json` and the
per-stage JSON files. The scientific scripts read CLI arguments, not JSON.

```sh
# Stage 1
"${PY_STAGE1}" -B code/atlantic_itcz_prepare_era5_fix1.py --era5-root "${ERA5_ROOT}" --years 1979 2024 --lon-range -65 -10 --lat-range -25 35 --resolution 1 --lsm-threshold 0.5 --params tcwv tp sst 10u 10v 2t blh msl --no-recursive-search --outdir "${RUN}/stage1"
```

```sh
# Stage 2
"${PY_ANALYSIS}" -B code/atlantic_itcz_tcwv_modes.py --prepare-dir "${RUN}/stage1" --outdir "${RUN}/stage2" --analysis-lon-range -50.0 -20.0 --analysis-lat-range -15.0 30.0 --center-search-lat -5.0 15.0 --rel-lat-range -20.0 20.0 --metric-rel-range -10.0 15.0 --smooth-sigma-deg 1.0 --fractional-thresholds 0.7 0.8 0.9 --absolute-thresholds 45.0 48.0 50.0 --n-modes 3 --classification-margin 0.15 --min-ocean-fraction 0.5
```

```sh
# Stage 3
"${PY_ANALYSIS}" -B code/atlantic_itcz_state_analysis.py --prepare-dir "${RUN}/stage1" --modes-dir "${RUN}/stage2" --outdir "${RUN}/stage3" --analysis-lon-range -50.0 -20.0 --analysis-lat-range -15.0 30.0 --sst-metric-lon-range -50.0 -20.0 --sst-metric-lat-range -25.0 35.0 --sst-peak-lat-range -5.0 20.0 --eq-south-lat-range -15.0 0.0 --eq-north-lat-range 0.0 15.0 --centered-sst-half-width-deg 15.0 --sst-scale-sweep-deg 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 22.5 25.0 --sst-scale-bootstrap-samples 1000 --sst-scale-bootstrap-seed 260904 --exhaustive-sst-series-types raw --series-types raw linear_detrended --map-vars sst tcwv tp u10 v10 blh lts --top-n 5 --bottom-n 5
```

```sh
# Stage 4
"${PY_ANALYSIS}" -B code/atlantic_itcz_daily_organization.py --prepare-dir "${RUN}/stage1" --modes-dir "${RUN}/stage2" --selected-shape-modes "${RUN}/stage2/selected_shape_modes.csv" --outdir "${RUN}/stage4" --analysis-lon-range -50.0 -20.0 --analysis-lat-range -15.0 30.0 --center-search-lat -5.0 15.0 --center-smooth-sigma-deg 1.0 --relative-half-width-km 1536.0 --relative-half-width-deg 5.0 7.5 10.0 12.5 15.0 17.5 --sweep-half-width-deg 5.0 18.0 0.5 --no-width-sweep --period-types monthly seasonal --series-types raw linear_detrended --top-n 5 --bottom-n 5 --event-percentile 90.0 --event-windows rel_5deg rel_7p5deg rel_10deg rel_12p5deg rel_15deg rel_17p5deg --representative-window rel_15deg --representative-n 3 --representative-min-separation-days 5 --no-representative-maps
```

```sh
# Stage 5
"${PY_ANALYSIS}" -B code/atlantic_itcz_sst_organization.py --state-dir "${RUN}/stage3" --stage4-dir "${RUN}/stage4" --outdir "${RUN}/stage5" --sst-half-width 12.5 --organization-window rel_5deg --season SON --mode 1 --series-type raw --dpi 250
```

| Stage | Inputs | Output under `${RUN}` |
|---|---|---|
| 1 | Daily GRIB, LSM, and pressure temperatures | `stage1/`: daily files, monthly/seasonal means, static fields, metadata |
| 2 | Complete Stage 1 monthly/seasonal/static fields | `stage2/`: EOF/PC fields and diagnostic/selection tables |
| 3 | Complete Stage 1 and new Stage 2 | `stage3/`: SST metrics, regressions, composites, and bootstrap results |
| 4 | Complete Stage 1 daily/static fields and new Stage 2 | `stage4/`: daily organization, event summaries, and PC groups |
| 5 | New Stage 3 and Stage 4 tables | `stage5/`: annual joins, correlations, and auxiliary figures |

Stage 3 uses 1,000 bootstrap samples with seed 260904. Stage 4 keeps the named
windows and omits the optional dense sweep/maps. Complete Stage 1 aggregation
has 552 months and 183 seasons: DJF 1980 includes December 1979, while DJF
1979 is incomplete. Full-period verification used annual processing followed
by aggregation. GRIB processing needs compute resources (about 9 GiB peak
memory in previous annual runs), rather than a login-node test.

The correlations describe associations, not causal effects. Event-day
fractions count days, not statistically independent episodes.

## Repository files

`GITHUB_FILES.txt` lists the 46 files intended for `atlantic-itcz-variability`.
`.gitignore` excludes generated figures, full analysis outputs, environments,
logs, and caches. Update the root allowlist when adding a release file.
Citation metadata are in [`CITATION.cff`](CITATION.cff).

AI assistance: AI tools helped prepare the figure-generation wrapper and
package documentation.
