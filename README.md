# aerosol-convection

The analysis code is in `scripts/` and the figures it writes are in `plots/`.

## Setup

```bash
mamba env create -f environment.yml
conda activate aerosol-convection
```

Then edit the paths at the top of `scripts/config.yaml` to point at your copies of the data
(see below). `plots_dir` is where the figures are written.

Every script is run with the config as its first argument:

```bash
python scripts/<script>.py scripts/config.yaml
```

The figures use Arial. On machines without it (e.g. NERSC), matplotlib warns and falls back to
another font. The figures still render.

## Input data

The data are on Zenodo ([doi:10.5281/zenodo.23094051](https://doi.org/10.5281/zenodo.23094051))
as four zips. Unzip them all into the same folder and set `celltracking_dir` and `gridded_dir`
to the two folders this creates. For the figures alone, `data.zip` is enough.

`celltracking_dir` must contain four subfolders:

| Folder          | Contents                                                        | Source |
|-----------------|-----------------------------------------------------------------|--------|
| `stats/`        | Track statistics and per-track environment/aerosol parameters   | [Zenodo](https://doi.org/10.5281/zenodo.23094051), made with [PyFLEXTRKR](https://github.com/FlexTRKR/PyFLEXTRKR) + [cacti-radar](https://github.com/ICLASS-SFA/cacti-radar/tree/time_series_enochjo) |
| `celltracking/` | Per-timestep cell masks, `20181015.0000_20190303.0000/celltracks_*.nc` | [Zenodo](https://doi.org/10.5281/zenodo.23094051) (case-study timestep only), made with [PyFLEXTRKR](https://github.com/FlexTRKR/PyFLEXTRKR) |
| `data/`         | Pipeline outputs (see below)                                    | [Zenodo](https://doi.org/10.5281/zenodo.23094051) |
| `terrain/`      | `terrain_d01.nc`, `lat_d01.nc`, `lon_d01.nc`                    | [Zenodo](https://doi.org/10.5281/zenodo.23094051) |

The pipeline reads these six files from `stats/`:

- `trackstats_20181015.0000_20190303.0000.nc`
- `stats_goes16_20181015.0000_20190303.0000.nc`
- `stats_3d_ppi_20181015.0000_20190303.0000.nc`
- `interpsonde_parameters_celltrack_20181015.0000_20190303.0000.nc`
- `aerosol_parameters_celltrack_20181015.0000_20190303.0000.nc`
- `combomet_parameters_celltrack_20181015.0000_20190303.0000.nc`

`gridded_dir` holds the TARANIS gridded radar files
(`taranis_corcsapr2cfrppiqcM1.c1.*_gridded.nc`). Only `plots_csec.py` needs it. Zenodo has
the case-study file. The full set on NERSC is in `/global/cfs/cdirs/m1657/enochjo/taranis/taranis_corcsapr2cfrppiqcM1_gridded.c1/`.

## Config options

`scripts/config.yaml` has two analysis switches besides the paths:

- **`CI_variable`** sets the CI definition: the metric whose first peak within 2 h of track
  initiation sets the time each cell is sampled. The choices are `eth_10dbz` (main), `Tb_min`,
  `zh_max`, `rr_max`, `volrain` and `rainarea`. Most figure names end in it, e.g.
  `corr_full_eth_10dbz.png`.
- **`mask`** sets which cell filter from `the_chosen_few_{CI_variable}.nc` is applied. The
  default is `default` (details in `scripts/pipe_filter.py`). The other masks are
  sensitivity tests:
  - radar range: `60km`, `70km`, `90km`, `100km`, `110km`
  - minimum lifetime: `2ts`, `3ts`, `5ts`, `6ts`
  - drop the Nth most prolific start date: `date1`–`date4`

  A non-default mask adds `_{mask}` to the figure name, so the manuscript figures are never
  overwritten. Only the `plots_*` scripts (except `plots_csec.py`) read `mask`. These are the
  figures marked `{sfx}` below. `plots_csec.py` and the `supp_*` scripts use fixed masks.

## Pipeline (optional)

The Zenodo `data/` folder already contains everything the pipeline produces. Run it only if
you change the inputs or the filter:

1. `pipe_reindex.py` finds each track's CI time for the configured `CI_variable`. It samples
   the radar, satellite, sounding, aerosol and surface statistics at that time and writes
   `data/df_updr_{CI_variable}.nc` (one row per track).
2. `pipe_filter.py` applies the cell filters (coupled to the surface, east of the ridge, not
   split-born, no early merge, distance from the radar, lifetime, daytime start, 10-dBZ
   ETH ≥ 6 km, no NaNs). It writes the boolean masks to `data/the_chosen_few_{CI_variable}.nc`
   as `good_indices` (default) and `good_indices_{mask}` (variants).

`plots_pies.py`, `supp_cellprop.py` and `supp_lingress.py` read several CI definitions at
once. So if you regenerate, run both steps once for each of the six `CI_variable` values.

## Making figures

Each figure script reads `df_updr_*.nc` and `the_chosen_few_*.nc` and writes PNGs to
`plots_dir`.

Three scripts run 5000-split bootstrap regressions (`lib_regression.py`) and cache the fits
in `data/cache_*.nc`: `plots_lingress.py`, `plots_pies.py` and `supp_lingress.py`. The
caches are not on Zenodo, so the first run of each script fits the regressions. After that,
the cache is reused. The cache is not checked against the current settings, so pass
`--refresh` after changing predictors, responses, masks or `lib_regression.py`:

```bash
python scripts/plots_lingress.py scripts/config.yaml --refresh
```

## Scripts and figures

`{CI}` = `CI_variable`, `{sfx}` = `_{mask}` for non-default masks (empty otherwise).

### Main figures

| Script                  | Figure(s)                         | What it shows |
|-------------------------|-----------------------------------|---------------|
| `plots_csec.py`         | `csec_xz.png`, `csec_xy.png`      | Case study: vertical cross-section through one case (xz) and a plan view with terrain and the sampled cells (xy). The case time and section endpoints are hard-coded near the top of the script. |
| `plots_hist_env.py`     | `hist_env_{CI}{sfx}.png`          | Histograms of storm, environment and CCN variables for the filtered cells |
| `plots_corr_full.py`    | `corr_full_{CI}{sfx}.png`         | Spearman correlogram of all variables |
| `plots_corr_mini.py`    | `corr_mini_{CI}{sfx}.png`         | Spearman correlogram of a subset of variables, for the main text |
| `plots_ridge_radar.py`  | `ridge_radar_{CI}{sfx}.png`       | Ridge plots of per-track radar percentiles (Z<sub>H</sub>, Z<sub>DR</sub>, D<sub>m</sub>, K<sub>DP</sub>, rain rate, …) |
| `plots_ridge_hid.py`    | `ridge_hid_{CI}{sfx}.png`         | Ridge plots of per-track HID fractions, in four panels (rimed ice, unrimed ice, liquid, melting rimed ice) |
| `plots_lingress.py`     | `lingress_{CI}{sfx}.png`          | Standardized regression coefficients of each response on CCN, CAPE, PW, RH and shear: univariate (red) vs multivariate (blue) |
| `plots_pies.py`         | `lingress_pies{sfx}.png`          | Each predictor's share of the total multivariate effect for six selected responses. Uses both `eth_10dbz` and `rainarea` |

### Supplementary figures

| Script              | Figure(s)                                                     | What it shows |
|---------------------|---------------------------------------------------------------|---------------|
| `supp_misc.py`      | `supp_thetae_diff.png`, `supp_dates_{CI}.png`, `supp_radardist_{CI}.png` | Checks on the filter choices: θ<sub>e</sub><sup>MU</sup> − θ<sub>e</sub><sup>SFC</sup> (surface coupling), cells per start date, cells vs radar range (`110km` mask) |
| `supp_cellprop.py`  | `supp_sampletime.png`, `supp_duration_{CI}.png`               | Box plots of CI time across all six CI definitions, and of echo top / rain area across the lifetime masks |
| `supp_lingress.py`  | `supp_lingress_{CI}.png`                                      | Sensitivity of the CCN coefficient across radar-range, lifetime, CI-definition and excluded-date variants |
| —                   | `supp_calibration.png`                                        | Radar calibration. Made in the [CSAPR2-calibration](https://github.com/conscripted-coder/CSAPR2-calibration) repo |

### Shared code (not run directly)

- `lib_regression.py`: `fit_bootstrap_regression`, the repeated 70/30 train/test split
  regression that the `*lingress*` and `pies` scripts use.
- `plots_ridge_radar.py` also exports `draw_ridge` and `fmt_median`. `plots_ridge_hid.py`
  and `plots_hist_env.py` import them, so keep these scripts together in `scripts/`.

Figures named `*_rainarea.png` in `plots/` are the same scripts run with
`CI_variable: rainarea`.
