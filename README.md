# aerosol-convection

The analysis code is in `scripts/` and the figures it writes are in `plots/`.

## Setup

```bash
mamba env create -f environment.yml
conda activate aerosol-convection
cp scripts/config.yaml scripts/config.local.yaml
```

Then edit the paths at the top of `scripts/config.local.yaml` (gitignored) to point at your
copies of the data (see below). `plots_dir` is where the figures are written.

Every script is run from the repo root, with the config as its first argument:

```bash
python scripts/<script>.py scripts/config.local.yaml
```

The figures use Arial. The environment installs it (`mscorefonts`), so the figures look the same
on machines without a system Arial (e.g. NERSC).

## Input data

The data are on Zenodo ([doi:10.5281/zenodo.24261154](https://doi.org/10.5281/zenodo.24261154))
as four zips. Unzip them all into the same folder and set `celltracking_dir` and `gridded_dir`
to the two folders this creates. `data.zip` is enough for every figure except `plots_csec.py`,
which also needs `terrain.zip` and `case.zip`.

`celltracking_dir` holds up to four subfolders:

| Folder          | Contents                                                        | Zip |
|-----------------|-----------------------------------------------------------------|-----|
| `data/`         | Preprocessing outputs (see below)                               | `data.zip` |
| `stats/`        | Track statistics and per-track environment/aerosol parameters, made with [PyFLEXTRKR](https://github.com/FlexTRKR/PyFLEXTRKR) + [cacti-radar](https://github.com/ICLASS-SFA/cacti-radar/tree/time_series_enochjo). Only preprocessing reads it | `stats.zip` |
| `terrain/`      | `terrain_d01.nc`, `lat_d01.nc`, `lon_d01.nc`                    | `terrain.zip` |
| `celltracking/` | Per-timestep cell masks, `20181015.0000_20190303.0000/celltracks_*.nc`, made with [PyFLEXTRKR](https://github.com/FlexTRKR/PyFLEXTRKR) (case-study timestep only) | `case.zip` |

The preprocessing scripts read these six files from `stats/`:

- `trackstats_20181015.0000_20190303.0000.nc`
- `stats_goes16_20181015.0000_20190303.0000.nc`
- `stats_3d_ppi_20181015.0000_20190303.0000.nc`
- `interpsonde_parameters_celltrack_20181015.0000_20190303.0000.nc`
- `aerosol_parameters_celltrack_20181015.0000_20190303.0000.nc`
- `combomet_parameters_celltrack_20181015.0000_20190303.0000.nc`

`gridded_dir` holds the TARANIS gridded radar files
(`taranis_corcsapr2cfrppiqcM1.c1.*_gridded.nc`). Only `plots_csec.py` needs it. `case.zip`
has the case-study file. The full set is on NERSC (m1657 members) in `/global/cfs/cdirs/m1657/enochjo/taranis/taranis_corcsapr2cfrppiqcM1_gridded.c1/`.

## Config options

Besides the paths, the config has four settings:

- **`CI_variable`** sets the CI definition: the metric whose first peak within 2 h of track
  initiation sets the time each cell is sampled. The choices are `eth_10dbz` (main), `Tb_min`,
  `zh_max`, `rr_max`, `volrain` and `rainarea`. Most figure names include it, e.g.
  `corr_full_eth_10dbz.png`.
- **`mask`** sets which cell filter from `the_chosen_few_{CI_variable}.nc` is applied. The
  default is `default` (details in `scripts/pipe_filter.py`). The other masks are
  sensitivity tests:
  - radar range: `60km`, `70km`, `90km`, `100km`, `110km`
  - minimum lifetime: `2ts`, `3ts`, `5ts`, `6ts`
  - drop the Nth most prolific start date: `date1`–`date4`

  A non-default mask adds `_{mask}` to the figure name, so the manuscript figures are never
  overwritten. Only the figures marked `{sfx}` below follow `mask`. `plots_csec.py` and the
  `supp_*` scripts use fixed masks.
- **`lingress_figure`** picks the `plots_lingress.py` figure: `main` or `supp` (every HID
  category).
- **`refresh`** refits the cached regressions (see [Making figures](#making-figures)).

## Preprocessing (optional)

The Zenodo `data/` folder already contains everything the `pipe_*.py` scripts produce. Run them only if
you change the inputs or the filter:

1. `pipe_reindex.py` finds each track's CI time for the configured `CI_variable`. It samples
   the radar and satellite statistics at that time, takes the sounding, surface and aerosol
   parameters as medians over the hour before track initiation, and writes
   `data/df_updr_{CI_variable}.nc` (one row per track).
2. `pipe_filter.py` applies the cell filters (coupled to the surface, east of the ridge, not
   split-born, no early merge, distance from the radar, lifetime, daytime start, 10-dBZ
   ETH ≥ 6 km, no NaNs). It writes the boolean masks to `data/the_chosen_few_{CI_variable}.nc`
   as `good_indices` (default) and `good_indices_{mask}` (variants).

`plots_pies.py`, `supp_cellprop.py` and `supp_lingress.py` read several CI definitions at
once. So if you regenerate, run both steps once for each of the six `CI_variable` values.

## Making figures

Each figure script reads `df_updr_*.nc` and `the_chosen_few_*.nc` and writes PNGs to
`plots_dir`. The exception is `plots_csec.py`, which reads only the default mask of
`the_chosen_few_eth_10dbz.nc`, plus the radar, cell-mask and terrain files.

Three scripts fit regressions over 5000 repeated 70/30 train/test splits (`lib_regression.py`)
and cache the fits in `data/cache_*.nc`: `plots_lingress.py`, `plots_pies.py` and
`supp_lingress.py`. The caches are not on Zenodo, so each script fits on its first run and
reuses the cache after that. Each combination of the settings a script reads (`CI_variable`,
`mask`, `lingress_figure`) gets its own cache. A cache is not refit when its inputs or the
fitting code change, so set `refresh: true` after re-running preprocessing or changing
predictors, responses, panels or `lib_regression.py`, then set it back to `false`.

## Scripts and figures

`{CI}` = `CI_variable`, `{sfx}` = `_{mask}` for non-default masks (empty otherwise).

The paper's figures use `CI_variable: eth_10dbz` and the default mask. `lingress` and
`supp_lingress` also appear with `rainarea`: set `CI_variable: rainarea` and re-run those two
scripts.

### Main figures

| Script                  | Figure(s)                         | What it shows |
|-------------------------|-----------------------------------|---------------|
| `plots_csec.py`         | `csec_xz.png`, `csec_xy.png`      | Case study: vertical cross-section through one case (xz) and a plan view with terrain and the sampled cells (xy). The case time and section endpoints are hard-coded near the top of the script. |
| `plots_hist_env.py`     | `hist_env_{CI}{sfx}.png`          | Histograms of storm, environment and CCN variables for the filtered cells |
| `plots_corr_mini.py`    | `corr_mini_{CI}{sfx}.png`         | Spearman correlogram of a subset of the `corr_full` variables |
| `plots_ridge_radar.py`  | `ridge_radar_{CI}{sfx}.png`       | Ridge plots of per-track radar percentiles (Z<sub>H</sub>, Z<sub>DR</sub>, D<sub>m</sub>, K<sub>DP</sub>, rain rate, …) |
| `plots_ridge_hid.py`    | `ridge_hid_{CI}{sfx}.png`         | Ridge plots of per-track HID fractions, in four panels (rimed ice, unrimed ice, liquid, melting rimed ice) |
| `plots_lingress.py`     | `lingress_{CI}{sfx}.png` (paper: `eth_10dbz`, `rainarea`) | Standardized regression coefficients of each response on CCN, CAPE, PW, RH and shear: univariate (red) vs multivariate (blue) |
| `plots_pies.py`         | `lingress_pies{sfx}.png`          | Each predictor's share of the total multivariate effect for six selected responses. Uses both `eth_10dbz` and `rainarea` |

### Supplementary figures

| Script              | Figure(s)                                                     | What it shows |
|---------------------|---------------------------------------------------------------|---------------|
| `plots_corr_full.py`| `corr_full_{CI}{sfx}.png`                                     | Spearman correlogram of all variables |
| `plots_lingress.py` | `lingress_{CI}_supp{sfx}.png` (`lingress_figure: supp`)      | The main `lingress` figure with every HID category |
| `supp_misc.py`      | `supp_thetae_diff.png`, `supp_dates_{CI}.png`, `supp_radardist_{CI}.png` | Checks on the filter choices: θ<sub>e</sub><sup>MU</sup> − θ<sub>e</sub><sup>SFC</sup> (surface coupling), cells per start date, cells vs radar range (`110km` mask) |
| `supp_cellprop.py`  | `supp_sampletime.png`, `supp_duration_{CI}.png`               | Box plots of CI time across all six CI definitions, and of echo top / rain area across the lifetime masks |
| `supp_lingress.py`  | `supp_lingress_{CI}.png` (paper: `eth_10dbz`, `rainarea`)     | Sensitivity of the CCN coefficient across radar-range, lifetime, CI-definition and excluded-date variants |
| —                   | `supp_calibration.png`                                        | Radar calibration. Made separately, not by this repo |
