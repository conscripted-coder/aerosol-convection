"""Supplementary sensitivity figure: responses down the rows, four sampling sweeps across the
columns, one bar per variant giving the median standardized CCN coefficient (deep = 95% split
range excludes zero, hatched = control run). Fits are cached; pass --refresh to refit.
"""

import os
import sys

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

import yaml
from lib_regression import fit_bootstrap_regression, DEFAULT_PERCENTILES
import lib_style

config_file = sys.argv[1]
stream      = open(config_file, 'r')
config      = yaml.full_load(stream)

celltracking_dir = config['celltracking_dir']
data_dir         = celltracking_dir + '/data'
plots_dir        = config['plots_dir']
CI_variable      = config['CI_variable']

# Coefficient span of one row lane: the single magnitude scale shared by every row. Not
# clipped on purpose: overflow into the neighbouring row is the cue to raise LANE_SPAN.
LANE_SPAN = 0.42

# Each row's baseline sits only as high as its most negative bar needs, since coefficients are
# mostly positive. Baselines therefore don't line up across rows; compare bar heights.
NEG_HEADROOM = 1.06  # slack so the longest negative bar doesn't touch the lane floor
MIN_NEG_FRAC = 0.05  # minimum downward room (fraction of LANE_SPAN), for rows with no negatives

# Dotted -/+ GRID_VALUE reference lines per row (no y ticks); skipped if outside the lane
GRID_VALUE = 0.2

BAR_WIDTH      = 0.7  # thickness of one bar, x data units
ROW_HALFHEIGHT = 0.5  # half-height of a row lane, y data units

# Horizontal edge-to-edge gaps, in BAR_WIDTHs
GAP_WITHIN_GROUP    = 0    # between variants of the same sweep
GAP_BETWEEN_COLUMNS = 1.5  # between sweeps

# Vertical gaps, y data units (a lane spans 2 * ROW_HALFHEIGHT)
GAP_BETWEEN_RESPONSES = 0.3  # between response lanes
GAP_BETWEEN_BANDS     = 1.0  # between bands

# Width is fixed; INCH_PER_Y_UNIT trades magnitude resolution against figure height
FIG_WIDTH       = 6.2
INCH_PER_Y_UNIT = 0.5

LABEL_MARGIN = 4.2  # left margin for row labels, x data units

# A faint rule separates rows; list the response above a boundary to make it heavy black
MAJOR_RULE_AFTER = {
    'rain_area_sampled',   # end of the radar/precip rows, start of the HID rows
}
RULE_RIGHT_EXT = 1.5  # rules run this far past the last bar group, under the +/-GRID_VALUE labels
RULE_MINOR     = ('0.85', 0.6)  # (color, linewidth)
RULE_MAJOR     = ('black', 0.9)

COLOR_SIG    = 'deepskyblue'  # 95% split range excludes 0
COLOR_NONSIG = '#d9effb'      # 95% split range includes 0

# The control is hatched, since colour already marks significance
CONTROL = (CI_variable, 'default'); CI_MECH = CI_variable

CONTROL_HATCH   = '/////'  # drop to '///' if narrower bars smear it into a solid tone
HATCH_LINEWIDTH = 0.4      # only settable through the global rcParam
EDGE_WIDTH      = 0.4

mpl.rcParams['hatch.linewidth'] = HATCH_LINEWIDTH

# Columns: four sampling sweeps of (CI_variable, mask) variants. "default" is the 80 km,
# 4-timestep mask, hence its place mid-sweep.
COLUMNS = [
    {
        'title':    'Radar Distance',
        'variants': [(CI_MECH, m) for m in
                     ['60km', '70km', 'default', '90km', '100km', '110km']],
        'labels':   ['60 km', '70 km', '80 km', '90 km', '100 km', '110 km'],
    },
    {
        'title':    'Cell Lifetime',
        'variants': [(CI_MECH, m) for m in ['2ts', '3ts', 'default', '5ts', '6ts']],
        'labels':   ['2 time steps', '3 time steps', '4 time steps', '5 time steps', '6 time steps'],
    },
    {
        'title':    'Sampling Var',
        'variants': [('eth_10dbz', 'default'), ('Tb_min', 'default'),
                     ('zh_max', 'default'), ('rr_max', 'default'),
                     ('volrain', 'default'), ('rainarea', 'default')],
        'labels':   ['10 dBZ ETH', r'T$_{\mathrm{B}}$', r'Z$_{\mathrm{H}}$',
                     'Rain Rate', 'Vol. Rain', 'Precip. Area'],
    },
    {
        'title':    'Excluded Date',
        # Rank-named masks; the actual dates are read from the mask attrs below
        'variants': [(CI_MECH, m) for m in ['default', 'date1', 'date2', 'date3', 'date4']],
        'labels':   None,
    },
]

N_SLOTS = 6  # widest column; shorter columns leave their rightmost slots blank

BANDS = [
    [
        (r'Z$_{\mathrm{H,liq,50^{th}}}$',    'zh_liq_p50'),
        (r'Z$_{\mathrm{H,ice,50^{th}}}$',    'zh_frz_p50'),
        (r'Z$_{\mathrm{DR,50^{th}}}$',       'zdr_liq_p50'),
        (r'D$_{\mathrm{m,50^{th}}}$',        'Dm_p50'),
        (r'K$_{\mathrm{DP,50^{th}}}$',       'kdp_liq_p50'),
        (r'Rain Rate $_{\mathrm{50^{th}}}$', 'rr_p50'),
        # ('Rain Water\nContent', 'rwc_p50'),
        ('Volumetric Rain',                  'volrain'),
        ('Precip. Area',                     'rain_area_sampled'),
        ('Graupel-low',                      'low_density_graupel'),
        ('Graupel-high',                     'high_density_graupel_frz'),
        ('Hail',                             'hail_frz'),
        ('Rimed Ice',                        'rime_frz'),
        ('Melt. Rimed Ice',                  'rime_liq')
    ],
]

PRED    = [col for band in BANDS for _, col in band]
row2idx = {col: i for i, col in enumerate(PRED)}

# Same predictors as plots_lingress, so the CCN bars are comparable to its blue bars
ENV_VARS = ['ccn', 'CAPE_sfc', 'pw_mwr', 'rh_600mb', 'shear_mag_bulk_0to9km']
# ENV_VARS = ['ccn', 'CAPE_sfc', 'pw_mwr', 'rh_600mb']
CCN_IDX = ENV_VARS.index('ccn')


def date_labels(ci_var, masks):
    """Label date masks 'excl. MM-DD' from their ``excluded_date`` attr ('all dates' if none)."""
    ds  = xr.open_dataset(data_dir + "/the_chosen_few_" + ci_var + ".nc")
    out = []
    for m in masks:
        var   = 'good_indices' if m == 'default' else f'good_indices_{m}'
        attrs = ds[var].attrs
        if 'excluded_date' in attrs:
            out.append('excl. ' + pd.Timestamp(attrs['excluded_date']).strftime('%m-%d'))
        else:
            out.append('all dates')
    return out


def sort_by_date(variants):
    """Reorder date variants chronologically by excluded_date, default mask first."""
    ci_var  = variants[0][0]
    ds      = xr.open_dataset(data_dir + "/the_chosen_few_" + ci_var + ".nc")
    default = [v for v in variants if v[1] == 'default']
    dated   = [v for v in variants if v[1] != 'default']
    dated.sort(key=lambda v: pd.Timestamp(ds[f'good_indices_{v[1]}'].attrs['excluded_date']))
    return default + dated


for column in COLUMNS:
    if column['labels'] is None:
        column['variants'] = sort_by_date(column['variants'])
        column['labels']   = date_labels(column['variants'][0][0],
                                         [m for _, m in column['variants']])


# Sample sizes from the mask sums (equal to len(df) at fit time, as nothing else is dropped)
_mask_files = {}


def variant_n(variant):
    """Number of cells a (CI_variable, mask) variant keeps."""
    ci_var, mask = variant
    if ci_var not in _mask_files:
        _mask_files[ci_var] = xr.open_dataset(data_dir + "/the_chosen_few_" + ci_var + ".nc")
    var = 'good_indices' if mask == 'default' else f'good_indices_{mask}'
    return int(_mask_files[ci_var][var].values.sum())


# Count goes in the label text so it rotates with it
for column in COLUMNS:
    column['labels'] = [f'{lab} ({variant_n(v)})'
                        for lab, v in zip(column['labels'], column['variants'])]


# Fit options are not checked against the cache: pass --refresh after changing the
# variants, responses, predictors, or lib_regression
CACHE_PATH = data_dir + '/cache_supp_lingress_' + CI_variable + '.nc'


# "ci_var|mask", since a NetCDF coordinate cannot hold tuples
def _key(variant):
    return '|'.join(variant)


if '--refresh' in sys.argv or not os.path.exists(CACHE_PATH):
    # Dedup, so the control is fitted once
    variants = sorted({v for column in COLUMNS for v in column['variants']})

    results = {}
    for variant in variants:
        ci_var, mask = variant
        df = xr.open_dataset(data_dir + '/df_updr_' + ci_var + '.nc').to_dataframe()
        var  = 'good_indices' if mask == 'default' else f'good_indices_{mask}'
        good = xr.open_dataset(data_dir + '/the_chosen_few_' + ci_var + '.nc')[var].values
        df   = df[good]
        ds   = fit_bootstrap_regression(df, ENV_VARS, PRED)
        # CCN only: (percentile, response); percentile index 0/2/4 = 2.5/50/97.5
        coef             = ds.coefficient.transpose('percentile', 'label', 'predictor').values
        results[variant] = coef[:, :, CCN_IDX]
        print(f"  fit {ci_var:>10s} / {mask:<8s}: n = {len(df)}")

    da = xr.DataArray(
        np.stack([results[v] for v in variants]),
        dims=('variant', 'percentile', 'response'),
        coords={'variant':    [_key(v) for v in variants],
                'percentile': list(DEFAULT_PERCENTILES),
                'response':   PRED},
        name='ccn_coefficient',
        attrs={'long_name': 'bootstrap percentiles of the standardized CCN coefficient'},
    )
    da.to_dataset().to_netcdf(CACHE_PATH)
    print(f"Fitted {len(variants)} variant(s); cache written to {CACHE_PATH}")
else:
    ds      = xr.open_dataset(CACHE_PATH)
    da      = ds.ccn_coefficient.transpose('variant', 'percentile', 'response')
    results = {tuple(str(v).split('|')): da.sel(variant=v).values for v in ds.variant.values}
    print(f"Read {len(results)} variant(s) from {CACHE_PATH}")


# Horizontal layout: one group of N_SLOTS bars per column
gap_slot = GAP_WITHIN_GROUP * BAR_WIDTH
gap_col  = GAP_BETWEEN_COLUMNS * BAR_WIDTH

BAR_PITCH    = BAR_WIDTH + gap_slot                               # centre-to-centre within group
GROUP_EXTENT = N_SLOTS * BAR_WIDTH + (N_SLOTS - 1) * gap_slot     # width of one full group

col_start   = [j * (GROUP_EXTENT + gap_col) for j in range(len(COLUMNS))]  # left edges
total_width = col_start[-1] + GROUP_EXTENT


def bar_x(j, k):
    """Centre of the k-th variant slot inside column group j."""
    return col_start[j] + BAR_WIDTH / 2 + k * BAR_PITCH


# Vertical layout: y increases upward and row 0 is on top, so rows step downward
ROW_HEIGHT = 2 * ROW_HALFHEIGHT                  # full lane height
ROW_PITCH  = ROW_HEIGHT + GAP_BETWEEN_RESPONSES  # centre-to-centre

# Downward room per row in coefficient units, from its most negative median
row_neg_room = {}
for col_name in PRED:
    iy      = row2idx[col_name]
    medians = [results[v][2, iy] for c in COLUMNS for v in c['variants']]
    row_neg_room[col_name] = max(max(0.0, -min(medians)) * NEG_HEADROOM,
                                 MIN_NEG_FRAC * LANE_SPAN)

# y of each band's first lane centre (not its zero line)
band_geom = []
cursor    = 0.0
for band in BANDS:
    band_geom.append({'start': cursor})
    extent = len(band) * ROW_PITCH - GAP_BETWEEN_RESPONSES
    cursor -= extent + GAP_BETWEEN_BANDS
top_y    = ROW_HALFHEIGHT  # top edge of the first lane
bottom_y = (band_geom[-1]['start'] - (len(BANDS[-1]) - 1) * ROW_PITCH) - ROW_HALFHEIGHT

# Title and label headroom, in y units. TITLE_PAD is from the top lane edge.
TITLE_PAD = 0.35
LABEL_PAD = 0.35  # bottom lane to the start of the 45-degree labels

# The longest 45-degree label (~0.7 in) needs ~0.5 in of height
LABEL_BLOCK = LABEL_PAD + 0.5 / INCH_PER_Y_UNIT

fig_height = (top_y + TITLE_PAD + 0.4 - bottom_y + LABEL_BLOCK) * INCH_PER_Y_UNIT
fig, ax    = plt.subplots(figsize=(FIG_WIDTH, fig_height))

# Titles centred over the slots each column uses
for j, column in enumerate(COLUMNS):
    n_used = len(column['variants'])
    ax.text((bar_x(j, 0) + bar_x(j, n_used - 1)) / 2, top_y + TITLE_PAD, column['title'],
            ha='center', va='bottom', fontsize=8, fontweight='bold')

# Row lanes: zero line, separator, label, bars
for b, (band, geom) in enumerate(zip(BANDS, band_geom)):
    for s, (row_label, col_name) in enumerate(band):
        lane_centre = geom['start'] - s * ROW_PITCH
        iy          = row2idx[col_name]

        zero_y = lane_centre - ROW_HALFHEIGHT + (row_neg_room[col_name] / LANE_SPAN) * ROW_HEIGHT

        ax.hlines(zero_y, -0.3, total_width + 0.3,
                  color='black', linestyle='--', alpha=0.4, linewidth=0.4, zorder=1)

        grid_offset = (GRID_VALUE / LANE_SPAN) * ROW_HEIGHT
        for sign in (-1, 1):
            grid_y = zero_y + sign * grid_offset
            if abs(grid_y - lane_centre) > ROW_HALFHEIGHT:
                continue
            ax.hlines(grid_y, -0.3, total_width + 0.3,
                      color='0.6', linestyle=':', linewidth=0.5, zorder=1)
            # Label every line, since some rows lack one of the two
            ax.text(total_width + 0.4, grid_y, rf'{sign * GRID_VALUE:+g}$\sigma$',
                    ha='left', va='center', fontsize=5.5, color='0.45', clip_on=False)

        # Divider in the gap above each lane but the first; hlines so it can run past xlim
        if not (b == 0 and s == 0):
            preceding_gap = GAP_BETWEEN_BANDS if s == 0 else GAP_BETWEEN_RESPONSES
            prev_col      = band[s - 1][1] if s > 0 else BANDS[b - 1][-1][1]
            color, lw     = RULE_MAJOR if prev_col in MAJOR_RULE_AFTER else RULE_MINOR
            ax.hlines(lane_centre + ROW_HALFHEIGHT + preceding_gap / 2,
                      -LABEL_MARGIN, total_width + RULE_RIGHT_EXT,
                      color=color, linewidth=lw, zorder=0, clip_on=False)

        # On the lane centre, not the baseline, so labels stay evenly spaced
        ax.text(-0.6, lane_centre, row_label, ha='right', va='center', fontsize=7)

        for j, column in enumerate(COLUMNS):
            for k, variant in enumerate(column['variants']):
                coef           = results[variant]
                med, low, high = coef[2, iy], coef[0, iy], coef[4, iy]

                excludes_zero = (low > 0 and high > 0) or (low < 0 and high < 0)
                color         = COLOR_SIG if excludes_zero else COLOR_NONSIG

                height = (med / LANE_SPAN) * ROW_HEIGHT
                ax.bar(bar_x(j, k), height, bottom=zero_y, width=BAR_WIDTH, color=color,
                       edgecolor='black', linewidth=EDGE_WIDTH, zorder=3,
                       hatch=CONTROL_HATCH if variant == CONTROL else None)

# Variant labels at 45 degrees under the bottom row, right end anchored at the bar centre
label_y = bottom_y - LABEL_PAD
for j, column in enumerate(COLUMNS):
    for k, label in enumerate(column['labels']):
        weight = 'bold' if column['variants'][k] == CONTROL else 'normal'
        ax.text(bar_x(j, k), label_y, label, rotation=45, rotation_mode='anchor',
                ha='right', va='center', fontsize=5.5, color='0.25', fontweight=weight)

ax.set_xlim(-LABEL_MARGIN, total_width + 0.3)
ax.set_ylim(bottom_y - LABEL_BLOCK, top_y + TITLE_PAD + 0.4)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

legend_elements = [
    Patch(facecolor=COLOR_SIG, edgecolor='black', linewidth=EDGE_WIDTH,
          label='95% interval excludes 0'),
    Patch(facecolor=COLOR_NONSIG, edgecolor='black', linewidth=EDGE_WIDTH,
          label='95% interval includes 0'),
    Patch(facecolor='white', edgecolor='black', linewidth=EDGE_WIDTH,
          hatch=CONTROL_HATCH, label='Control run'),
]
# ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.020),
#           ncol=3, fontsize=7, frameon=False)

plt.savefig(plots_dir + '/supp_lingress_' + CI_variable + '.png', dpi=300, bbox_inches='tight')
plt.close('all')
print(f"\nSaved {plots_dir}/supp_lingress_{CI_variable}.png")
