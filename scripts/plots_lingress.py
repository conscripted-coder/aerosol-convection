"""Standardized regression coefficients, univariate vs multivariate, as a grid of bars.

Responses run down in bands, predictors across. Each cell stacks RED (upper, one-predictor
model) over BLUE (lower, partial coefficient from the full model); both use
fit_bootstrap_regression with identical options, hence identical splits. Bar length = median
over splits; deep colour = 2.5-97.5 split range excludes zero. The right-edge R^2 is the
multivariate test R^2. Fits are cached per CI variable and mask.
"""


import sys

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import os
import yaml
from lib_regression import fit_bootstrap_regression, DEFAULT_PERCENTILES
mpl.rcParams['font.family'] = 'Arial'

config_file = sys.argv[1]
stream      = open(config_file, 'r')
config      = yaml.full_load(stream)

celltracking_dir = config['celltracking_dir']
data_dir         = celltracking_dir + '/data'
plots_dir        = config['plots_dir']
CI_variable      = config['CI_variable']
# Non-default masks also suffix the figure name, leaving the manuscript figure alone
mask     = config['mask']
mask_var = 'good_indices' if mask == 'default' else 'good_indices_' + mask
mask_sfx = '' if mask == 'default' else '_' + mask

# Coefficient scale shared by both bar types. Not clipped on purpose: overflow into the
# next column is the cue to raise BAR_SCALE.
BAR_SCALE     = 0.7
BAR_HEIGHT    = 0.7  # thickness of one bar, y data units
COL_HALFWIDTH = 0.5  # half-width available to a bar within its column

# Edge-to-edge gaps, in BAR_HEIGHTs. A pair must read as one object, so keep the
# within-pair gap well below the others.
GAP_WITHIN_PAIR       = 0     # univariate bar <-> its multivariate bar
GAP_BETWEEN_RESPONSES = 0.70  # one response's pair <-> the next
GAP_BETWEEN_BANDS     = 1.90  # last pair of a band <-> first pair of the next

# Height = total_extent * FIG_INCHES_PER_Y, so bar thickness is fixed however many rows
FIG_WIDTH        = 6
FIG_INCHES_PER_Y = 0.117

HEADER_GAP = 1.5  # top of first bar to column headers, y data units

# Dotted grey lines at -/+ GRID_VALUE in each column (no x ticks); must be <= BAR_SCALE
GRID_VALUE      = 0.3
GRID_LABEL_DROP = .5  # gridline labels' distance below the last bar, y data units

# Right edge of the response labels; above -0.5 they collide with the CCN bars
RESPONSE_LABEL_X = -0.65

# Space right of the last column for the R^2 text; much below ~0.45 clips it
RIGHT_PAD = 0.5

# A faint rule separates bands; list the last response of a band to make the rule below it
# heavy black
MAJOR_RULE_AFTER = {
    # 'rain_area_sampled',           # end of the percentile + bulk-rain section
    'cloud_top_temperature_min',   # end of the cell-scale section, start of the HIDs
}
RULE_MINOR = ('0.85', 0.6)  # (color, linewidth)
RULE_MAJOR = ('black', 0.9)

# Deep = 95% split range excludes 0, pale = includes 0. Both pales are equally light.
COLOR_SIG        = 'deepskyblue'
COLOR_NONSIG     = '#d9effb'
COLOR_UNI_SIG    = "#e63852"
COLOR_UNI_NONSIG = '#fbd9de'

# Response groups, top to bottom
# PERCENTILES = ['p02', 'p10', 'p25', 'p50', 'p75', 'p90', 'p98']
# PERCENTILES = ['p50', 'p75', 'p90', 'p98']
PERCENTILES = ['p50', 'p98']

# Labels for the within-cell percentiles (key 'p98' is the 97.5th)
PERCENTILE_LABELS = {'p02': '2.5', 'p10': '10', 'p25': '25',
                     'p50': '50', 'p75': '75', 'p90': '90', 'p98': '97.5'}
PCTL_LABEL_X = - 0.28  # just left of the first column's zero line

percentile_groups = [
    (r'Z$_{\mathrm{H,liq}}$', 'zh_liq'),
    (r'Z$_{\mathrm{H,ice}}$', 'zh_frz'),
    (r'Z$_{\mathrm{DR}}$',    'zdr_liq'),
    (r'D$_{\mathrm{m}}$',     'Dm'),
    (r'K$_{\mathrm{DP}}$',    'kdp_liq'),
    ('Rain Rate',             'rr'),
    # ('Rain Water\nContent','rwc'),
]

# Non-percentile responses; each inner list is one band, each row labelled individually
single_groups = [
    [   # bulk rain amount vs. footprint
        ('Volumetric Rain',             'volrain'),
        ('Precipitation Area',          'rain_area_sampled'),
    ],
    [   # cell scale
        ('10 dBZ ETH',                  'max_echotop_10dbz'),
        (r'Cloud Top T$_{\mathrm{B}}$', 'cloud_top_temperature_min'),
    ],
    # [   # liquid HID categories
    #     ('Drizzle',             'drizzle'),
    #     ('Rain',                'rain'),
    #     ('Wet Snow',            'wet_snow'),
    # ],
    [   # rimed ice: composite of the three categories commented out below
        # ('Graupel-low',         'low_density_graupel'),
        # ('Graupel-high',        'high_density_graupel_frz'),
        # ('Hail',                'hail_frz'),
        ('**Rimed Ice**',           'rime_frz'),
    ],
    [   # melting rimed ice: composite of the three categories commented out below
        # ('Big Drops',           'big_drops'),
        # ('Graupel-high',        'high_density_graupel_liq'),
        # ('Hail',                'hail_liq'),
        ('**Melting Rimed Ice**',   'rime_liq'),
    ],
    # [   # unrimed ice: composite of the three categories below
    #     # ('Aggregates',          'aggregates'),
    #     # ('Ice Crystals',        'ice_crystals'),
    #     # ('Vertical Ice',        'vertical_ice'),
    #     ('**Non-Rimed Ice**',       'snow'),
    # ],
]


# Percentile bands get one centred label; single-variable bands label each row
bands = []
for label, base in percentile_groups:
    bands.append({'label': label, 'slots': [f'{base}_{p}' for p in PERCENTILES]})
for group in single_groups:
    slot_labels = []
    slot_italics = []
    for lab, _ in group:
        # **label** -> italics, marking composite rows
        is_italic   = lab.startswith('**') and lab.endswith('**')
        clean_label = lab.strip('*')
        slot_labels.append(clean_label)
        slot_italics.append(is_italic)
    bands.append({'label': None,
                  'slot_labels': slot_labels,
                  'slot_italics': slot_italics,
                  'slots': [col for _, col in group]})

pred    = [c for b in bands for c in b['slots'] if c is not None]  # responses, in order
col2idx = {c: i for i, c in enumerate(pred)}

lst        = ['ccn',  'CAPE_sfc','pw_mwr', 'rh_600mb', 'shear_mag_bulk_0to9km']
lst_labels = ['CCN', 'CAPE', 'PW', '600-hPa RH', '0-9 km BWD']

# Cache of the slow fits, one file per CI variable and mask. Fit options are not checked:
# pass --refresh after changing the responses, predictors, or lib_regression.
CACHE_PATH = data_dir + '/cache_lingress_' + CI_variable + mask_sfx + '.nc'

if '--refresh' in sys.argv or not os.path.exists(CACHE_PATH):
    df_updr = xr.open_dataset(data_dir + '/df_updr_' + CI_variable + '.nc').to_dataframe()
    good    = xr.open_dataset(data_dir + '/the_chosen_few_' + CI_variable + '.nc')[mask_var].values
    df_updr = df_updr[good]

    # Multivariate: all predictors in one model
    ds   = fit_bootstrap_regression(df_updr, lst, pred)
    Coef = ds.coefficient.transpose('percentile', 'label', 'predictor').values
    R2   = ds.r_squared.transpose('percentile', 'label').values

    # Univariate: one model per predictor, laid out like Coef (percentile, response, predictor)
    CoefUni = np.full_like(Coef, np.nan)
    for j, predictor in enumerate(lst):
        ds_uni = fit_bootstrap_regression(df_updr, [predictor], pred)
        CoefUni[:, :, j] = ds_uni.coefficient.transpose(
            'percentile', 'label', 'predictor').values[:, :, 0]

    dims3 = ('percentile', 'response', 'predictor')
    ds_out = xr.Dataset(
        {
            'coef_multi': (dims3, Coef),
            'coef_uni':   (dims3, CoefUni),
            'r_squared':  (('percentile', 'response'), R2),
        },
        coords={'percentile': list(DEFAULT_PERCENTILES),
                'response': pred,
                'predictor': lst},
    )
    ds_out.coef_multi.attrs['long_name'] = ('split-ensemble percentiles of the standardized '
                                            'coefficient, five-predictor model')
    ds_out.coef_uni.attrs['long_name']   = ('split-ensemble percentiles of the standardized '
                                            'coefficient, one-predictor models')
    ds_out.to_netcdf(CACHE_PATH)
    print(f"  fit {CI_variable:>10s}: n = {len(df_updr)} -> {CACHE_PATH}")
else:
    with xr.open_dataset(CACHE_PATH) as ds:
        ds = ds.load()
    Coef    = ds.coef_multi.transpose('percentile', 'response', 'predictor').values
    CoefUni = ds.coef_uni.transpose('percentile', 'response', 'predictor').values
    R2      = ds.r_squared.transpose('percentile', 'response').values
    print(f"  read {CI_variable:>10s} from {CACHE_PATH}")


n_cols = len(lst)

# Vertical layout in y data units; the y axis is inverted, so "above" means smaller y
gap_pair = GAP_WITHIN_PAIR * BAR_HEIGHT
gap_resp = GAP_BETWEEN_RESPONSES * BAR_HEIGHT
gap_band = GAP_BETWEEN_BANDS * BAR_HEIGHT

# A slot is one response: two stacked bars plus the gap between them
SLOT_EXTENT = 2 * BAR_HEIGHT + gap_pair
SLOT_PITCH  = SLOT_EXTENT + gap_resp       # centre-to-centre, consecutive responses
PAIR_OFFSET = (BAR_HEIGHT + gap_pair) / 2  # slot centre to either bar centre

cursor = 0
for band in bands:
    band['start']  = cursor  # y of the band's top edge
    band['extent'] = len(band['slots']) * SLOT_PITCH - gap_resp
    cursor += band['extent'] + gap_band
total_extent = cursor - gap_band

fig, ax = plt.subplots(figsize=(FIG_WIDTH, total_extent * FIG_INCHES_PER_Y))

# Per-column zero lines plus dotted grey lines at -/+ GRID_VALUE (converted to x units)
grid_offset = (GRID_VALUE / BAR_SCALE) * COL_HALFWIDTH
for j in range(n_cols):
    ax.axvline(x=j, color='black', linestyle='--', alpha=0.4, linewidth=0.4, zorder=1)
    for sign in (-1, 1):
        ax.axvline(x=j + sign * grid_offset, color='0.6', linestyle=':',
                   linewidth=0.5, zorder=1)

for j, label in enumerate(lst_labels):
    ax.text(j, -HEADER_GAP, label, ha='center', va='bottom', fontsize=7, fontweight='bold')

# Bars, group labels, band separators, and per-row R^2
for b, band in enumerate(bands):
    band_start = band['start']
    n_slots    = len(band['slots'])
    if band['label'] is not None:
        ax.text(RESPONSE_LABEL_X, band_start + band['extent'] / 2, band['label'],
                ha='right', va='center', fontsize=7)
    else:
        for s, slot_label in enumerate(band['slot_labels']):
            is_italic = band.get('slot_italics', [False] * len(band['slot_labels']))[s]
            ax.text(RESPONSE_LABEL_X, band_start + SLOT_EXTENT / 2 + s * SLOT_PITCH,
                    slot_label,
                    ha='right', va='center', fontsize=7,
                    style='italic' if is_italic else 'normal')

    # Separator in the gap above each band but the first
    if b > 0:
        prev_last = bands[b - 1]['slots'][-1]
        color, lw = (RULE_MAJOR if prev_last in MAJOR_RULE_AFTER else RULE_MINOR)
        ax.axhline(y=band_start - gap_band / 2, color=color, linewidth=lw, zorder=0)

    for s, col in enumerate(band['slots']):
        if col is None:
            continue
        y  = band_start + SLOT_EXTENT / 2 + s * SLOT_PITCH  # slot centre
        iy = col2idx[col]

        # Percentile label on percentile-band rows
        if band['label'] is not None:
            ax.text(PCTL_LABEL_X, y,
                    rf'{PERCENTILE_LABELS[PERCENTILES[s]]}$^{{\mathrm{{th}}}}$',
                    ha='right', va='center', fontsize=5, color='0.3')

        # Multivariate R^2 only, between the response's two bars
        ax.text(n_cols - 1 + COL_HALFWIDTH + 0.15, y,
                f"{R2[2, iy]:.2f}", ha='center', va='center', fontsize=6, color='0.3')

        for j in range(n_cols):
            # Univariate: upper bar
            uni_med           = CoefUni[2, iy, j]
            uni_excludes_zero = ((CoefUni[0, iy, j] > 0 and CoefUni[4, iy, j] > 0) or
                                 (CoefUni[0, iy, j] < 0 and CoefUni[4, iy, j] < 0))
            uni_color         = COLOR_UNI_SIG if uni_excludes_zero else COLOR_UNI_NONSIG
            uni_width         = (uni_med / BAR_SCALE) * COL_HALFWIDTH
            ax.barh(y - PAIR_OFFSET, uni_width, left=j, height=BAR_HEIGHT,
                    color=uni_color, edgecolor='black', linewidth=0.4, zorder=3)

            # Multivariate partial coefficient: lower bar
            med           = Coef[2, iy, j]
            excludes_zero = ((Coef[0, iy, j] > 0 and Coef[4, iy, j] > 0) or
                             (Coef[0, iy, j] < 0 and Coef[4, iy, j] < 0))
            color         = COLOR_SIG if excludes_zero else COLOR_NONSIG
            width         = (med / BAR_SCALE) * COL_HALFWIDTH
            ax.barh(y + PAIR_OFFSET, width, left=j, height=BAR_HEIGHT,
                    color=color, edgecolor='black', linewidth=0.4, zorder=3)

ax.set_xlim(-2.0, n_cols - 1 + COL_HALFWIDTH + RIGHT_PAD)
ax.set_ylim(-BAR_HEIGHT / 2, total_extent + BAR_HEIGHT / 2)
ax.invert_yaxis()  # first band on top
ax.set_yticks([])
ax.set_xticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# R^2 column header
ax.text(n_cols - 1 + COL_HALFWIDTH +0.15, -HEADER_GAP, r'R$^{\mathrm{2}}$',
        ha='center', va='bottom', fontsize=7, fontweight='bold')

# Gridline labels under the first column only, past ylim (clip_on=False; bbox_inches="tight"
# grows the canvas). Units: response SDs per predictor SD.
y_label = total_extent + BAR_HEIGHT / 2 + GRID_LABEL_DROP
for sign in (-1, 1):
    ax.text(sign * grid_offset, y_label, rf'{sign * GRID_VALUE:+g}$\sigma$',
            ha='center', va='top', fontsize=6, color='0.45', clip_on=False)

# Significant colours only; red first to match the bar stacking
legend_elements = [
    Patch(facecolor=COLOR_UNI_SIG, edgecolor='black', linewidth=0.4,
          label='Univariate'),
    Patch(facecolor=COLOR_SIG, edgecolor='black', linewidth=0.4,
          label='Multivariate'),
]
ax.legend(handles=legend_elements, loc='upper center',
          bbox_to_anchor=(0.7, -0.01), ncol=2, fontsize=8, frameon=False)

plt.savefig(plots_dir + '/lingress_' + CI_variable + mask_sfx + '.png', dpi=300,
            bbox_inches="tight")
plt.close('all')
