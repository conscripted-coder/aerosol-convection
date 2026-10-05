"""Lower-triangle Spearman correlogram (upper triangle and diagonal masked)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from scipy.stats import pearsonr,spearmanr
import sys
import yaml
import xarray as xr
import lib_style

config_file = sys.argv[1]
stream      = open(config_file, 'r')
config      = yaml.full_load(stream)

celltracking_dir = config['celltracking_dir']
data_dir         = celltracking_dir + '/data'
plots_dir        = config['plots_dir']
CI_variable      = config['CI_variable']

mask     = config['mask']
mask_var = 'good_indices' if mask == 'default' else 'good_indices_' + mask
mask_sfx = '' if mask == 'default' else '_' + mask

df_updr = xr.open_dataset(data_dir + '/df_updr_' + CI_variable + '.nc').to_dataframe()
good    = xr.open_dataset(data_dir + '/the_chosen_few_' + CI_variable + '.nc')[mask_var].values
df_updr = df_updr[good]

# Axis order, both axes. To drop a variable, comment it out here, in ``labels``, and fix ``spans``.
vars_all = [
    "ccn",
    # Radar
    "zh_liq_p50",
    "zh_liq_p98",
    "zh_frz_p50",
    "zh_frz_p98",
    "zdr_liq_p50",
    "zdr_liq_p98",
    "Dm_p50",
    "Dm_p98",
    "kdp_liq_p50",
    "kdp_liq_p98",
    "rr_p50",
    "rr_p98",
    "volrain",
    "rain_area_sampled",
    "rime_frz",
    "rime_liq",
    # Thermodynamics
    "rh_500mb",
    "rh_600mb",
    "qv_sfc",
    "pw_mwr",
    "CAPE_sfc",
    "CIN_sfc",
    "EL_height_sfc",
    "LCL_height_sfc",
    "LFC_height_sfc",
    # Winds
    "shear_mag_bulk_0to1km",
    "shear_mag_bulk_0to3km",
    "shear_mag_bulk_0to6km",
    "shear_mag_bulk_0to9km",
    "U_met",
    "V_met",
    "U_700mb",
    "V_700mb",
]

# Index-aligned with ``vars_all``
labels = [
    'CCN',
    # Radar
    '$\\mathrm{Z_{H,liq,50^{th}}}$',
    '$\\mathrm{Z_{H,liq,97.5^{th}}}$',
    '$\\mathrm{Z_{H,ice,50^{th}}}$',
    '$\\mathrm{Z_{H,ice,97.5^{th}}}$',
    '$\\mathrm{Z_{DR,50^{th}}}$',
    '$\\mathrm{Z_{DR,97.5^{th}}}$',
    '$\\mathrm{D_{m,50^{th}}}$',
    '$\\mathrm{D_{m,97.5^{th}}}$',
    '$\\mathrm{K_{DP,50^{th}}}$',
    '$\\mathrm{K_{DP,97.5^{th}}}$',
    'Rain Rate $_{50^{th}}$',
    'Rain Rate $_{97.5^{th}}$',
    'Volumetric Rain',
    'Precip. Area',
    'Rimed Ice$_{frz}$',
    'Rimed Ice$_{liq}$',
    # Thermodynamics
    '500-hPa RH',
    '600-hPa RH',
    'Spc. Humidity',
    'PW (MWR)',
    'CAPE',
    'CIN',
    'EL Height',
    'LCL Height',
    'LFC Height',
    # Winds
    '0-1 km BWD',
    '0-3 km BWD',
    '0-6 km BWD',
    '0-9 km BWD',
    'Sfc. U',
    'Sfc. V',
    '700-hPa U',
    '700-hPa V',
]

assert len(vars_all) == len(labels), (
    f"vars_all ({len(vars_all)}) and labels ({len(labels)}) are out of sync — "
    "did you comment out a variable but not its label?"
)

# Group brackets: (name, start, stop) as [start, stop) into ``vars_all``; CCN (0) is unlabelled
spans = [
    ('Radar', 1, 17),
    ('Thermodynamics', 17, 26),
    ('Wind', 26, 34),
]

# Symmetric Spearman matrix, one call per pair. No NaN handling: the mask already excludes NaN/Inf.
df_sel      = df_updr[vars_all]
n           = len(vars_all)
corr_matrix = pd.DataFrame(np.eye(n), index=vars_all, columns=vars_all)
pval_matrix = pd.DataFrame(np.zeros((n, n)), index=vars_all, columns=vars_all)

for i in range(n):
    for j in range(i + 1, n):
        res = spearmanr(df_sel.iloc[:, i], df_sel.iloc[:, j])
        corr_matrix.iloc[i, j] = corr_matrix.iloc[j, i] = res.statistic
        pval_matrix.iloc[i, j] = pval_matrix.iloc[j, i] = res.pvalue

# k=0 also masks the diagonal
mask = np.triu(np.ones((n, n), dtype=bool), k=0)

plt.figure(figsize=(8, 8))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
ax   = sns.heatmap(
    corr_matrix, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1, square=True,
    linewidths=.5, cbar=False, annot=False,
)

# Black if p < 0.05, gray otherwise
for i in range(n):
    for j in range(i):
        color_s = 'black' if pval_matrix.iloc[i, j] < 0.05 else 'gray'
        # .76 rather than 0.76, and a true minus sign so signs align
        text = (f"{corr_matrix.iloc[i, j]:.2f}"
                .replace('0.', '.', 1)
                .replace('-', '\N{MINUS SIGN}'))
        ax.text(j + 0.5, i + 0.5, text,
                ha='center', va='center', fontsize=7, color=color_s)

# Skip the fully masked first row and last column
ax.set_xticks(np.arange(n - 1) + 0.5)
ax.set_yticks(np.arange(1, n) + 0.5)
ax.set_xticklabels(labels[:-1], rotation=90, ha='center', fontsize=8)
ax.set_yticklabels(labels[1:], rotation=0, va='center', fontsize=8)

# Bold the regression predictors
HIGHLIGHT = {'ccn', 'rh_600mb', 'pw_mwr', 'CAPE_sfc', 'shear_mag_bulk_0to9km'}
for var, tick in zip(vars_all[:-1], ax.get_xticklabels()):
    if var in HIGHLIGHT:
        tick.set_fontweight('bold')
for var, tick in zip(vars_all[1:], ax.get_yticklabels()):
    if var in HIGHLIGHT:
        tick.set_fontweight('bold')

# Group brackets on both axes, placed just past the measured tick-label extents (known
# only after a draw). Blended coords: data units along the axis, axes fraction across.
fig = ax.figure
fig.canvas.draw()
inv = ax.transAxes.inverted()
y_ticks = inv.transform((0, min(t.get_window_extent().y0
                                for t in ax.get_xticklabels())))[1]
x_ticks = inv.transform((min(t.get_window_extent().x0
                             for t in ax.get_yticklabels()), 0))[0]
TICK   = 0.012            # end ticks, turned back towards the cells
x_rule = y_ticks - 0.02   # bottom rule, just under the tick labels
x_text = x_rule - 0.015   # name just below its rule
y_rule = x_ticks - 0.02   # left rule, likewise
y_text = y_rule - 0.015
trans_x = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
trans_y = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)

for name, start, stop in spans:
    # Clipped like the tick labels and inset 0.1 cell so neighbours don't touch
    lo, hi = start, min(stop, n - 1)  # columns
    if lo < hi:
        a, b = lo + 0.1, hi - 0.1
        ax.plot([a, a, b, b], [x_rule + TICK, x_rule, x_rule, x_rule + TICK],
                transform=trans_x, color='0.35', lw=0.8, clip_on=False)
        ax.text((lo + hi) / 2, x_text, name, transform=trans_x,
                ha='center', va='top', fontsize=8, color='0.2', clip_on=False)

    lo, hi = max(start, 1), stop  # rows
    if lo < hi:
        a, b = lo + 0.1, hi - 0.1
        ax.plot([y_rule + TICK, y_rule, y_rule, y_rule + TICK], [a, a, b, b],
                transform=trans_y, color='0.35', lw=0.8, clip_on=False)
        ax.text(y_text, (lo + hi) / 2, name, transform=trans_y, rotation=90,
                ha='right', va='center', fontsize=8, color='0.2', clip_on=False)

plt.tight_layout()
plt.savefig(plots_dir + '/corr_full_' + CI_variable + mask_sfx + '.png', dpi=300, bbox_inches="tight")
plt.close('all')
