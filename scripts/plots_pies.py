"""Pie charts of each predictor's share of the total effect (median standardized coefficients,
five-predictor model) for six hand-picked responses. Each row uses one CI definition, labelled
beside it; the printed caption line names it per panel.
"""

import os
import sys

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox
import yaml
from lib_regression import fit_bootstrap_regression
import lib_style

config_file = sys.argv[1]
stream      = open(config_file, 'r')
config      = yaml.full_load(stream)

celltracking_dir = config['celltracking_dir']
data_dir         = celltracking_dir + '/data'
plots_dir        = config['plots_dir']
# Non-default masks also suffix the figure name, leaving the manuscript figure alone
mask     = config['mask']
mask_var = 'good_indices' if mask == 'default' else 'good_indices_' + mask
mask_sfx = '' if mask == 'default' else '_' + mask

# Typeset $...$ in Arial too, not mathtext's DejaVu fallback
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm']      = 'Arial'
mpl.rcParams['mathtext.it']      = 'Arial:italic'
mpl.rcParams['mathtext.bf']      = 'Arial:bold'

HATCH_NEG        = '//////'    # negative coefficient
HATCH_NONSIG     = '.....'     # split range spans zero
HATCH_LINEWIDTH  = 0.2         # only settable through the global rcParam
START_ANGLE      = 90          # smallest wedge starts at 12 o'clock, running clockwise
MIN_LABEL_SHARE  = 5.0         # no % label below this share, so slivers don't overprint
N_COLS, N_ROWS   = 3, 2        # panel grid
FIG_SIZE         = (6, 4)
PANEL_LABEL_SIZE = 9
TITLE_PAD        = 1           # shared by titles and panel letters
WSPACE, HSPACE   = 0.1, 0.2    # main lever on title/letter collisions between panels
# Figure-fraction padding around each row's pies and titles, then the gaps to the row's
# CI label and from the label to its legend
ROW_BOX_PAD    = 0.02
ROW_LABEL_GAP  = 0.015
ROW_LEGEND_GAP = 0.04
ROW_LABEL = {
    'eth_10dbz': 'max 10-dBZ ETH',
    'rainarea':  'max precip. area',
}

mpl.rcParams['hatch.linewidth'] = HATCH_LINEWIDTH

lst        = ['ccn', 'CAPE_sfc', 'pw_mwr', 'rh_600mb', 'shear_mag_bulk_0to9km']
lst_labels = ['CCN', 'CAPE', 'PW', 'RH', 'BWD']

# One pastel per predictor, fixed across panels (commented: earlier choices)
COLOR_PREDICTOR = {
    'ccn':                   '#fbb4ae',   # pastel red
    'CAPE_sfc':              '#b3cde3',   # pastel blue
    # 'ccn':                   '#8dd3c7',   # pastel teal
    # 'CAPE_sfc':              '#ffed6f',   # pastel yellow
    # 'ccn':                   '#fccde5',   # pastel pink
    # 'CAPE_sfc':              '#d9d9d9',   # pastel grey
    'pw_mwr':                '#ccebc5',   # pastel green
    'rh_600mb':              '#decbe4',   # pastel purple
    'shear_mag_bulk_0to9km': '#fed9a6',   # pastel orange
}

# Panels, row by row: (title, CI definition, response column). eth_10dbz is the primary CI
# definition, rainarea a supplementary one; each response uses whichever samples it better.
RESPONSES = [
    (r'Z$_{\mathrm{DR,97.5th}}$',       'eth_10dbz', 'zdr_liq_p98'),
    (r'D$_{\mathrm{m,97.5th}}$',        'eth_10dbz', 'Dm_p98'),
    (r'Z$_{\mathrm{H,liq,97.5th}}$',    'eth_10dbz', 'zh_liq_p98'),
    (r'Z$_{\mathrm{H,ice,97.5th}}$',    'rainarea',  'zh_frz_p98'),
    ('Rimed Ice',                       'rainarea',  'rime_frz'),
    # ('10-dBZ ETH',                      'rainarea',  'max_echotop_10dbz'),
    # ('High Dens. Graup.',               'rainarea',  'high_density_graupel_frz'),
    ('Volumetric Rain',                 'rainarea',  'volrain'),
]

# One cache per mask. Fit options are not checked: set refresh: true after changing the panels,
# predictors, or lib_regression.
CACHE_PATH = data_dir + '/cache_pies' + mask_sfx + '.nc'


def panel_key(ci_var, col):
    """Cache key for a panel: the response plus the CI definition it was sampled under."""
    return f'{ci_var}:{col}'


def fit_panels(ci_var, cols):
    """Multivariate fit of `cols` for one CI definition, as a (percentile, panel, predictor)
    DataArray keyed by panel_key."""
    df   = xr.open_dataset(data_dir + '/df_updr_' + ci_var + '.nc').to_dataframe()
    good = xr.open_dataset(data_dir + '/the_chosen_few_' + ci_var + '.nc')[mask_var].values
    df   = df[good]
    out  = fit_bootstrap_regression(df, lst, cols)
    print(f"  fit {ci_var:>10s}: n = {len(df)}, {len(cols)} response(s) "
          f"[{', '.join(cols)}]")
    coef = out.coefficient.transpose('percentile', 'label', 'predictor')
    return xr.DataArray(
        coef.values,
        dims=('percentile', 'panel', 'predictor'),
        coords={'percentile': coef.percentile.values,
                'panel':      [panel_key(ci_var, c) for c in cols],
                'predictor':  lst},
    )


if config['refresh'] or not os.path.exists(CACHE_PATH):
    # One fit per CI definition
    by_ci = {}
    for _, ci_var, col in RESPONSES:
        by_ci.setdefault(ci_var, []).append(col)

    coef_multi = xr.concat([fit_panels(ci_var, cols) for ci_var, cols in by_ci.items()],
                           dim='panel')
    ds_out     = xr.Dataset({'coef_multi': coef_multi})
    ds_out.coef_multi.attrs['long_name'] = ('split-ensemble percentiles of the standardized '
                                            'coefficient, five-predictor model')
    ds_out.to_netcdf(CACHE_PATH)
    print(f"  wrote {CACHE_PATH}")
else:
    with xr.open_dataset(CACHE_PATH) as ds:
        coef_multi = ds.load().coef_multi
    print(f"  read {CACHE_PATH}")

# Split-ensemble percentiles; indices read off the coord so any cached set lines up
pct        = list(coef_multi.percentile.values)
i_med      = pct.index(50)
i_lo, i_hi = 0, len(pct) - 1

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=FIG_SIZE)
axes      = axes.ravel()

for i_panel, (ax, (title, ci_var, col)) in enumerate(zip(axes, RESPONSES)):
    panel = coef_multi.sel(panel=panel_key(ci_var, col), predictor=lst).values
    med         = panel[i_med, :]
    lo, hi      = panel[i_lo, :], panel[i_hi, :]
    significant = (lo * hi) > 0  # range wholly on one side of zero

    # Share of total |effect|; the sign is shown by hatching
    share = 100.0 * np.abs(med) / np.abs(med).sum()
    order = np.argsort(share)  # ascending, from 12 o'clock

    sizes, colors, hatches, labels = [], [], [], []
    for j in order:
        sizes.append(share[j])
        colors.append(COLOR_PREDICTOR[lst[j]])
        if not significant[j]:
            # Dots override slashes and no % is printed: a null effect isn't quantified
            hatches.append(HATCH_NONSIG)
            labels.append('')
        else:
            hatches.append(HATCH_NEG if med[j] < 0 else None)
            labels.append(f'{share[j]:.0f}%'
                          if share[j] >= MIN_LABEL_SHARE else '')

    wedges, texts = ax.pie(sizes, labels=labels, colors=colors,
                           startangle=START_ANGLE,
                           counterclock=False,
                           wedgeprops=dict(edgecolor='black', linewidth=1),
                           textprops=dict(fontsize=8))
    # Move each label to the centre of the wedge's largest inscribed circle,
    # R / (1 + sin(half-angle)) along the bisector (R/2 past a half pie), centre-anchored
    for w, t in zip(wedges, texts):
        half = np.deg2rad(w.theta2 - w.theta1) / 2
        mid  = np.deg2rad(w.theta1 + w.theta2) / 2
        d    = w.r * (0.5 if half >= np.pi / 2 else 1 / (1 + np.sin(half)))
        t.set_position((w.center[0] + d * np.cos(mid), w.center[1] + d * np.sin(mid)))
        t.set_ha('center')
        t.set_va('center')
    # Per-wedge hatches (wedgeprops applies to all wedges alike)
    for w, h in zip(wedges, hatches):
        if h:
            w.set_hatch(h)

    ax.set_title(title, fontsize=10, pad=TITLE_PAD)
    # Panel letter as the left title, so it shares the title's baseline
    ax.set_title(f'({chr(ord("a") + i_panel)})', loc='left',
                 fontsize=PANEL_LABEL_SIZE, pad=TITLE_PAD)

# Separate colour (predictor) and pattern (coefficient) legends, so hatched swatches don't
# read as extra predictors
handles_predictor = [
    Patch(facecolor=COLOR_PREDICTOR[p], edgecolor='black', linewidth=0.6, label=lab)
    for p, lab in zip(lst, lst_labels)
]
handles_pattern = [
    Patch(facecolor='white', edgecolor='black', linewidth=0.6, hatch=HATCH_NEG,
          label='Negative'),
    Patch(facecolor='white', edgecolor='black', linewidth=0.6, hatch=HATCH_NONSIG,
          label='Statistically\nInsignificant'),
]

fig.subplots_adjust(wspace=WSPACE, hspace=HSPACE)

# Label each row with its CI definition (taken from its first panel), with a legend below:
# colours by the top row, patterns by the bottom. Row extents come from the wedges and
# titles, not the axes, and are only known after a draw.
fig.canvas.draw()
renderer    = fig.canvas.get_renderer()
to_fig      = fig.transFigure.inverted()
row_handles = [handles_predictor, handles_pattern]
row_extents, side_extents = [], []
for i_row in range(N_ROWS):
    row_axes = axes[i_row * N_COLS:(i_row + 1) * N_COLS]
    bb = Bbox.union([a.get_window_extent(renderer) for ax in row_axes
                     for a in [*ax.patches, *ax.texts, ax.title, ax._left_title]]
                    ).transformed(to_fig)
    x0, y0 = bb.x0 - ROW_BOX_PAD, bb.y0 - ROW_BOX_PAD
    w, h   = bb.width + 2 * ROW_BOX_PAD, bb.height + 2 * ROW_BOX_PAD
    row_extents.append((x0, y0, w, h))
    ci_var = RESPONSES[i_row * N_COLS][1]
    # Label on the row's title baseline
    t              = row_axes[0]._left_title
    title_baseline = to_fig.transform(t.get_transform().transform(t.get_position()))[1]
    label          = fig.text(x0 + w + ROW_LABEL_GAP + 0.08, title_baseline, ROW_LABEL[ci_var],
                              ha='center', va='baseline', fontsize=9, fontweight='bold',fontstyle='italic')
    # Legend hangs from the label's bottom edge; borderpad=0 aligns it with the label's left
    label_bb = label.get_window_extent(renderer).transformed(to_fig)
    leg      = fig.legend(handles=row_handles[i_row], loc='upper left', ncol=1, fontsize=7,
                          frameon=False, borderpad=0, borderaxespad=0,
                          bbox_to_anchor=(label_bb.x0, label_bb.y0 - ROW_LEGEND_GAP))
    side_extents += [label_bb, leg.get_window_extent(renderer).transformed(to_fig)]

# Rule midway between the rows, from the pies' left edge to the label column's right edge
y_bar   = (row_extents[0][1] + row_extents[1][1] + row_extents[1][3]) / 2
x_left  = min(x0 for x0, _, _, _ in row_extents)
x_right = max(b.x1 for b in side_extents)
fig.add_artist(Line2D([x_left, x_right], [y_bar, y_bar], transform=fig.transFigure,
                      color='black', linewidth=0.8))

plt.savefig(plots_dir + '/lingress_pies' + mask_sfx + '.png', dpi=300, bbox_inches='tight')
plt.close('all')
print(f"Wrote {plots_dir}/lingress_pies{mask_sfx}.png")

print('\nCaption line (panels are drawn from two different CI definitions):')
print('  ' + '; '.join(f'{col} ({ci_var})' for _, ci_var, col in RESPONSES))
