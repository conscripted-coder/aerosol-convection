"""Ridge plots of per-track spatial percentiles across the filtered cells.

Each ridge is the KDE of one percentile column of a variable family (e.g. ``zh_liq_p90``).
``plots_ridge_radar()`` tiles seven families into a 4-row x 2-col grid of subfigures,
(a)-(g); ``volrain`` has no percentiles, so its panel is one ridge.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sys
import yaml
import xarray as xr

mpl.rcParams['font.family'] = 'Arial'

# Percentile levels drawn, high -> low
PCTLS       = [ 'p98', 'p90', 'p75', 'p50']
PCTL_LABELS = {'p02': '2.5$^{th}$', 'p10': '10$^{th}$', 'p25': '25$^{th}$', 'p50': '50$^{th}$',
               'p75': '75$^{th}$', 'p90': '90$^{th}$', 'p98': '97.5$^{th}$'}

# Per-family display name and x-axis units
FAMILIES = {
    'zh_liq':  ('Radar Refl., $Z_{H,liq}$', r'$(\mathrm{dBZ})$'),
    'zh_frz':  ('Radar Refl., $Z_{H,ice}$', r'$(\mathrm{dBZ})$'),
    'zdr_liq': ('Diff. Refl., $Z_{DR}$', r'$(\mathrm{dB})$'),
    'Dm':      ('Mass-wtd. Mean Drop Diam., $D_m$', r'$(\mathrm{mm})$'),
    'kdp_liq': ('Spec. Diff. Phase, $K_{DP}$', r'$(\degree \cdot \mathrm{km}^{-1})$'),
    # 'rwc':     ('Rain Water Content', r'$(\mathrm{g} \cdot \mathrm{m}^{-3})$'),
    'rr':      ('Rain Rate', r'$(\mathrm{mm} \cdot \mathrm{hr}^{-1})$'),
    'volrain': ('Vol. Rain', r'$(\mathrm{m}^3 \cdot \mathrm{hr}^{-1})$'),
}

FILL_COLOR = '0.85'  # light gray fill for every ridge; matches plots_ridge_hid.py

# Panels, row-major
PANEL_FAMILIES = ['zh_liq', 'zh_frz', 'zdr_liq', 'Dm', 'kdp_liq', 'rr', 'volrain']
PANEL_LETTERS  = ['a', 'b', 'c', 'd', 'e', 'f', 'g']


def _xlimits(df, family):
    """Panel x-window: 2.5th pct of ``p50`` to 97.5th pct of ``p98`` (``volrain``: itself)."""
    if family == 'volrain':
        lo_col = hi_col = 'volrain'
    else:
        lo_col, hi_col = f'{family}_{PCTLS[-1]}', f'{family}_{PCTLS[0]}'
    return np.nanpercentile(df[lo_col], 2.5), np.nanpercentile(df[hi_col], 97.5)


def fmt_median(v):
    """Median label, precision scaled to magnitude."""
    a = abs(v)
    if a >= 1e4 or (0 < a < 1e-2):
        return f'{v:.1e}'
    if a >= 100:
        return f'{v:.0f}'
    if a >= 10:
        return f'{v:.1f}'
    return f'{v:.2f}'


def _ridge_data(df, family, pctl):
    """Across-track values for one rung; ``volrain`` (no percentiles) fills only ``p50``."""
    if family == 'volrain':
        return df['volrain'].dropna() if pctl == 'p50' else None
    return df[f'{family}_{pctl}'].dropna()


def draw_ridge(ax, data, color, xlo, xhi, bw_adjust, zorder, outline_color='black', fill_color=None,
               outline_lw=1.1):
    """Draw one ridge (filled KDE + outline) on a transparent axis; also used by plots_ridge_hid.

    Empty or None ``data`` draws no curve. ``fill_color`` defaults to ``color``.
    """
    if fill_color is None:
        fill_color = color
    if data is not None and len(data):
        sns.kdeplot(x=data, ax=ax, clip_on=False, color=outline_color, lw=outline_lw,
                    clip=(xlo, xhi), bw_adjust=bw_adjust)
        sns.kdeplot(x=data, ax=ax, clip_on=False, fill=True, alpha=0.85, linewidth=0,
                    color=fill_color, clip=(xlo, xhi), bw_adjust=bw_adjust)
    ax.patch.set_alpha(0)  # so stacked ridges show through
    ax.set(yticks=[], ylabel='', xlim=(xlo, xhi))
    ax.set_zorder(zorder)  # higher draws in front
    for side in ('left', 'right', 'top', 'bottom'):
        ax.spines[side].set_visible(False)


def _panel(subfig, df, family, letter):
    """One panel into ``subfig``: ridges from ``p50`` (top) to ``p98`` (bottom, in front),
    with percentile and median labels."""
    display_name, units = FAMILIES[family]
    xlo, xhi = _xlimits(df, family)
    subfig.text(0.03, 0.9, f'({letter})', fontweight='normal', ha='left', va='top')

    # volrain has no percentiles, so it gets just the median rung
    pctls = ['p50'] if family == 'volrain' else PCTLS

    axes = subfig.subplots(len(pctls), 1, sharex=True,
                           gridspec_kw={'hspace': -0.4}, squeeze=False)[:, 0]

    # "median" column header only on the reflectivity panels (a, b)
    if family in ('zh_liq', 'zh_frz'):
        axes[0].text(1.03, .4, 'median', fontsize=6, fontweight='normal',
                     ha='left', va='bottom', color='black', transform=axes[0].transAxes)
    for zo, (ax, pctl) in enumerate(zip(axes, pctls[::-1])):
        data = _ridge_data(df, family, pctl)
        draw_ridge(ax, data, 'black', xlo, xhi, 1, zorder=zo, fill_color=FILL_COLOR)
        if data is not None and len(data):
            # Median marker rises only to the KDE curve, not the full rung height
            median           = np.nanmedian(data)
            curve_x, curve_y = ax.lines[0].get_data()
            median_height    = np.interp(median, curve_x, curve_y)
            ax.vlines(median, 0, median_height, color='black',
                      linewidth=.6, linestyle='-', clip_on=False)
        if family != 'volrain':
            ax.text(-0.03, 0.1, PCTL_LABELS[pctl], color='black',
                    fontweight='normal', fontsize=9,
                    ha='right', va='center', transform=ax.transAxes)
        if data is not None and len(data):
            # Match volrain's 1e6 x-axis offset (e.g. 0.11, not 1.1e5)
            scale = 1e6 if family == 'volrain' else 1.0
            ax.text(1.03, 0.1, fmt_median(np.nanmedian(data) / scale), color='black',
                    fontweight='normal', fontsize=9,
                    ha='left', va='center', transform=ax.transAxes)
        if ax is not axes[-1]:
            ax.tick_params(labelbottom=False, bottom=False)
    axes[-1].set_xlabel(f'{display_name} {units}',fontsize=11)
    # axes[-1].tick_params(axis='x', labelsize=9)
    axes[-1].tick_params(axis='x', labelsize=9, bottom=True, length=3, width=.6)
    axes[-1].spines['bottom'].set_visible(True)
    axes[-1].spines['bottom'].set_color('black')
    axes[-1].spines['bottom'].set_linewidth(.6)


def plots_ridge_radar(plots_dir, df, families=PANEL_FAMILIES, letters=PANEL_LETTERS):
    """Render and save the 4-row x 2-col figure (``ridge_radar_*.png``). ``df`` is the
    filtered cells, one row per track; ``families`` and ``letters`` are aligned, row-major."""
    sns.set_theme(style='white', rc={'axes.facecolor': (0, 0, 0, 0),
                                     'font.family': 'Arial'})

    fig     = plt.figure(figsize=(6, 6))
    subfigs = fig.subfigures(4, 2, wspace=0.08, hspace=0.2)
    for i, (family, letter) in enumerate(zip(families, letters)):
        row, col = divmod(i, 2)
        _panel(subfigs[row, col], df, family, letter=letter)

    fig.savefig(plots_dir + '/ridge_radar_' + CI_variable + mask_sfx + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
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

    df_updr = xr.open_dataset(data_dir + '/df_updr_' + CI_variable + '.nc').to_dataframe()
    good    = xr.open_dataset(data_dir + '/the_chosen_few_' + CI_variable + '.nc')[mask_var].values
    df_updr = df_updr[good]
    plots_ridge_radar(plots_dir, df_updr)
    plt.close('all')
