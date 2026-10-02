"""Ridge plots of HID (hydrometeor identification) fractions across the filtered cells.

Each ridge is one per-track HID fraction, in percent on a fixed [0, 100] axis. The 15
categories are split into 4 panels: rimed ice, unrimed ice, liquid, and melting rimed ice.
Left labels name the category; right labels give its median.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sys
import yaml
import xarray as xr
from plots_ridge_radar import draw_ridge, fmt_median

mpl.rcParams['font.family'] = 'Arial'

LABEL_OF = {
        'rime_liq':                 'Rimed Ice $_{liq}$',
        'hail_liq':                 'Hail $_{liq}$',
        'high_density_graupel_liq': 'High Density\nGraupel $_{liq}$',
        'big_drops':                'Big Drops $_{liq}$',
        'rain':                     'Rain $_{liq}$',
        'drizzle':                  'Drizzle $_{liq}$',
        'wet_snow':                 'Wet Snow $_{liq}$',
        'snow':                     'Non-rimed\nIce $_{ice}$',
        'vertical_ice':             'Vertical Ice $_{ice}$',
        'ice_crystals':             'Ice Crystals $_{ice}$',
        'aggregates':               'Aggregates $_{ice}$',
        'rime_frz':                 'Rimed Ice $_{ice}$',
        'hail_frz':                 'Hail $_{ice}$',
        'high_density_graupel_frz': 'High Density\nGraupel $_{ice}$',
        'low_density_graupel':      'Low Density\nGraupel $_{ice}$',
        }

# 2x2 panels, row-major; each list is in bottom -> top ridge order
PANELS = [
        ('a', ['rime_frz', 'hail_frz', 'high_density_graupel_frz', 'low_density_graupel']),
        ('b', ['snow', 'vertical_ice', 'ice_crystals', 'aggregates']),
        ('c', ['rain', 'drizzle', 'wet_snow']),
        ('d', ['rime_liq', 'hail_liq', 'high_density_graupel_liq', 'big_drops']),
        ]

BW       = 1
XLO, XHI = 0.0, 100.0


FILL_COLOR = '0.85'  # light gray fill for every ridge

# Composite categories, labelled in bold
BOLD_VARS = {'rime_frz', 'rime_liq', 'snow'}


def _panel(subfig, df, group, letter):
    """One panel into ``subfig``: an overlapping ridge stack."""
    subfig.text(0, 0.85, f'({letter})', fontweight='normal', ha='left', va='top')

    axes = subfig.subplots(len(group), 1, sharex=True,
                           gridspec_kw={'hspace': -0.4}, squeeze=False)[:, 0]

    # "median" column header on the top panels only
    if letter in ('a', 'b'):
        axes[0].text(1.02, .3, 'median', fontsize=6, fontweight='normal',
                     ha='left', va='bottom', color='black', transform=axes[0].transAxes)

    # axes[0] = top (last var in group), axes[-1] = bottom (first var in group)
    for zo, (ax, var) in enumerate(zip(axes, group[::-1])):
        data = df[var].dropna() * 100  # fraction -> percent
        draw_ridge(ax, data, 'black', XLO, XHI, BW, zorder=zo, fill_color=FILL_COLOR,
                   outline_color='black', outline_lw=1.2)
        if len(data):
            # Median marker rises only to the KDE curve, not the full rung height
            median           = np.nanmedian(data)
            curve_x, curve_y = ax.lines[0].get_data()
            median_height    = np.interp(median, curve_x, curve_y)
            ax.vlines(median, 0, median_height, color='black',
                      linewidth=.8, linestyle='-', clip_on=False)
        ax.text(-0.06, 0, LABEL_OF[var], color='black',
                fontweight='bold' if var in BOLD_VARS else 'normal', fontsize=9,
                ha='right', va='bottom', transform=ax.transAxes)
        if len(data):
            ax.text(1.02, 0.1, fmt_median(np.nanmedian(data)), color='black',
                    fontweight='normal', fontsize=9,
                    ha='left', va='center', transform=ax.transAxes)
        if ax is not axes[-1]:
            ax.tick_params(labelbottom=False, bottom=False)
    axes[-1].set_xticks(np.linspace(XLO, XHI, 5))  # 0, 25, 50, 75, 100
    axes[-1].set_xlabel('Percent (%)', fontsize=11)
    axes[-1].tick_params(axis='x', labelsize=9, bottom=True, length=3, width=.8)
    axes[-1].spines['bottom'].set_visible(True)
    axes[-1].spines['bottom'].set_color('black')
    axes[-1].spines['bottom'].set_linewidth(.8)


def plots_ridge_hid(plots_dir, df):
    """Render and save the 4-panel HID ridge figure."""
    sns.set_theme(style='white', rc={'axes.facecolor': (0, 0, 0, 0),
                                     'font.family': 'Arial'})

    fig     = plt.figure(figsize=(6, 4.5))
    subfigs = fig.subfigures(2, 2, wspace=0.3, hspace=0.05)
    for (letter, group), subfig in zip(PANELS, subfigs.flat):
        _panel(subfig, df, group, letter=letter)

    fig.savefig(plots_dir + '/ridge_hid_' + CI_variable + mask_sfx + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
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

    df   = xr.open_dataset(data_dir + '/df_updr_' + CI_variable + '.nc').to_dataframe()
    good = xr.open_dataset(data_dir + '/the_chosen_few_' + CI_variable + '.nc')[mask_var].values
    df   = df[good]
    plots_ridge_hid(plots_dir, df)
    plt.close('all')
