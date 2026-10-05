"""3x3 histograms of storm, environment, and CCN variables for the filtered cells.

Saves hist_env_{CI_variable}{mask_sfx}.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.transforms import offset_copy
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import sys
import yaml
import xarray as xr
from plots_ridge_radar import fmt_median

import lib_style

# Overrides below are keyed by panel letter
NBINS          = 15
NBINS_OVERRIDE = {'c': 16}
XLIM_OVERRIDE  = {'c': (0, 4)}  # otherwise the data-driven window from _xlimits

FILL_ALPHA = 0.35

# Inline name/median legend: first line's position (axes fraction) and per-line drop
LEGEND_X  = 0.04
LEGEND_Y  = 0.97
LEGEND_DY = 0.09

HEADROOM = 0.16  # space above the peak count for each legend line, fraction of the peak

# ~6 ticks at 1/2/5 multiples; the default leaves only 2-3 on these narrow panels
XTICK_NBINS = 6
XTICK_STEPS = [1, 2, 5, 10]

# Fewer ticks for panel (i, CCN), whose 4-digit labels collide
XTICK_NBINS_OVERRIDE = {'i': 3}

# Inline legend names
NAMES = {
    'max_echotop_10dbz':     '10-dBZ Max Echo Top Hgt',
    'cloud_top_height_max':  'GOES16 Cld Top Hgt',
    'core_area_sampled':     'Sampled Core Area',
    'rain_area_sampled':     '2-km AGL Precip Area',
    'track_duration_full':   'Track Duration',
    'sample_index':          'Time of Max ETH',
    'CAPE_sfc':              'CAPE',
    'EL_height_sfc':         'EL Height',
    'LCL_height_sfc':        'LCL Height',
    'LFC_height_sfc':        'LFC Height',
    'rh_500mb':              '500-hPa RH',
    'rh_600mb':              '600-hPa RH',
    'rh_700mb':              '700-hPa RH',
    'shear_mag_bulk_0to3km': '0-3 km AGL Bulk Wind Diff',
    'shear_mag_bulk_0to6km': '0-6 km AGL Bulk Wind Diff',
    'shear_mag_bulk_0to9km': '0-9 km AGL Bulk Wind Diff',
    'ccn':                   'CCN',
}

M_TO_KM        = {'EL_height_sfc', 'LCL_height_sfc', 'LFC_height_sfc'}  # panels e, f
TIMESTEP_TO_HR = {'track_duration_full', 'sample_index'}                # panel c (15-min steps)

# (variables, unit) per panel, row-major
PANELS = [
    (['max_echotop_10dbz', 'cloud_top_height_max'],      'km AMSL'),
    (['core_area_sampled', 'rain_area_sampled'],         r'$\mathrm{km}^2$'),
    (['track_duration_full', 'sample_index'],            r'$\mathrm{hr}$'),
    (['CAPE_sfc'],                                       r'$\mathrm{J} \cdot \mathrm{kg}^{-1}$'),
    (['EL_height_sfc'],                                  r'$\mathrm{km}$'),
    (['LCL_height_sfc', 'LFC_height_sfc'],               r'$\mathrm{km}$'),
    (['rh_500mb', 'rh_600mb', 'rh_700mb'],               r'$\%$'),
    (['shear_mag_bulk_0to3km', 'shear_mag_bulk_0to6km',
      'shear_mag_bulk_0to9km'],                          r'$\mathrm{m} \cdot \mathrm{s}^{-1}$'),
    (['ccn'],                                            r'$\mathrm{cm}^{-3}$'),
]
PANEL_LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']


def _series(df, var):
    """Across-track values of ``var``, NaNs dropped, converted to plotting units."""
    data = df[var].dropna()
    if var in M_TO_KM:
        data = data / 1000.0
    if var in TIMESTEP_TO_HR:
        data = data * 0.25
    return data


def _xlimits(df, variables):
    """Dynamic x-window for a panel: smallest 2.5th pct, largest 97.5th pct across its vars."""
    lo = min(np.nanpercentile(_series(df, v), 2.5) for v in variables)
    hi = max(np.nanpercentile(_series(df, v), 97.5) for v in variables)
    return lo, hi


def _panel(ax, df, variables, unit, palette, letter):
    """One panel: a step histogram per variable on shared bins, each named inline in its
    colour with its median."""
    xlo, xhi      = XLIM_OVERRIDE.get(letter, _xlimits(df, variables))
    nbins         = NBINS_OVERRIDE.get(letter, NBINS)
    bins          = np.linspace(xlo, xhi, nbins + 1)
    letter_pad_pt = mpl.rcParams['ytick.major.pad'] + mpl.rcParams['ytick.major.size']
    letter_tf     = offset_copy(ax.transAxes, fig=ax.figure, x=-letter_pad_pt, y=0, units='points')
    ax.text(0.0, 1.06, f'({letter})', transform=letter_tf, fontsize=10,
            fontweight='bold', ha='right', va='bottom')

    for idx, var in enumerate(variables):
        color = palette[idx]
        data  = _series(df, var)
        sns.histplot(x=data, bins=bins, ax=ax, element='step',
                     fill=True, alpha=FILL_ALPHA, linewidth=1, color=color)
        label = NAMES[var]
        if len(data):
            label = f'{label} ({fmt_median(np.nanmedian(data))} {unit})'
        ax.text(LEGEND_X, LEGEND_Y - idx * LEGEND_DY, label, transform=ax.transAxes,
                color=color, fontweight='regular', fontsize=7, ha='left', va='top')

    ax.set_ylim(0, ax.get_ylim()[1] * (1 + HEADROOM * len(variables)))
    ax.set_xlim(xlo, xhi)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=XTICK_NBINS_OVERRIDE.get(letter, XTICK_NBINS),
                                                 steps=XTICK_STEPS))
    ax.set_xlabel(f'({unit})', fontsize=10)
    ax.set_ylabel('')
    for side in ('left', 'right', 'top', 'bottom'):
        ax.spines[side].set_visible(True)
    ax.tick_params(which='both', direction='out', bottom=True, left=True,
                   top=False, right=False)


def plots_hist_env(plots_dir, df, panels=PANELS, letters=PANEL_LETTERS):
    """Render and save the 3x3 histogram figure. ``df`` is the filtered cells, one row per
    track; ``panels`` and ``letters`` are aligned, row-major."""
    sns.set_theme(style='white', rc={'font.family': 'Arial'})
    palette = sns.color_palette('deep')

    fig, axs = plt.subplots(3, 3, figsize=(9, 6.5))
    for ax, (variables, unit), letter in zip(axs.ravel(), panels, letters):
        _panel(ax, df, variables, unit, palette, letter=letter)
    for ax in axs[:, 0]:
        ax.set_ylabel('Count', fontsize=10)

    fig.subplots_adjust(wspace=0.25, hspace=0.6)
    fig.savefig(plots_dir + '/hist_env_' + CI_variable + mask_sfx + '.png', dpi=300, bbox_inches='tight')
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

    df_updr = xr.open_dataset(data_dir + '/df_updr_' + CI_variable + '.nc').to_dataframe()
    good    = xr.open_dataset(data_dir + '/the_chosen_few_' + CI_variable + '.nc')[mask_var].values
    df_updr = df_updr[good]
    plots_hist_env(plots_dir, df_updr)
    plt.close('all')
