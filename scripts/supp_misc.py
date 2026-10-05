"""Three histograms documenting the filter choices in pipe_filter.py.

supp_thetae_diff.png             - theta_e^MU - theta_e^SFC for all tracks
supp_dates_{CI_variable}.png     - filtered cells per start date
supp_radardist_{CI_variable}.png - cells vs. radar range under the 110km mask
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import sys
import yaml
import xarray as xr

mpl.rcParams['font.family'] = 'Arial'

THRESH      = 2  # coupling threshold (K); keep in step with pipe_filter.py
THETAE_BINS = np.arange(0, 15, 2)

DATES_MASK      = 'default'
HIGHLIGHT_ABOVE = 9  # days with more cells than this are red

DIST_MASK = '110km'

# Radar at the domain centre; core_mean_x/y are km from the grid origin (as in pipe_filter.py)
RADAR_X = 110
RADAR_Y = 110

# 20 km (inner filter cut, radar cone of silence) to 110 km (the mask's outer cut)
DIST_BINS = np.arange(20, 120, 10)


def hist_thetae_diff(plots_dir, thetae_diff):
    """Draw and save the theta_e difference histogram with the coupling threshold marked."""
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.histplot(thetae_diff, color='skyblue', bins=THETAE_BINS, ax=ax)
    ax.axvline(THRESH, color='red')
    ax.set_xlabel(r'$\theta_e^{\mathrm{MU}} - \theta_e^{\mathrm{SFC}}$ (°K)')
    ax.set_ylabel('Count')
    ax.set_xticks(THETAE_BINS)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    fig.tight_layout()
    fig.savefig(plots_dir + '/supp_thetae_diff.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def hist_dates(plots_dir, start_times, CI_variable):
    """Draw and save one bar per active start date, in chronological order."""
    counts = pd.Series(pd.to_datetime(start_times).date).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    # Consecutive positions, so days without cells leave no gaps
    colors = ['red' if n > HIGHLIGHT_ABOVE else 'C0' for n in counts.values]
    ax.bar(range(len(counts)), counts.values, color=colors, alpha=0.5, label='Filtered Tracks')
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([d.isoformat() for d in counts.index], rotation=45, ha='right')
    ax.set_xlim(-0.6, len(counts) - 0.4)
    ax.set_xlabel('Cell Initiation Date')
    ax.set_ylabel('Count')

    fig.savefig(plots_dir + '/supp_dates_' + CI_variable + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'dates / mask {DATES_MASK}: {len(start_times)} cells, {len(counts)} distinct start dates')


def hist_dist_radar(plots_dir, df, CI_variable):
    """Draw and save the range histogram for the DIST_MASK sample."""
    # Core distance from the radar at the CI timestep
    dist = np.sqrt((df['core_mean_x'] - RADAR_X) ** 2 + (df['core_mean_y'] - RADAR_Y) ** 2)

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.histplot(dist, bins=DIST_BINS, color='skyblue', ax=ax)
    ax.set_xlabel('Distance From Radar (km)')
    ax.set_ylabel('Count')
    ax.set_xticks(DIST_BINS)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    fig.tight_layout()
    fig.savefig(plots_dir + '/supp_radardist_' + CI_variable + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'dist_radar / mask {DIST_MASK}: {len(dist)} cells, '
          f'range {dist.min():.1f}-{dist.max():.1f} km')


if __name__ == '__main__':
    config_file = sys.argv[1]
    stream      = open(config_file, 'r')
    config      = yaml.full_load(stream)

    celltracking_dir = config['celltracking_dir']
    data_dir         = celltracking_dir + '/data'
    plots_dir        = config['plots_dir']
    CI_variable      = config['CI_variable']

    df_updr = xr.open_dataset(data_dir + '/df_updr_' + CI_variable + '.nc').to_dataframe()
    chosen  = xr.open_dataset(data_dir + '/the_chosen_few_' + CI_variable + '.nc')

    # Unfiltered, to show the full population the coupling filter acts on
    hist_thetae_diff(plots_dir, df_updr['thetae_mu'] - df_updr['thetae_sfc'])

    # base_time is the track's first timestep, i.e. its start time
    good = chosen['good_indices' if DATES_MASK == 'default' else f'good_indices_{DATES_MASK}'].values
    hist_dates(plots_dir, df_updr[good]['base_time'].values, CI_variable)

    good = chosen['good_indices' if DIST_MASK == 'default' else f'good_indices_{DIST_MASK}'].values
    hist_dist_radar(plots_dir, df_updr[good], CI_variable)
