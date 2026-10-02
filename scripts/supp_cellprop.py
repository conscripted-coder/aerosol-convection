"""Box plots of cell-property distributions across CI definitions and lifetime filters.

  - supp_sampletime.png             → CI time, one box per CI variable
  - supp_duration_{CI_variable}.png → echo top and rain area, one box per lifetime mask
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import yaml
import xarray as xr

mpl.rcParams['font.family'] = 'Arial'

ALL_CI_VARS = ["eth_10dbz", "Tb_min", "zh_max", "rr_max", "volrain", "rainarea"]

# Minimum-lifetime mask suffixes in the_chosen_few_{CI_variable}.nc ("default" = 4 timesteps)
DURATION_VARIANTS = ["2ts", "3ts", "default", "5ts", "6ts"]

# y tick label per box, keyed by CI variable / mask
DEFAULT_TICK_LABELS = {
    "eth_10dbz": "10-dBZ ETH",
    "Tb_min":    "T$_B$",
    "zh_max":    "Z$_{H}$",
    "rr_max":    "Rain Rate",
    "volrain":   "Vol. Rain",
    "rainarea":  "2-km\nPrecip. Area",
}
DURATION_TICK_LABELS = {
    "2ts":     "2 Timesteps",
    "3ts":     "3 Timesteps",
    "default": "4 Timesteps",
    "5ts":     "5 Timesteps",
    "6ts":     "6 Timesteps",
}

# Keys whose tick labels are drawn bold
DEFAULT_BOLD  = {"eth_10dbz", "rainarea"}
DURATION_BOLD = {"default"}

BOX_COLOR = '#AEC6E8'  # light pastel blue


def load_cells(data_dir, ci_var, mask='default'):
    """Cells of df_updr_{ci_var} that pass ``mask`` in the_chosen_few_{ci_var}.nc."""
    df   = xr.open_dataset(data_dir + '/df_updr_' + ci_var + '.nc').to_dataframe()
    var  = 'good_indices' if mask == 'default' else f'good_indices_{mask}'
    good = xr.open_dataset(data_dir + '/the_chosen_few_' + ci_var + '.nc')[var].values
    cells = df[good]
    print(f"  {ci_var} {mask}: good samples = {len(cells)}")
    return cells


def draw_boxplot(ax, cells, labels, bold, column, xlabel, scale=1):
    """Horizontal box plot of ``column`` (times ``scale``), one box per entry in ``cells``.

    Boxes span the 25th-75th percentiles, whiskers the 2.5th-97.5th; each box's mean is
    written beside its median line.
    """
    keys      = list(cells)
    plot_data = [cells[k][column].dropna().values * scale for k in keys]

    bp = ax.boxplot(plot_data, tick_labels=[labels[k] for k in keys], patch_artist=True,
                    showfliers=False, vert=False, whis=[2.5, 97.5])

    for patch in bp['boxes']:
        patch.set_facecolor(BOX_COLOR)
        patch.set_edgecolor('black')
        patch.set_linewidth(1)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    for key, tick in zip(keys, ax.get_yticklabels()):
        if key in bold:
            tick.set_fontweight('bold')

    # First box on top
    ax.invert_yaxis()

    ax.grid(True, alpha=0.3)
    ax.set_xlabel(xlabel, fontsize=12)

    # Mean just right of the median line; boxes sit at y = 1, 2, ...
    for pos, values in enumerate(plot_data, start=1):
        ax.annotate(f"{values.mean():.2f}", xy=(np.median(values), pos),
                    xytext=(4, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=8)


def save(fig, out):
    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    config_file = sys.argv[1]
    stream      = open(config_file, 'r')
    config      = yaml.full_load(stream)

    celltracking_dir = config['celltracking_dir']
    data_dir         = celltracking_dir + '/data'
    plots_dir        = config['plots_dir']
    CI_variable      = config['CI_variable']

    os.makedirs(plots_dir, exist_ok=True)

    # CI time, one box per CI variable. sample_index is in 15-min timesteps since track birth.
    print("\nProcessing default")
    cells = {ci_var: load_cells(data_dir, ci_var) for ci_var in ALL_CI_VARS}
    fig, ax = plt.subplots(figsize=(5, 3))
    draw_boxplot(ax, cells, DEFAULT_TICK_LABELS, DEFAULT_BOLD,
                 'sample_index', 'Time of Max 10-dBZ Echo Top Height (min)', scale=15)
    save(fig, plots_dir + "/supp_sampletime.png")

    # Echo top and rain area, one box per minimum-lifetime mask
    print(f"\nProcessing {CI_variable}_duration")
    cells = {mask: load_cells(data_dir, CI_variable, mask) for mask in DURATION_VARIANTS}
    fig, axes = plt.subplots(2, 1, figsize=(5, 6))
    draw_boxplot(axes[0], cells, DURATION_TICK_LABELS, DURATION_BOLD,
                 'max_echotop_10dbz', 'Maximum 10-dBZ Echo Top Height (km)')
    draw_boxplot(axes[1], cells, DURATION_TICK_LABELS, DURATION_BOLD,
                 'rain_area_sampled', '2-km AGL Precipitation Area (km$^2$)')
    for ax, letter in zip(axes, ['(a)', '(b)']):
        ax.text(-0.08, 1.05, letter, transform=ax.transAxes, fontsize=14, fontweight='bold')
    save(fig, plots_dir + f"/supp_duration_{CI_variable}.png")
