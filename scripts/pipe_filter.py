import numpy as np
import pandas as pd
import xarray as xr
import hashlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sys
import yaml
mpl.rcParams['font.family'] = 'Arial'

config_file = sys.argv[1]
stream      = open(config_file, 'r')
config      = yaml.full_load(stream)

celltracking_dir = config['celltracking_dir']
stats_dir        = celltracking_dir + '/stats'
data_dir         = celltracking_dir + '/data'
plots_dir        = config['plots_dir']
CI_variable      = config['CI_variable']

# Numeric columns the NaN/Inf filter ignores
EXCLUDE_FROM_NAN_FILTER = [
    'cpcf',
]

sx_tstats = xr.open_dataset(stats_dir + '/trackstats_20181015.0000_20190303.0000.nc',decode_times=True)
sx_updr   = xr.open_dataset(data_dir + '/df_updr_' + CI_variable + '.nc')


initial_ht_parcel_mu = sx_updr['initial_ht_parcel_mu'].load()
thetae_sfc           = sx_updr['thetae_sfc'].load()
thetae_mu            = sx_updr['thetae_mu'].load()
thetae_diff          = thetae_mu - thetae_sfc
thresh               = 2  # keep in step with THRESH in supp_misc.py
# i_coupled = thetae_diff < thresh
# Coupled: MU parcel within thresh K of the surface theta-e, or starting at or below 500 m
i_coupled = (thetae_diff < thresh) | (initial_ht_parcel_mu <= 500)


cell_meanlon = sx_tstats['cell_meanlon'].sel(times=0)
split        = sx_tstats['start_split_tracknumber']

# Start east of -64.9 (Varble et al. 2024); no eastern bound, the radar range limits it
# i_eastofridge = (cell_meanlon > -64.9) & (cell_meanlon < -64.5)
i_eastofridge = cell_meanlon > -64.9

# Exclude split-born cells; ending in a merge is allowed (Zhe, pers. comm., Oct 2024)
i_notsplit = np.isnan(split)

# Exclude cells that end in a merge at timestep <= 4
merge               = sx_tstats['end_merge_tracknumber'].load()
end_merge_timeindex = sx_tstats['end_merge_timeindex'].load()
i_notmerge          = (end_merge_timeindex > 4) | (np.isnan(merge))

# 20-80 km from the radar at the CI timestep. core_mean_x/y are km from the grid origin,
# not the radar, which sits at the domain centre (110, 110).
core_mean_x = sx_updr['core_mean_x']
core_mean_y = sx_updr['core_mean_y']
radar_x     = 110
radar_y     = 110
dist        = np.sqrt( (core_mean_x-radar_x)**2 + (core_mean_y-radar_y)**2 )
dist_thresh = 80
i_near      = (dist > 20) & (dist < dist_thresh)  # (tracks)


# track_duration counts every timestep, not just those after CI
track_duration = sx_tstats['track_duration']
i_duration     = track_duration >= 4

# Daytime: start between 12 and 24 UTC
base_time = sx_tstats['base_time'].isel(times=0)
i_daytime = (base_time.dt.hour >= 12) & (base_time.dt.hour < 24)

# maxETH_10dbz = sx_updr['max_echotop_10dbz_anytime'].load()
maxETH_10dbz = sx_updr['max_echotop_10dbz'].load()
i_ETH        = maxETH_10dbz >= 6


# No NaN/Inf in any numeric column (datetimes can't go through isfinite and are skipped)
numeric_vars = [v for v in sx_updr.data_vars
                if np.issubdtype(sx_updr[v].dtype, np.number)
                and v not in EXCLUDE_FROM_NAN_FILTER]
no_nan_inf = np.all(np.isfinite(sx_updr[numeric_vars].to_array().values), axis=0)

# Filter registry: each filter has a 'default' and optional sensitivity variants
all_start_dates = pd.to_datetime(sx_tstats['start_basetime'].values).date
n_tracks        = len(sx_updr.tracks)
filters = {
    'nan':         {'default': no_nan_inf},
    'coupled':     {'default': i_coupled.to_numpy()},
    'eastofridge': {'default': i_eastofridge.to_numpy()},
    'notsplit':    {'default': i_notsplit.to_numpy()},
    'notmerge':    {'default': i_notmerge.to_numpy()},
    'daytime':     {'default': i_daytime.to_numpy()},
    'ETH':         {'default': i_ETH.to_numpy()},
    'near': {
        'default': i_near.to_numpy(),
        '110km':   ((dist > 20) & (dist < 110)).to_numpy(),
        '100km':   ((dist > 20) & (dist < 100)).to_numpy(),
        '90km':    ((dist > 20) & (dist < 90)).to_numpy(),
        '70km':    ((dist > 20) & (dist < 70)).to_numpy(),
        '60km':    ((dist > 20) & (dist < 60)).to_numpy(),
    },
    'duration': {
        'default': i_duration.to_numpy(),
        '2ts':     (track_duration >= 2).to_numpy(),
        '3ts':     (track_duration >= 3).to_numpy(),
        '5ts':     (track_duration >= 5).to_numpy(),
        '6ts':     (track_duration >= 6).to_numpy(),
    },
}

# Date variants each drop one of the 4 most prolific start dates, ranked over tracks passing
# every other default filter. Ties break chronologically so date1..4 are stable across runs.
ranking_mask = np.ones(n_tracks, dtype=bool)
for f in filters.values():
    ranking_mask &= f['default']

date_counts  = pd.Series(all_start_dates[ranking_mask]).value_counts()
ranked_dates = sorted(date_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:4]

filters['date'] = {'default': np.ones(n_tracks, dtype=bool)}
date_meta = {}
for rank, (day, count) in enumerate(ranked_dates, start=1):
    label = f'date{rank}'
    filters['date'][label] = all_start_dates != day
    date_meta[label]       = (day, int(count))

print('Most prolific dates under the default mask:')
for label, (day, count) in date_meta.items():
    print(f'  {label}: {day} ({count} cells)')


# Base good_indices: AND all defaults
good_indices = np.ones(len(sx_updr.tracks), dtype=bool)
for f in filters.values():
    good_indices &= f['default']

print('Good Cells: ', sum(good_indices))

# Tracks each filter keeps on its own (marginal counts, for the manuscript's filter table)
print('\nIndividual filter retention (out of %d tracks):' % n_tracks)
for fname, fopts in filters.items():
    n_pass = int(fopts['default'].sum())
    print(f'  {fname:12s} {n_pass:5d} ({100*n_pass/n_tracks:.1f}%)')
n_joint = int(good_indices.sum())
print(f'  {"ALL joint":12s} {n_joint:5d} ({100*n_joint/n_tracks:.1f}%)')

ds_filter = xr.Dataset(
    {"good_indices": xr.DataArray(good_indices, dims=['tracks'], coords={'tracks': sx_updr.coords['tracks']})},
)

# Sensitivity variants: swap one filter at a time
for fname, fopts in filters.items():
    for vlabel, vmask in fopts.items():
        if vlabel == 'default':
            continue
        gi_var = np.ones(n_tracks, dtype=bool)
        for fname2, fopts2 in filters.items():
            gi_var &= vmask if fname2 == fname else fopts2['default']
        ds_filter['good_indices_' + vlabel] = xr.DataArray(gi_var, dims=['tracks'], coords={'tracks': sx_updr.coords['tracks']})
        print(f"Variant {vlabel}: {sum(gi_var)} cells pass")

# Store each excluded date on its own variant
for label, (day, count) in date_meta.items():
    ds_filter['good_indices_' + label].attrs['excluded_date'] = day.isoformat()
    ds_filter['good_indices_' + label].attrs['n_excluded']    = count

# Hashes for all variants
for var_name in ds_filter.data_vars:
    ii_var = np.where(ds_filter[var_name].values == True)[0]
    ds_filter.attrs[var_name + '_hash'] = hashlib.sha256(ii_var.tobytes()).hexdigest()[:12]

ds_filter.attrs['CI_variable']        = CI_variable
ds_filter.attrs['coupled_filter']     = f'abs(thetav_mu - thetav_sfc) < {thresh} K'
ds_filter.attrs['eastofridge_filter'] = 'cell_meanlon > -64.9'
ds_filter.attrs['notsplit_filter']    = 'start_split_tracknumber is NaN'
ds_filter.attrs['notmerge_filter']    = 'end_merge_timeindex > 4 or end_merge_tracknumber is NaN'
ds_filter.attrs['daytime_filter']     = 'base_time.hour >= 12 and < 24 UTC'
ds_filter.attrs['near_filter']        = 'default: 20 km < distance to radar < 80 km; variants: 110km, 100km, 90km, 70km, 60km upper bound'
ds_filter.attrs['ETH_filter']         = 'max_echotop_10dbz >= 6 km'
ds_filter.attrs['duration_filter']    = 'default: track_duration >= 4; variants: 2, 3, 5, 6 timesteps'
ds_filter.attrs['nan_filter']         = 'rows with NaN or Inf in any feature excluded'
ds_filter.attrs['date_filter'] = (
    'default: all dates; '
    + '; '.join(f'{label}={day.isoformat()} ({count} cells)'
                for label, (day, count) in date_meta.items())
    + '. Ranked by cell count under the default mask of all other filters, '
      'ties broken chronologically.'
)
ds_filter.attrs['n_good']  = int(sum(good_indices))
ds_filter.attrs['n_total'] = len(good_indices)
ds_filter.to_netcdf(data_dir + '/the_chosen_few_' + CI_variable + '.nc')
print(f"Saved filter output to {data_dir}/the_chosen_few_{CI_variable}.nc")
