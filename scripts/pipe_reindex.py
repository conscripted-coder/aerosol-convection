import numpy as np
import pandas as pd
import xarray as xr
import warnings
import sys
import yaml

config_file = sys.argv[1]
stream      = open(config_file, 'r')
config      = yaml.full_load(stream)

celltracking_dir = config['celltracking_dir']
stats_dir        = celltracking_dir + '/stats'
data_dir         = celltracking_dir + '/data'
CI_variable      = config['CI_variable']

# Per-timestep track stats (tracks, times)
sx_tstats = xr.open_dataset(stats_dir + '/trackstats_20181015.0000_20190303.0000.nc',decode_times=True)
sx_goes16 = xr.open_dataset(stats_dir + '/stats_goes16_20181015.0000_20190303.0000.nc',decode_times=True)
sx_3dpctl = xr.open_dataset(stats_dir + '/stats_3d_ppi_20181015.0000_20190303.0000.nc',decode_times=True)

# Environment and aerosol stats (tracks, reltime)
sx_interp = xr.open_dataset(stats_dir + '/interpsonde_parameters_celltrack_20181015.0000_20190303.0000.nc',decode_times=True)
sx_aerosl = xr.open_dataset(stats_dir + '/aerosol_parameters_celltrack_20181015.0000_20190303.0000.nc',decode_times=True)
sx_surfac = xr.open_dataset(stats_dir + '/combomet_parameters_celltrack_20181015.0000_20190303.0000.nc',decode_times=True)

# Candidate CI metrics. CI_variable (config.yaml) picks the one that drives the pipeline;
# every metric's peak time is also saved, to test whether they agree on CI timing.
CI_SOURCES = {
    'eth_10dbz': sx_tstats['maxETH_10dbz'],
    'Tb_min':    sx_goes16['cloud_top_temperature_min'] * -1,
    'zh_max':    sx_3dpctl['max_reflectivity'].isel(z=3),
    'rr_max':    sx_3dpctl['max_rainrate'].isel(z=3),
    'volrain':   sx_3dpctl['volrain'].isel(z=3),
    'rainarea':  sx_3dpctl['num_dbz'].isel(z=3),
}
timeseries = CI_SOURCES[CI_variable]

CI_MAX_TIMESTEPS = 8  # CI window: first 8 timesteps (2 h) after track birth


def first_max_index(series, nmax=CI_MAX_TIMESTEPS):
    """Timestep at which each track's series first peaks within the first `nmax` timesteps.

    `series` is (tracks, times). Returns a float array (tracks,): the 0-based index of the
    first occurrence of the window maximum (rainarea can plateau), or NaN if the window is
    all-NaN. NaNs inside the window are skipped, so a track can have a peak for one metric
    and none for another.
    """
    idx = np.full(series.shape[0], np.nan)
    for track in range(series.shape[0]):
        track_data = series[track, :nmax]
        if not np.all(np.isnan(track_data)):
            max_value   = np.nanmax(track_data)
            max_indices = np.where(track_data == max_value)[0]
            if len(max_indices) > 0:
                idx[track] = max_indices[0]
    return idx


sample_index = first_max_index(timeseries)


sample_index = xr.DataArray(
    sample_index,
    coords={'tracks': timeseries.coords['tracks']},
    dims=['tracks']
)
np.sum(np.isnan(sample_index))

# CI mask: True only at each track's sample_index timestep (all z)
tracks_coord = sx_3dpctl.coords['tracks']
times_coord  = sx_3dpctl.coords['times']
z_coord      = sx_3dpctl.coords['z']
data_array = np.zeros((len(tracks_coord), len(times_coord), len(z_coord)), dtype=bool)

for track_idx in range(len(tracks_coord)):
    idx = sample_index[track_idx].values
    if not np.isnan(idx):
        idx = int(idx)
        data_array[track_idx, idx, :] = True
        # data_array[track_idx, :idx+1, :] = True
        # data_array[track_idx, max(idx-3, 0):idx+1, :] = True
        # data_array[track_idx, max(ci_time_idx-1, 0):ci_time_idx+1, :] = True  # 1 timestep before and including CI_time
        # data_array[track_idx, max(ci_time_idx-3, 0):ci_time_idx+1, :] = True  # 4 timesteps before and including CI_time
        # data_array[track_idx, max(ci_time_idx-1, 0):ci_time_idx+2, :] = True  # 3 timesteps: one before, CI, one after
ci_3d = xr.DataArray(
    data_array,
    coords={
        'tracks': tracks_coord,
        'times': times_coord,
        'z': z_coord
    },
    dims=['tracks', 'times', 'z']
)

ci_2d = xr.DataArray(
    data_array[:,:,0],
    coords={
        'tracks': tracks_coord,
        'times': times_coord,
    },
    dims=['tracks', 'times']
)
maxETH_10dbz             = sx_tstats['maxETH_10dbz'].where(ci_2d).max(dim='times')
maxETH_10dbz_timeseries   = sx_tstats['maxETH_10dbz'].isel(times=slice(0, CI_MAX_TIMESTEPS)).max(dim='times')
max_dbz                   = sx_tstats['max_dbz'].where(ci_2d).max(dim='times')
core_area_sampled         = sx_tstats['core_area'].where(ci_2d).max(dim='times')
core_area_initial         = sx_tstats['core_area'].sel(times=0)
base_time                 = sx_tstats['base_time'].sel(times=0)
# skipna=True is required: datetime max() returns NaT if any entry is NaT, and where(ci_2d)
# makes all but one timestep NaT
sample_time               = sx_tstats['base_time'].where(ci_2d).max(dim='times', skipna=True)
track_duration_full       = sx_tstats['track_duration']  # TODO: make relative to CI time
core_mean_x               = sx_tstats['core_mean_x'].where(ci_2d).median(dim='times')
core_mean_y               = sx_tstats['core_mean_y'].where(ci_2d).median(dim='times')
cloud_top_temperature_min = sx_goes16['cloud_top_temperature_min'].where(ci_2d).max(dim='times')
temperature_ir_min        = sx_goes16['temperature_ir_min'].where(ci_2d).max(dim='times')
cloud_top_height_max      = sx_goes16['cloud_top_height_max'].where(ci_2d).max(dim='times')

# Peak timestep and time of every CI metric. The active metric's columns duplicate
# sample_index / sample_time on purpose, keeping all six uniform.
sample_index_by_ci = {}  # metric -> peak timestep, per track
sample_time_by_ci  = {}  # metric -> peak time, per track
for ci_name, ci_series in CI_SOURCES.items():
    idx = first_max_index(ci_series)

    # (tracks, times) mask True at the peak, to read its time off like sample_time
    peak_mask = np.zeros((len(tracks_coord), len(times_coord)), dtype=bool)
    for track_idx, track_peak in enumerate(idx):
        if not np.isnan(track_peak):
            peak_mask[track_idx, int(track_peak)] = True
    peak_mask = xr.DataArray(
        peak_mask,
        coords={'tracks': tracks_coord, 'times': times_coord},
        dims=['tracks', 'times'],
    )

    sample_index_by_ci[ci_name] = xr.DataArray(
        idx, coords={'tracks': timeseries.coords['tracks']}, dims=['tracks']
    )
    # No peak -> all-False mask -> NaT
    sample_time_by_ci[ci_name] = sx_tstats['base_time'].where(peak_mask).max(dim='times', skipna=True)

tmin = -4; tmax = -0
# CAPE/CIN Metrics
# CAPEacbl_IB_mu              = sx_interp['CAPEacbl_IB_mu'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
# CAPEacbl_IB_sfc             = sx_interp['CAPEacbl_IB_sfc'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
# CAPE_mu                     = sx_interp['CAPE_mu'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
CAPE_sfc                    = sx_interp['CAPE_sfc'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
CAPE_ms                     = np.sqrt(2.0 * CAPE_sfc)  # parcel-theory velocity, m/s
# CIN_NA_mu                   = sx_interp['CIN_NA_mu'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
CIN_sfc                     = sx_interp['CIN_NA_sfc'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
# LFC_height_mu               = sx_interp['LFC_height_mu'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
LFC_height_sfc              = sx_interp['LFC_height_sfc'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
LCL_height_sfc              = sx_interp['LCL_height_sfc'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
LCL_height_mu               = sx_interp['LCL_height_mu'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
# EL_height_mu                = sx_interp['EL_height_mu'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
EL_height_sfc               = sx_interp['EL_height_sfc'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
initial_ht_parcel_mu        = sx_interp['initial_ht_parcel_mu'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
BLtop_height_LiuLiang       = sx_interp['BLtop_height_LiuLiang'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')

# MISC
theta_mu                    = sx_interp['theta_mu'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
thetae_mu                   = sx_interp['thetae_mu'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
thetav_mu                   = sx_interp['thetav_mu'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
thetae_sfc                  = sx_interp['thetae_sfc'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
thetav_sfc                  = sx_interp['thetav_sfc'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
qv_sfc                      = sx_interp['rvap_sfc'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')

# thetae_mean_subcloud_sfc    = sx_interp['thetae_mean_subcloud_sfc'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
# thetae_mean_ACBL_sfc        = sx_interp['thetae_mean_ACBL_sfc'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
# thetae_mean_FT              = sx_interp['thetae_mean_FT'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
# LapseRate_Tv_meanACBL_sfc   = sx_interp['LapseRate_Tv_meanACBL_sfc'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
# LapseRate_Tv_meanBL         = sx_interp['LapseRate_Tv_meanBL'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')

# Humidity Metrics
RH_BLmean                   = sx_interp['RH_BLmean'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
RH_mean_ACBL                = sx_interp['RH_mean_ACBL'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
RH_FTMean                   = sx_interp['RH_FTMean'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
RH_sfc                      = sx_interp['RH_sfc'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
rh_850mb                    = sx_interp['rh_850mb'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
rh_700mb                    = sx_interp['rh_700mb'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
rh_600mb                    = sx_interp['rh_600mb'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
rh_500mb                    = sx_interp['rh_500mb'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
rvap_mean_BL                = sx_interp['rvap_mean_BL'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
rvap_850mb                  = sx_interp['rvap_850mb'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
rvap_700mb                  = sx_interp['rvap_700mb'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
rvap_600mb                  = sx_interp['rvap_600mb'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
rvap_500mb                  = sx_interp['rvap_500mb'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
PW_interpsonde              = sx_interp['PW_interpsonde'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
PW_mwr                      = sx_interp['PW_mwr'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
PW_mwr                      = PW_mwr.where(PW_mwr >= 0, np.nan)

# Wind shear
Ucrel_mean_FT               = sx_interp['Ucrel_mean_FT'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
Vcrel_mean_FT               = sx_interp['Vcrel_mean_FT'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
U_500mb                     = sx_interp['U_500mb'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
V_500mb                     = sx_interp['V_500mb'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
U_700mb                     = sx_interp['U_700mb'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
V_700mb                     = sx_interp['V_700mb'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
U_850mb                     = sx_interp['U_850mb'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
V_850mb                     = sx_interp['V_850mb'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
U_met                       = sx_surfac['U_met'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
V_met                       = sx_surfac['V_met'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
shear_mag_bulk_FT_sfc       = sx_interp['shear_mag_bulk_FT_sfc'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
shear_mag_bulk_0to1km       = sx_interp['shear_mag_bulk_0to1km'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
shear_mag_bulk_0to3km       = sx_interp['shear_mag_bulk_0to3km'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
shear_mag_bulk_0to6km       = sx_interp['shear_mag_bulk_0to6km'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
shear_mag_bulk_0to9km       = sx_interp['shear_mag_bulk_0to9km'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
shear_mag_mean_subcloud_sfc = sx_interp['shear_mag_mean_subcloud_sfc'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
shear_mag_mean_acbl_sfc     = sx_interp['shear_mag_mean_acbl_sfc'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')

freezing_ht                 = sx_interp['freezing_ht'].sel(reltime=slice(tmin,tmax)).median(dim='reltime')
ind_frz                     = np.round(freezing_ht.data/500).astype(int)

tracks_coord = sx_3dpctl.coords['tracks']
times_coord  = sx_3dpctl.coords['times']
z_coord      = sx_3dpctl.coords['z']


hid_temp = sx_3dpctl['hid_temp']  # (tracks, times, z), deg C

# Freezing level: lowest z with T < 0, per (track, time)
is_frozen    = (hid_temp < 0)
freezing_idx = is_frozen.argmax(dim='z')
any_frozen   = is_frozen.any(dim='z')
freezing_idx = freezing_idx.where(any_frozen)  # NaN where no level is frozen

nfrz_levels = 6  # levels above freezing included in ifrz
z_indices   = xr.DataArray(np.arange(len(z_coord)), dims='z', coords={'z': z_coord})

# iliq: surface up to and including the freezing level
iliq = (z_indices <= freezing_idx)
# iliq: 2 km up to and including the freezing level
# iliq = (z_indices <= freezing_idx) & (z_indices >= 4)

# ifrz: freezing_idx+1 through freezing_idx+nfrz_levels
ifrz = (z_indices > freezing_idx) & (z_indices <= freezing_idx + nfrz_levels)

# Where no freezing level exists, both masks are False
ifrz = ifrz.where(any_frozen, other=False).fillna(False).astype(bool)
iliq = iliq.where(any_frozen, other=False).fillna(False).astype(bool)

# icol: surface to the top of ifrz
icol = (iliq | ifrz)

ifrz = ifrz.transpose('tracks', 'times', 'z')
ifrz = xr.DataArray(
    ifrz.values,
    coords={'tracks': tracks_coord, 'times': times_coord, 'z': z_coord},
    dims=['tracks', 'times', 'z'],
    name='above_freezing_mask'
)
iliq = iliq.transpose('tracks', 'times', 'z')
iliq = xr.DataArray(
    iliq.values,
    coords={'tracks': tracks_coord, 'times': times_coord, 'z': z_coord},
    dims=['tracks', 'times', 'z'],
    name='below_freezing_mask'
)
icol = icol.transpose('tracks', 'times', 'z')
icol = xr.DataArray(
    icol.values,
    coords={'tracks': tracks_coord, 'times': times_coord, 'z': z_coord},
    dims=['tracks', 'times', 'z'],
    name='full_column_mask'
)

# Polarimetric percentiles: median over the masked levels at the CI timestep
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
# zh_frz_p02  = sx_3dpctl['p02_reflectivity'].where(ci_3d).where(ifrz).median(dim=['z', 'times'])
# zh_liq_p02  = sx_3dpctl['p02_reflectivity'].where(ci_3d).where(iliq).median(dim=['z', 'times'])
# zh_frz_p10  = sx_3dpctl['p10_reflectivity'].where(ci_3d).where(ifrz).median(dim=['z', 'times'])
# zh_liq_p10  = sx_3dpctl['p10_reflectivity'].where(ci_3d).where(iliq).median(dim=['z', 'times'])
# zh_frz_p25  = sx_3dpctl['p25_reflectivity'].where(ci_3d).where(ifrz).median(dim=['z', 'times'])
# zh_liq_p25  = sx_3dpctl['p25_reflectivity'].where(ci_3d).where(iliq).median(dim=['z', 'times'])
zh_frz_p50  = sx_3dpctl['p50_reflectivity'].where(ci_3d).where(ifrz).median(dim=['z', 'times'])
zh_liq_p50  = sx_3dpctl['p50_reflectivity'].where(ci_3d).where(iliq).median(dim=['z', 'times'])
zh_frz_p75  = sx_3dpctl['p75_reflectivity'].where(ci_3d).where(ifrz).median(dim=['z', 'times'])
zh_liq_p75  = sx_3dpctl['p75_reflectivity'].where(ci_3d).where(iliq).median(dim=['z', 'times'])
zh_frz_p90  = sx_3dpctl['p90_reflectivity'].where(ci_3d).where(ifrz).median(dim=['z', 'times'])
zh_liq_p90  = sx_3dpctl['p90_reflectivity'].where(ci_3d).where(iliq).median(dim=['z', 'times'])
zh_frz_p98  = sx_3dpctl['p98_reflectivity'].where(ci_3d).where(ifrz).median(dim=['z', 'times'])
zh_liq_p98  = sx_3dpctl['p98_reflectivity'].where(ci_3d).where(iliq).median(dim=['z', 'times'])
# zh_frz_max  = sx_3dpctl['max_reflectivity'].where(ci_3d).where(ifrz).median(dim=['z', 'times'])
# zh_liq_max  = sx_3dpctl['max_reflectivity'].where(ci_3d).where(iliq).median(dim=['z', 'times'])

# zdr_frz_p02 = sx_3dpctl['p02_zdr'].where(ci_3d).where(ifrz).median(dim=['z', 'times'])
# zdr_liq_p02 = sx_3dpctl['p02_zdr'].where(ci_3d).where(iliq).median(dim=['z', 'times'])
# zdr_frz_p10 = sx_3dpctl['p10_zdr'].where(ci_3d).where(ifrz).median(dim=['z', 'times'])
# zdr_liq_p10 = sx_3dpctl['p10_zdr'].where(ci_3d).where(iliq).median(dim=['z', 'times'])
# zdr_frz_p25 = sx_3dpctl['p25_zdr'].where(ci_3d).where(ifrz).median(dim=['z', 'times'])
# zdr_liq_p25 = sx_3dpctl['p25_zdr'].where(ci_3d).where(iliq).median(dim=['z', 'times'])
# zdr_frz_p50 = sx_3dpctl['p50_zdr'].where(ci_3d).where(ifrz).median(dim=['z', 'times'])
zdr_liq_p50 = sx_3dpctl['p50_zdr'].where(ci_3d).where(iliq).median(dim=['z', 'times'])
# zdr_frz_p75 = sx_3dpctl['p75_zdr'].where(ci_3d).where(ifrz).median(dim=['z', 'times'])
zdr_liq_p75 = sx_3dpctl['p75_zdr'].where(ci_3d).where(iliq).median(dim=['z', 'times'])
# zdr_frz_p90 = sx_3dpctl['p90_zdr'].where(ci_3d).where(ifrz).median(dim=['z', 'times'])
zdr_liq_p90 = sx_3dpctl['p90_zdr'].where(ci_3d).where(iliq).median(dim=['z', 'times'])
# zdr_frz_p98 = sx_3dpctl['p98_zdr'].where(ci_3d).where(ifrz).median(dim=['z', 'times'])
zdr_liq_p98 = sx_3dpctl['p98_zdr'].where(ci_3d).where(iliq).median(dim=['z', 'times'])
# zdr_frz_max = sx_3dpctl['max_zdr'].where(ci_3d).where(ifrz).median(dim=['z', 'times'])
# zdr_liq_max = sx_3dpctl['max_zdr'].where(ci_3d).where(iliq).median(dim=['z', 'times'])
# kdp_frz_p02 = sx_3dpctl['p02_kdp'].where(ci_3d).where(ifrz).median(dim=['z', 'times'])
# kdp_liq_p02 = sx_3dpctl['p02_kdp'].where(ci_3d).where(iliq).median(dim=['z', 'times'])
# kdp_frz_p10 = sx_3dpctl['p10_kdp'].where(ci_3d).where(ifrz).median(dim=['z', 'times'])
# kdp_liq_p10 = sx_3dpctl['p10_kdp'].where(ci_3d).where(iliq).median(dim=['z', 'times'])
# kdp_frz_p25 = sx_3dpctl['p25_kdp'].where(ci_3d).where(ifrz).median(dim=['z', 'times'])
# kdp_liq_p25 = sx_3dpctl['p25_kdp'].where(ci_3d).where(iliq).median(dim=['z', 'times'])
# kdp_frz_p50 = sx_3dpctl['p50_kdp'].where(ci_3d).where(ifrz).median(dim=['z', 'times'])
kdp_liq_p50 = sx_3dpctl['p50_kdp'].where(ci_3d).where(iliq).median(dim=['z', 'times'])
# kdp_frz_p75 = sx_3dpctl['p75_kdp'].where(ci_3d).where(ifrz).median(dim=['z', 'times'])
kdp_liq_p75 = sx_3dpctl['p75_kdp'].where(ci_3d).where(iliq).median(dim=['z', 'times'])
# kdp_frz_p90 = sx_3dpctl['p90_kdp'].where(ci_3d).where(ifrz).median(dim=['z', 'times'])
kdp_liq_p90 = sx_3dpctl['p90_kdp'].where(ci_3d).where(iliq).median(dim=['z', 'times'])
# kdp_frz_p98 = sx_3dpctl['p98_kdp'].where(ci_3d).where(ifrz).median(dim=['z', 'times'])
kdp_liq_p98 = sx_3dpctl['p98_kdp'].where(ci_3d).where(iliq).median(dim=['z', 'times'])
# kdp_frz_max = sx_3dpctl['max_kdp'].where(ci_3d).where(ifrz).median(dim=['z', 'times'])
# kdp_liq_max = sx_3dpctl['max_kdp'].where(ci_3d).where(iliq).median(dim=['z', 'times'])


iz = 3
# rr_min      = sx_3dpctl['min_rainrate'].where(ci_3d).isel(z=iz).median(dim=['times'])
# rr_p02      = sx_3dpctl['p02_rainrate'].where(ci_3d).isel(z=iz).median(dim=['times'])
# rr_p10      = sx_3dpctl['p10_rainrate'].where(ci_3d).isel(z=iz).median(dim=['times'])
# rr_p25      = sx_3dpctl['p25_rainrate'].where(ci_3d).isel(z=iz).median(dim=['times'])
rr_p50      = sx_3dpctl['p50_rainrate'].where(ci_3d).isel(z=iz).median(dim=['times'])
rr_p75      = sx_3dpctl['p75_rainrate'].where(ci_3d).isel(z=iz).median(dim=['times'])
rr_p90      = sx_3dpctl['p90_rainrate'].where(ci_3d).isel(z=iz).median(dim=['times'])
rr_p98      = sx_3dpctl['p98_rainrate'].where(ci_3d).isel(z=iz).median(dim=['times'])
# rr_max      = sx_3dpctl['max_rainrate'].where(ci_3d).isel(z=iz).median(dim=['times'])
# rwc_min     = sx_3dpctl['min_rwc'].where(ci_3d).isel(z=iz).median(dim=['times'])
# rwc_p02     = sx_3dpctl['p02_rwc'].where(ci_3d).isel(z=iz).median(dim=['times'])
# rwc_p10     = sx_3dpctl['p10_rwc'].where(ci_3d).isel(z=iz).median(dim=['times'])
# rwc_p25     = sx_3dpctl['p25_rwc'].where(ci_3d).isel(z=iz).median(dim=['times'])
# rwc_p50     = sx_3dpctl['p50_rwc'].where(ci_3d).isel(z=iz).median(dim=['times'])
# rwc_p75     = sx_3dpctl['p75_rwc'].where(ci_3d).isel(z=iz).median(dim=['times'])
# rwc_p90     = sx_3dpctl['p90_rwc'].where(ci_3d).isel(z=iz).median(dim=['times'])
# rwc_p98     = sx_3dpctl['p98_rwc'].where(ci_3d).isel(z=iz).median(dim=['times'])
# rwc_max     = sx_3dpctl['max_rwc'].where(ci_3d).isel(z=iz).median(dim=['times'])
# Dm_min      = sx_3dpctl['min_Dm'].where(ci_3d).isel(z=iz).median(dim=['times'])
# Dm_p02      = sx_3dpctl['p02_Dm'].where(ci_3d).isel(z=iz).median(dim=['times'])
# Dm_p10      = sx_3dpctl['p10_Dm'].where(ci_3d).isel(z=iz).median(dim=['times'])
# Dm_p25      = sx_3dpctl['p25_Dm'].where(ci_3d).isel(z=iz).median(dim=['times'])
Dm_p50      = sx_3dpctl['p50_Dm'].where(ci_3d).isel(z=iz).median(dim=['times'])
Dm_p75      = sx_3dpctl['p75_Dm'].where(ci_3d).isel(z=iz).median(dim=['times'])
Dm_p90      = sx_3dpctl['p90_Dm'].where(ci_3d).isel(z=iz).median(dim=['times'])
Dm_p98      = sx_3dpctl['p98_Dm'].where(ci_3d).isel(z=iz).median(dim=['times'])
# Dm_max      = sx_3dpctl['max_Dm'].where(ci_3d).isel(z=iz).median(dim=['times'])
volrain     = sx_3dpctl['volrain'].where(ci_3d).isel(z=iz).median(dim=['times'])
num_dbz_sfc = sx_3dpctl['num_dbz'].where(ci_3d).isel(z=iz).median(dim=['times']) * (500/1000)**2  # km^2


# CCN metrics
tmin = -12
cpcf_concentration     = sx_aerosl['cpcf_concentration'].sel(reltime=slice(tmin,0)).median(dim='reltime')
ccn_concentration_colb = sx_aerosl['ccn_concentration_colb'].sel(reltime=slice(tmin,0)).median(dim='reltime')

# HID fractions, each normalized by the echo count in the region where the species can occur
# (Dolan et al. 2013). HDG and hail straddle 0 C and are split by region; LDG is T<0 only.
num_dbz_liq          = sx_3dpctl['num_dbz'].where(ci_3d).where(iliq)
num_dbz_frz          = sx_3dpctl['num_dbz'].where(ci_3d).where(ifrz)
num_dbz              = sx_3dpctl['num_dbz'].where(ci_3d)

drizzle              = sx_3dpctl['drizzle'].where(ci_3d).where(iliq)
rain                 = sx_3dpctl['rain'].where(ci_3d).where(iliq)
wet_snow             = sx_3dpctl['wet_snow'].where(ci_3d).where(iliq)  # melting non-rimed; not in snow
big_drops            = sx_3dpctl['big_drops'].where(ci_3d).where(iliq)

ice_crystals         = sx_3dpctl['ice_crystals'].where(ci_3d).where(ifrz)
aggregates           = sx_3dpctl['aggregates'].where(ci_3d).where(ifrz)
vertical_ice         = sx_3dpctl['vertical_ice'].where(ci_3d).where(ifrz)
low_density_graupel  = sx_3dpctl['low_density_graupel'].where(ci_3d).where(ifrz)

high_density_graupel_frz = sx_3dpctl['high_density_graupel'].where(ci_3d).where(ifrz)
high_density_graupel_liq = sx_3dpctl['high_density_graupel'].where(ci_3d).where(iliq)
hail_frz                 = sx_3dpctl['hail'].where(ci_3d).where(ifrz)
hail_liq                 = sx_3dpctl['hail'].where(ci_3d).where(iliq)

hid_01 = ( drizzle.sum(dim='z')                  / num_dbz_liq.sum(dim='z') ).median(dim='times')
hid_02 = ( rain.sum(dim='z')                     / num_dbz_liq.sum(dim='z') ).median(dim='times')
hid_03 = ( ice_crystals.sum(dim='z')             / num_dbz_frz.sum(dim='z') ).median(dim='times')
hid_04 = ( aggregates.sum(dim='z')               / num_dbz_frz.sum(dim='z') ).median(dim='times')
hid_05 = ( wet_snow.sum(dim='z')                 / num_dbz_liq.sum(dim='z') ).median(dim='times')
hid_06 = ( vertical_ice.sum(dim='z')             / num_dbz_frz.sum(dim='z') ).median(dim='times')
hid_07 = ( low_density_graupel.sum(dim='z')      / num_dbz_frz.sum(dim='z') ).median(dim='times')
hid_08 = ( high_density_graupel_frz.sum(dim='z') / num_dbz_frz.sum(dim='z') ).median(dim='times')
hid_09 = ( hail_frz.sum(dim='z')                 / num_dbz_frz.sum(dim='z') ).median(dim='times')
hid_10 = ( big_drops.sum(dim='z')                / num_dbz_liq.sum(dim='z') ).median(dim='times')

# Liquid-region parts of the straddling species
hid_11 = ( high_density_graupel_liq.sum(dim='z') / num_dbz_liq.sum(dim='z') ).median(dim='times')
hid_12 = ( hail_liq.sum(dim='z')                 / num_dbz_liq.sum(dim='z') ).median(dim='times')

# Group fractions. ci_3d keeps one timestep, so each equals the sum of its members':
#   snow     = icec + aggr + vice              (ice region)
#   riming   = LDG + HDG_frz + hail_frz        (ice region)
#   melthail = big_drops + HDG_liq + hail_liq  (liquid region)
snow     = ( (ice_crystals + aggregates + vertical_ice).sum(dim='z')                    / num_dbz_frz.sum(dim='z') ).median(dim='times')
riming   = ( (low_density_graupel + high_density_graupel_frz + hail_frz).sum(dim='z')   / num_dbz_frz.sum(dim='z') ).median(dim='times')
melthail = ( (big_drops + high_density_graupel_liq + hail_liq).sum(dim='z')             / num_dbz_liq.sum(dim='z') ).median(dim='times')

ds_updr = xr.Dataset({
    "max_echotop_10dbz": maxETH_10dbz
    ,"max_echotop_10dbz_anytime": maxETH_10dbz_timeseries
    ,"track_duration_full": track_duration_full
    ,"core_area_sampled": core_area_sampled
    ,"core_area_initial": core_area_initial
    ,"rain_area_sampled": num_dbz_sfc
    ,"base_time": base_time
    ,"sample_time": sample_time
    ,"sample_index": sample_index
    ,"cloud_top_temperature_min": cloud_top_temperature_min
    ,"cloud_top_height_max": cloud_top_height_max
    ,"temperature_ir_min": temperature_ir_min
    ,"core_mean_x": core_mean_x
    ,"core_mean_y": core_mean_y
    # ,"zh_liq_p02": zh_liq_p02
    # ,"zh_liq_p10": zh_liq_p10
    # ,"zh_liq_p25": zh_liq_p25
    ,"zh_liq_p50": zh_liq_p50
    ,"zh_liq_p75": zh_liq_p75
    ,"zh_liq_p90": zh_liq_p90
    ,"zh_liq_p98": zh_liq_p98
    # ,"zh_liq_max": zh_liq_max
    # ,"zh_frz_p02": zh_frz_p02
    # ,"zh_frz_p10": zh_frz_p10
    # ,"zh_frz_p25": zh_frz_p25
    ,"zh_frz_p50": zh_frz_p50
    ,"zh_frz_p75": zh_frz_p75
    ,"zh_frz_p90": zh_frz_p90
    ,"zh_frz_p98": zh_frz_p98
    # ,"zh_frz_max": zh_frz_max
    # ,"zdr_liq_p02": zdr_liq_p02
    # ,"zdr_liq_p10": zdr_liq_p10
    # ,"zdr_liq_p25": zdr_liq_p25
    ,"zdr_liq_p50": zdr_liq_p50
    ,"zdr_liq_p75": zdr_liq_p75
    ,"zdr_liq_p90": zdr_liq_p90
    ,"zdr_liq_p98": zdr_liq_p98
    # ,"zdr_liq_max": zdr_liq_max
    # ,"zdr_frz_p50": zdr_frz_p50
    # ,"zdr_frz_p75": zdr_frz_p75
    # ,"zdr_frz_p90": zdr_frz_p90
    # ,"zdr_frz_max": zdr_frz_max
    # ,"kdp_liq_p02": kdp_liq_p02
    # ,"kdp_liq_p10": kdp_liq_p10
    # ,"kdp_liq_p25": kdp_liq_p25
    ,"kdp_liq_p50": kdp_liq_p50
    ,"kdp_liq_p75": kdp_liq_p75
    ,"kdp_liq_p90": kdp_liq_p90
    ,"kdp_liq_p98": kdp_liq_p98
    # ,"kdp_liq_max": kdp_liq_max
    # ,"kdp_frz_p50": kdp_frz_p50
    # ,"kdp_frz_p75": kdp_frz_p75
    # ,"kdp_frz_p90": kdp_frz_p90
    # ,"kdp_frz_max": kdp_frz_max
    # ,"rho_frz_p10": rho_frz_p10
    # ,"rho_frz_p25": rho_frz_p25
    # ,"rho_frz_p50": rho_frz_p50
    # ,"rho_frz_p75": rho_frz_p75
    # ,"rho_frz_p90": rho_frz_p90
    # ,"rr_min": rr_min
    # ,"rr_p02": rr_p02
    # ,"rr_p10": rr_p10
    # ,"rr_p25": rr_p25
    ,"rr_p50": rr_p50
    ,"rr_p75": rr_p75
    ,"rr_p90": rr_p90
    ,"rr_p98": rr_p98
    # ,"rr_max": rr_max
    # ,"rwc_min": rwc_min
    # ,"rwc_p10": rwc_p10
    # ,"rwc_p02": rwc_p02
    # ,"rwc_p25": rwc_p25
    # ,"rwc_p50": rwc_p50
    # ,"rwc_p75": rwc_p75
    # ,"rwc_p90": rwc_p90
    # ,"rwc_p98": rwc_p98
    # ,"rwc_max": rwc_max
    # ,"Dm_min": Dm_min
    # ,"Dm_p02": Dm_p02
    # ,"Dm_p10": Dm_p10
    # ,"Dm_p25": Dm_p25
    ,"Dm_p50": Dm_p50
    ,"Dm_p75": Dm_p75
    ,"Dm_p90": Dm_p90
    ,"Dm_p98": Dm_p98
    # ,"Dm_max": Dm_max
    ,"volrain": volrain
    # ,"Dzh_liq_p50": Dzh_liq_p50
    # ,"Dzh_liq_p75": Dzh_liq_p75
    # ,"Dzh_liq_p90": Dzh_liq_p90
    # ,"Dzh_liq_max": Dzh_liq_max
    # ,"Dzdr_liq_p50": Dzdr_liq_p50
    # ,"Dzdr_liq_p75": Dzdr_liq_p75
    # ,"Dzdr_liq_p90": Dzdr_liq_p90
    # ,"Dzdr_liq_max": Dzdr_liq_max
    # ,"Dzh_cold": Dzh_cold
    # ,"Dzdr_cold": Dzdr_cold
    # ,"rvap_mean_BL": rvap_mean_BL
    # ,"rvap_850mb": rvap_850mb
    # ,"rvap_700mb": rvap_700mb
    # ,"rvap_600mb": rvap_600mb
    # ,"rvap_500mb": rvap_500mb
    ,"rh_500mb": rh_500mb
    ,"rh_600mb": rh_600mb
    ,"rh_700mb": rh_700mb
    ,"rh_850mb": rh_850mb
    ,"rh_BL": RH_BLmean
    ,"rh_mean_ACBL": RH_mean_ACBL
    ,"rh_FTMean": RH_FTMean
    ,"rh_sfc": RH_sfc
    # ,"PW_interpsonde": PW_interpsonde
    ,"pw_mwr": PW_mwr
    # ,"thetae_mean_subcloud_mu": thetae_mean_subcloud_mu
    # ,"thetae_mean_ACBL_mu": thetae_mean_ACBL_mu
    # ,"thetae_mean_FT": thetae_mean_FT
    # ,"LapseRate_Tv_meanACBL_sfc": LapseRate_Tv_meanACBL_sfc
    # ,"LapseRate_Tv_meanBL": LapseRate_Tv_meanBL
    ,"CAPE_sfc": CAPE_sfc
    ,"CAPE_sfc_ms": CAPE_ms
    # ,"ACBL_CAPE": CAPEacbl_IB_sfc
    ,"CIN_sfc": CIN_sfc
    ,"LCL_height_mu": LCL_height_mu
    ,"LCL_height_sfc": LCL_height_sfc
    ,"LFC_height_sfc": LFC_height_sfc
    ,"EL_height_sfc": EL_height_sfc
    ,"initial_ht_parcel_mu": initial_ht_parcel_mu
    ,"BLtop_height_LiuLiang": BLtop_height_LiuLiang
    ,"theta_mu": theta_mu
    ,"thetae_mu": thetae_mu
    ,"thetav_mu": thetav_mu
    ,"thetae_sfc": thetae_sfc
    ,"thetav_sfc": thetav_sfc
    ,"qv_sfc": qv_sfc
    ,"shear_mag_bulk_FT_sfc": shear_mag_bulk_FT_sfc
    ,"shear_mag_bulk_0to1km": shear_mag_bulk_0to1km
    ,"shear_mag_bulk_0to3km": shear_mag_bulk_0to3km
    ,"shear_mag_bulk_0to6km": shear_mag_bulk_0to6km
    ,"shear_mag_bulk_0to9km": shear_mag_bulk_0to9km
    ,"U_500mb": U_500mb
    ,"V_500mb": V_500mb
    ,"U_700mb": U_700mb
    ,"V_700mb": V_700mb
    ,"U_850mb": U_850mb
    ,"V_850mb": V_850mb
    ,"U_met": U_met
    ,"V_met": V_met
    ,"drizzle": hid_01
    ,"rain": hid_02
    ,"ice_crystals": hid_03
    ,"aggregates": hid_04
    ,"wet_snow": hid_05
    ,"vertical_ice": hid_06
    ,"low_density_graupel": hid_07
    ,"high_density_graupel_frz": hid_08
    ,"hail_frz": hid_09
    ,"big_drops": hid_10
    ,"high_density_graupel_liq": hid_11
    ,"hail_liq": hid_12
    ,"rime_liq": melthail
    ,"rime_frz": riming
    ,"snow": snow
    ,"cpcf": cpcf_concentration
    ,"ccn": ccn_concentration_colb
})

# Peak timestep and time columns for every CI metric
for ci_name in CI_SOURCES:
    ds_updr[f'sample_index_{ci_name}'] = sample_index_by_ci[ci_name]
    ds_updr[f'sample_time_{ci_name}']  = sample_time_by_ci[ci_name]

# Data sources
ds_updr.attrs['source_trackstats']  = 'trackstats_20181015.0000_20190303.0000.nc'
ds_updr.attrs['source_goes16']      = 'stats_goes16_20181015.0000_20190303.0000.nc'
ds_updr.attrs['source_3dpctl']      = 'stats_3d_ppi_20181015.0000_20190303.0000.nc'
ds_updr.attrs['source_interpsonde'] = 'interpsonde_parameters_celltrack_20181015.0000_20190303.0000.nc'
ds_updr.attrs['source_aerosol']     = 'aerosol_parameters_celltrack_20181015.0000_20190303.0000.nc'
ds_updr.attrs['source_combomet']    = 'combomet_parameters_celltrack_20181015.0000_20190303.0000.nc'

# CI definition
ds_updr.attrs['CI_variable']      = CI_variable
ds_updr.attrs['CI_max_timesteps'] = CI_MAX_TIMESTEPS

# Freezing level
ds_updr.attrs['nfrz_levels']     = nfrz_levels
ds_updr.attrs['ifrz_definition'] = f'freezing_idx+1 through freezing_idx+{nfrz_levels} ({nfrz_levels} levels above freezing)'
ds_updr.attrs['iliq_definition'] = 'z=0 through freezing_idx (inclusive)'

output_path = data_dir + '/df_updr_' + CI_variable + '.nc'
ds_updr.to_netcdf(output_path)
print(f"Saved Dataset to {output_path}")