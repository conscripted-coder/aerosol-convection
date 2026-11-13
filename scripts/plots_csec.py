import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
from cartopy.geodesic import Geodesic
import glob
import os
import pandas as pd
import sys
import yaml
import lib_style
mpl.rcParams['hatch.linewidth'] = 0.4

config           = yaml.full_load(open(sys.argv[1]))
celltracking_dir = config['celltracking_dir']
data_dir         = celltracking_dir + '/data'
terrain_dir      = celltracking_dir + '/terrain'
tracking_dir     = celltracking_dir + '/celltracking/20181015.0000_20190303.0000'
gridded_dir      = config['gridded_dir']
plots_dir        = config['plots_dir']

# Case time (YYYYMMDD.hhmm) and section endpoints (x, y), km from the radar
time_str   = '20181102.2245'
xkm1, ykm1 = -5, -68
xkm2, ykm2 = 50, -68
# Section x-axis: 'along' (distance along section), 'zonal' or 'meridional' (distance from radar)
xaxis = 'zonal'

# Radar file nearest the case time; seconds vary, so match on the full timestamp
radar_files = sorted(glob.glob(gridded_dir + '/taranis_corcsapr2cfrppiqcM1.c1.*_gridded.nc'))
stamps      = [os.path.basename(f)[31:46] for f in radar_files]
radar_times = pd.to_datetime(stamps, format='%Y%m%d.%H%M%S')
i_nearest   = np.abs(radar_times - pd.to_datetime(time_str, format='%Y%m%d.%H%M')).argmin()
radar_file  = radar_files[i_nearest]
# Celltracks share the radar timestamp, separated by '_' instead of '.'
tracks_file = tracking_dir + f"/celltracks_{stamps[i_nearest].replace('.', '_')}.nc"
print(radar_file)
print(tracks_file)

radar = xr.open_dataset(radar_file)
x_1d  = radar['x'].values / 1000
y_1d  = radar['y'].values / 1000

# Sample the line at the grid spacing, taking the nearest grid column at each point
dx     = x_1d[1] - x_1d[0]
length = np.hypot(xkm2 - xkm1, ykm2 - ykm1)
dist   = np.linspace(0, length, int(length / dx) + 1)
ix     = xr.DataArray(np.abs(x_1d[:, None] - (xkm1 + dist / length * (xkm2 - xkm1))).argmin(0), dims='dist')
iy     = xr.DataArray(np.abs(y_1d[:, None] - (ykm1 + dist / length * (ykm2 - ykm1))).argmin(0), dims='dist')

# Plotted x coordinate; zonal/meridional need a line along that axis
if xaxis == 'along':
    xcoord, xlabel = dist, 'Distance Along Section (km)'
elif xaxis == 'zonal':
    if ykm1 != ykm2:
        sys.exit(f"xaxis='zonal' needs a zonal section (ykm1 == ykm2), got {ykm1}, {ykm2}")
    xcoord, xlabel = xkm1 + dist / length * (xkm2 - xkm1), 'Zonal Distance (km)'
elif xaxis == 'meridional':
    if xkm1 != xkm2:
        sys.exit(f"xaxis='meridional' needs a meridional section (xkm1 == xkm2), got {xkm1}, {xkm2}")
    xcoord, xlabel = ykm1 + dist / length * (ykm2 - ykm1), 'Meridional Distance from Radar (km)'
else:
    sys.exit(f"xaxis must be 'along', 'zonal' or 'meridional', got {xaxis!r}")
# Increasing axis, so west/south is on the left; ticks on multiples of 10 km
xlim   = (xcoord.min(), xcoord.max())
xticks = np.arange(np.ceil(xlim[0] / 10) * 10, xlim[1] + 1, 10)

sec = radar.isel(time=0, x=ix, y=iy)
# point_altitude is m MSL
sec_alt  = sec['point_altitude']
sec_dist = np.broadcast_to(xcoord, sec_alt.shape)

dz   = float(sec['z'][1] - sec['z'][0])
ylim = (float(sec['origin_altitude']), 16000)
# ylim = (float(sec['origin_altitude']) - dz / 2, 16000)

# Melting level: lowest grid level with T < 0 C. Temperature is horizontally uniform, so one
# column suffices; z is above the radar, so add origin_altitude for m MSL
temp     = radar['sounding_temperature_post_grid'].isel(time=0, x=0, y=0).values
melt_msl = float(radar['z'][np.argmax(temp < 0)]) + float(radar['origin_altitude'][0])
print(f'Freezing level: {melt_msl:.0f} m MSL')

# Radar and celltracks grids are co-registered, so ix/iy index both
tracks = xr.open_dataset(tracks_file).isel(time=0)

# Section-axis edges of each convective core
line       = tracks['tracknumber'].where(tracks['conv_core'] != 0).isel(lon=ix, lat=iy).values
cell_edges = []
for tn in np.unique(line[line > 0]):
    idxs = np.flatnonzero(line == tn)
    for run in np.split(idxs, np.flatnonzero(np.diff(idxs) > 1) + 1):
        cell_edges += [xcoord[run[0]], xcoord[run[-1]]]

# hid_classes = [
#     ("#1f77b4", 'Drizzle'),
#     ("#ff7f0e", 'Rain'),
#     ("#2ca02c", 'Ice Crystals'),
#     ("#d62728", 'Aggregates'),
#     ("#9467bd", 'Wet Snow'),
#     ("#8c564b", 'Vertically\nOriented Ice'),
#     ("#e377c2", 'Low-density\nGraupel'),
#     ("#7f7f7f", 'High-density\nGraupel'),
#     ("#bcbd22", 'Hail'),
#     ("#17becf", 'Big Drops'),
# ]

# Warm colors for liquid/melting classes, cold colors for ice
hid_classes = [
    ("#fdd835", 'Drizzle'),                       # yellow
    ("#fb8c00", 'Rain'),                          # orange
    ("#b3e5fc", 'Ice Crystals'),                  # pale blue
    ("#4fc3f7", 'Aggregates'),                    # sky blue
    ("#f06292", 'Wet Snow'),                      # pink
    ("#00897b", 'Vertically\nOriented Ice'),      # teal
    ("#7986cb", 'Low-density\nGraupel'),          # periwinkle
    ("#3f51b5", 'High-density\nGraupel'),         # indigo
    ("#6a1b9a", 'Hail'),                          # deep purple
    ("#b71c1c", 'Big Drops'),                     # dark red
]
hid_colors, hid_labels = zip(*hid_classes)
hid_cmap               = ListedColormap(hid_colors)
# Boundaries centered on integers 1–10
hid_norm = BoundaryNorm(np.arange(0.5, 11.5, 1), hid_cmap.N)

fig, axs = plt.subplots(3, 1, figsize=(5, 5), sharex=True)
plt.subplots_adjust(hspace=0.1)

pm1 = axs[0].pcolormesh(sec_dist, sec_alt, sec['taranis_attenuation_corrected_reflectivity'],
                        cmap='rainbow', vmin=0, vmax=60)
pm2 = axs[1].pcolormesh(sec_dist, sec_alt, sec['taranis_rain_rate'],
                        cmap='rainbow', vmin=0, vmax=40)
pm3 = axs[2].pcolormesh(sec_dist, sec_alt, sec['hydrometeor_identification_post_grid'],
                        cmap=hid_cmap, norm=hid_norm)

# Formatting shared by all three panels
caxs = []
for ax, panel_label in zip(axs, ['(a)', '(b)', '(c)']):
    for x in cell_edges:
        ax.axvline(x=x, color='k', linestyle=':', lw=.6, zorder=15)
    ax.axhline(y=melt_msl, color='k', linestyle='--', lw=.4, zorder=15)
    ax.axhline(y=melt_msl+3000, color='k', linestyle='--', lw=.4, zorder=15)
    ax.text(0.98, melt_msl+100, 'T = 0 °C', transform=ax.get_yaxis_transform(),
            ha='right', va='bottom', style='italic', fontsize=5, zorder=20)

    for spine in ax.spines.values():
        spine.set_linewidth(1)
    ax.set_yticks(np.arange(0, 16000, 5000), np.arange(0, 16, 5))
    ax.set_xticks(xticks)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.text(0.02, 0.95, panel_label, transform=ax.transAxes,
            ha='left', va='top', fontsize=9, fontweight='bold', zorder=20)

    # Colorbar axes just right of the panel, matching its height
    pos = ax.get_position()
    caxs.append(fig.add_axes((pos.x1 + 0.03, pos.y0, 0.03, pos.height)))

cb1 = fig.colorbar(pm1, cax=caxs[0])
cb1.set_label('Radar Reflectivity (dBZ)', fontsize=6)

cb2 = fig.colorbar(pm2, cax=caxs[1])
cb2.set_label(r'Rain Rate (mm$\cdot$ hr$^{-1}$)', fontsize=6)

# HID colorbar: one centred label per class, no tick marks
cb3 = fig.colorbar(pm3, cax=caxs[2])
cb3.set_ticks(np.arange(1, 11), labels=hid_labels)
for t in cb3.ax.get_yticklabels():
    t.set_linespacing(1)
# cb3.outline.set_linewidth(1)
cb3.ax.tick_params(which='both', direction='in', labelsize=4, length=0)

# Ice/liquid layer labels and arrows either side of the freezing line, on every panel
for ax in axs:
    blend = ax.get_yaxis_transform()
    for text, (y_top, y_bot) in [('Ice Layer', (melt_msl + 3000, melt_msl)),
                                 ('Liquid Layer', (melt_msl, ylim[0]))]:
        # annotate clips each arrow to the text's bbox, so they start at its top/bottom edge
        for y_end, alpha in [(y_top, 1), (y_bot, 0)]:
            ax.annotate(text, xy=(0.85, y_end), xytext=(0.85, (y_top + y_bot) / 2),
                        xycoords=blend, textcoords=blend, alpha=alpha,
                        ha='center', va='center', fontsize=6, style='italic',
                        arrowprops=dict(arrowstyle='->', color='k', lw=.5, mutation_scale=6,
                                        shrinkA=0, shrinkB=0),
                        zorder=20)

# Convective core label on every panel, arrows reaching out to each core's edges
for ax in axs:
    blend_x = ax.get_xaxis_transform()
    for x_left, x_right in zip(cell_edges[::2], cell_edges[1::2]):
        for x_end, alpha in [(x_left, 1), (x_right, 0)]:
            ax.annotate('Convective\nCore', xy=(x_end, 0.8), xytext=((x_left + x_right) / 2, 0.8),
                        xycoords=blend_x, textcoords=blend_x, alpha=alpha,
                        ha='center', va='center', fontsize=6, style='italic',
                        arrowprops=dict(arrowstyle='->', color='k', lw=.5, mutation_scale=6,
                                        shrinkA=0, shrinkB=0),
                        zorder=20)

# Precipitation level (m MSL)
precip_msl = 3141

# Extra ticks on panel (c) marking the radar altitude at the bottom of the axis and the precipitation level
axs[2].set_yticks([ylim[0], precip_msl, 5000, 10000, 15000],
                  [f'{ylim[0]:.0f} m', f'{precip_msl} m', 5, 10, 15])
for t in axs[2].get_yticklabels()[:2]:
    t.set_fontsize(7)

axs[1].set_ylabel('Height Above Sea Level (km)')
axs[2].set_xlabel(xlabel)

plt.savefig(plots_dir + '/csec_xz.png', dpi=300, bbox_inches="tight")

radar_lon, radar_lat = -64.73, -32.13
proj                 = ccrs.PlateCarree()

hgt = xr.open_dataset(terrain_dir + '/terrain_d01.nc')['HGT'].squeeze() / 1000
lat = xr.open_dataset(terrain_dir + '/lat_d01.nc')['XLAT'].squeeze()
lon = xr.open_dataset(terrain_dir + '/lon_d01.nc')['XLONG'].squeeze()

trk_lon  = tracks['longitude']
trk_lat  = tracks['latitude']
cores    = tracks['tracknumber'].where(tracks['conv_core'] == 1)
dbz_comp = tracks['dbz_comp'].where(tracks['dbz_comp'] >= 5)

# Cores split by sample membership (tracks dim is 0-indexed, tracknumber 1-indexed); a sampled
# cell is not necessarily at its sample time here
sample          = xr.open_dataset(data_dir + '/the_chosen_few_eth_10dbz.nc')
good_tns        = np.flatnonzero(sample['good_indices'].values) + 1
tracked_cores   = cores.where(cores.isin(good_tns))
untracked_cores = cores.where(~cores.isin(good_tns))

# Section line endpoints in lon/lat, at the grid columns nearest each end
sec_lon = [float(trk_lon[iy[0], ix[0]]), float(trk_lon[iy[-1], ix[-1]])]
sec_lat = [float(trk_lat[iy[0], ix[0]]), float(trk_lat[iy[-1], ix[-1]])]
print(f'Section start: y = {ykm1} km -> lat {sec_lat[0]:.3f}, lon {sec_lon[0]:.3f}')
print(f'Section end:   y = {ykm2} km -> lat {sec_lat[1]:.3f}, lon {sec_lon[1]:.3f}')

fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw={'projection': proj})

m1 = ax.pcolormesh(lon, lat, hgt, cmap='Greys', zorder=5, transform=proj)
m2 = ax.pcolormesh(trk_lon, trk_lat, dbz_comp, vmin=0, vmax=60, cmap='rainbow',
                   transform=proj, zorder=10)

# Hatched, outlined footprints of the convective cores
for cells in [tracked_cores, untracked_cores]:
    footprint = cells.notnull().astype(float)
    cs = ax.contourf(trk_lon, trk_lat, footprint, levels=[0.5, 1.5], colors='none',
                     hatches=['////////'], transform=proj, zorder=15)
    cs.set_edgecolor('k')
    cs.set_linewidth(0)
    ax.contour(trk_lon, trk_lat, footprint, levels=[0.5], colors='k', linewidths=0.6,
               transform=proj, zorder=15)

# Range rings, labelled just outside the ring at label_az (degrees clockwise from north)
label_az = 45
for radius_km, n_samples in [(80, 1000), (20, 1000)]:
    ring_lon, ring_lat = Geodesic().circle(lon=radar_lon, lat=radar_lat,
                                           radius=radius_km * 1000, n_samples=n_samples).T
    ax.plot(ring_lon, ring_lat, transform=proj, color='red', linewidth=.75, zorder=20)
    lab_lon, lab_lat = Geodesic().direct((radar_lon, radar_lat), label_az,
                                         radius_km * 1000)[0, :2]
    ax.annotate(f'{radius_km} km', xy=(lab_lon, lab_lat), xycoords=proj._as_mpl_transform(ax),
                xytext=(2, 2), textcoords='offset points', color='red', fontsize=8,
                fontweight='bold', fontstyle='italic', ha='left', va='bottom', zorder=300, clip_on=False)

ax.scatter(radar_lon, radar_lat, color='red', marker='.', zorder=15, transform=proj)
ax.plot(sec_lon, sec_lat, color='k', marker='.',linestyle=':', linewidth=1, zorder=100)
# Label 25 pt above the point 70% along the line, arrow pointing to it
frac = 0.7
ax.annotate('Vertical\nCross Section', xy=(sec_lon[0] + frac * (sec_lon[1] - sec_lon[0]),
                                            sec_lat[0] + frac * (sec_lat[1] - sec_lat[0])), xycoords=proj._as_mpl_transform(ax),
            xytext=(0, 25), textcoords='offset points', fontsize=7, fontweight='bold',
            fontstyle='italic', ha='center', va='bottom', multialignment='center',
            arrowprops=dict(arrowstyle='->', color='k', linewidth=1, shrinkA=1, shrinkB=1), zorder=300)
ax.text(0.3, 0.7, 'Sierras de\nCórdoba', color='k', fontsize=8, fontweight='bold', fontstyle='italic',
        ha='center', va='center', transform=ax.transAxes, zorder=300)
# Arrow head at xy (lon, lat); tail at xytext (axes fraction, 0-1)
ax.annotate('', xy=(-64.95, -31.95), xycoords=proj._as_mpl_transform(ax),
            xytext=(0.3, 0.67), textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='k', linewidth=1), zorder=300)

ax.annotate('AMF Site', xy=(radar_lon, radar_lat), color='r', fontweight='bold', fontstyle='italic',
            xycoords=proj._as_mpl_transform(ax), xytext=(0, -18), textcoords='offset points',
            ha='center', va='center', fontsize=7,
            arrowprops=dict(arrowstyle='->', color='r', linewidth=1.5),
            zorder=100, clip_on=False)

ax.set_aspect('auto', adjustable=None)
ax.tick_params(axis='both', which='major', labelsize=10)
# Ticks must lie inside the extent, else the view expands to include them
ax.set_xticks(np.arange(-65.5, -63.9, 0.5), crs=proj)
ax.set_yticks(np.arange(-32.5, -31.4, 0.5), crs=proj)
# Even ~12 km margin around the 80 km ring; set after the ticks
ax.set_extent([-65.705, -63.755, -32.96, -31.30], crs=proj)
ax.set_xlabel('Longitude (°)', fontsize=10)
ax.set_ylabel('Latitude (°)', fontsize=10)
for spine in ax.spines.values():
    spine.set_linewidth(2)

# Two horizontal colorbars side by side below the map
pos   = ax.get_position()
gap   = 0.08
width = (pos.width - gap) / 2
for x0, m, label in [(pos.x0, m1, 'Terrain Height (km)'),
                     (pos.x0 + width + gap, m2, 'Radar Reflectivity (dBZ)')]:
    cb = fig.colorbar(m, cax=fig.add_axes((x0, pos.y0 - 0.12, width, 0.025)),
                      orientation='horizontal')
    cb.set_label(label, fontsize=10)
    cb.outline.set_linewidth(1)
    cb.ax.tick_params(direction='in', labelsize=10)

plt.savefig(plots_dir + '/csec_xy.png', dpi=300, bbox_inches="tight")
