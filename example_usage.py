"""
========================================================
Fused XCH4 Dataset — Example Usage
========================================================
Dataset : Global Daily Fused XCH4 (2020-2023)
Grid    : 0.1° x 0.1°, 70°S-70°N, land only
Units   : ppb (parts per billion)
Format  : HDF5 (.h5), internal gzip compression

Requirements:
    pip install h5py numpy matplotlib cartopy
========================================================
"""

import os
import h5py
import numpy as np
import warnings
import calendar
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# ===== CONFIG — update your own dir =====
daily_dir   = "/share/air-5/ishi/Fused_xch4"
mean_dir    = "/share/air-5/ishi/Fused_xch4_monthly_mean"
YEAR        = 2021
MONTH       = 1
DAY         = 1
VMIN, VMAX  = 1800, 1940
CMAP        = "jet"

# ========================================================
# HELPER FUNCTIONS
# ========================================================

def load_daily_file(daily_dir, year, month):
    fpath = os.path.join(daily_dir, str(year), f"fused_xch4_{year}{month:02d}.h5")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"File not found: {fpath}")
    return fpath

def load_mean_file(mean_dir, year, month):
    fpath = os.path.join(mean_dir, str(year), f"fused_xch4_{year}{month:02d}_mean.h5")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"File not found: {fpath}")
    return fpath

def load_day(fpath, year, month, day):
    day_key = f"{year}{month:02d}{day:02d}"
    with h5py.File(fpath, "r") as f:
        if day_key not in f:
            raise KeyError(f"Day {day_key} not found in file.")
        lat        = f["lat"][:]
        lon        = f["lon"][:]
        fused      = f[f"{day_key}/fused_xch4"][:]
        tropomi_bc = f[f"{day_key}/tropomi_bc"][:]
        date       = f[f"{day_key}/date"][()].decode()
    return lat, lon, fused, tropomi_bc, date

def load_monthly_mean(fpath):
    with h5py.File(fpath, "r") as f:
        lat            = f["lat"][:]
        lon            = f["lon"][:]
        fused_mean     = f["fused_xch4"][:]
        tropomi_mean   = f["tropomi_bc"][:]
        n_days_fused   = f["n_days_fused"][:]
        n_days_tropomi = f["n_days_tropomi"][:]
        yr             = int(f["year"][()])
        mo             = int(f["month"][()])
    return lat, lon, fused_mean, tropomi_mean, n_days_fused, n_days_tropomi, yr, mo

def nearest_pixel(lat, lon, target_lat, target_lon):
    lat_idx = np.argmin(np.abs(lat - target_lat))
    lon_idx = np.argmin(np.abs(lon - target_lon))
    return lat_idx, lon_idx

def plot_comparison(fused, tropomi, lat, lon, title,
                    lon_min=None, lon_max=None, lat_min=None, lat_max=None):
    """Plot fused vs TROPOMI side by side."""
    if lon_min is not None:
        lat_mask = (lat >= lat_min) & (lat <= lat_max)
        lon_mask = (lon >= lon_min) & (lon <= lon_max)
        fused   = fused[np.ix_(lat_mask, lon_mask)]
        tropomi = tropomi[np.ix_(lat_mask, lon_mask)]
        lat     = lat[lat_mask]
        lon     = lon[lon_mask]
        proj    = ccrs.PlateCarree()
        global_view = False
    else:
        proj        = ccrs.Robinson()
        global_view = True

    lo2d, la2d = np.meshgrid(lon, lat)

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.04], wspace=0.05)
    ax0 = fig.add_subplot(gs[0], projection=proj)
    ax1 = fig.add_subplot(gs[1], projection=proj)
    cax = fig.add_subplot(gs[2])

    for i, (ax, data, label) in enumerate(zip(
            [ax0, ax1],
            [fused, tropomi],
            ["Fused XCH4", "TROPOMI Official BC"])):

        if global_view:
            ax.set_global()
        else:
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        ax.set_facecolor("white")
        ax.add_feature(cfeature.LAND,      color="white", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3, linestyle="--", zorder=2)

        p = ax.pcolormesh(lo2d, la2d, data,
                          transform=ccrs.PlateCarree(),
                          cmap=CMAP, vmin=VMIN, vmax=VMAX,
                          shading="auto", zorder=1)

        gl = ax.gridlines(draw_labels=(i == 0), linewidth=0.3,
                          linestyle="--", color="gray")
        if i == 0:
            gl.top_labels   = False
            gl.right_labels = False
            gl.xformatter   = LONGITUDE_FORMATTER
            gl.yformatter   = LATITUDE_FORMATTER
            gl.xlabel_style = {"size": 9}
            gl.ylabel_style = {"size": 9}

        ax.set_title(label, fontsize=12, fontweight="bold")

    fig.colorbar(p, cax=cax).set_label("XCH4 (ppb)", fontsize=11)
    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
    plt.close()


# ========================================================
# EXAMPLE 1 — Load and map a single day (global)
# ========================================================
print("=" * 50)
print("EXAMPLE 1 — Single day global map")
print("=" * 50)

fpath = load_daily_file(daily_dir, YEAR, MONTH)
lat, lon, fused, tropomi_bc, date = load_day(fpath, YEAR, MONTH, DAY)

print(f"Date              : {date}")
print(f"Fused valid pixels: {np.sum(~np.isnan(fused))}")
print(f"Fused mean XCH4   : {np.nanmean(fused):.2f} ppb")

plot_comparison(fused, tropomi_bc, lat, lon,
                title=f"Fused vs TROPOMI Official BC — {date}")

# ========================================================
# EXAMPLE 2 — Regional map (change bounds as needed)
# ========================================================
print("\n" + "=" * 50)
print("EXAMPLE 2 — Regional map")
print("=" * 50)

REGIONS = {
    "USA":   [-130, -60, 20, 55],
    "India": [65,  100,  5, 40],
    "China": [70,  135, 15, 55],
}

for region, (lon_min, lon_max, lat_min, lat_max) in REGIONS.items():
    plot_comparison(fused, tropomi_bc, lat, lon,
                    title=f"Fused vs TROPOMI Official BC — {region} — {date}",
                    lon_min=lon_min, lon_max=lon_max,
                    lat_min=lat_min, lat_max=lat_max)

# ========================================================
# EXAMPLE 3 — Extract value at a point
# ========================================================
print("\n" + "=" * 50)
print("EXAMPLE 3 — Point extraction")
print("=" * 50)

my_lat, my_lon = 35.5665, 129.3780   # Ulsan, South Korea
lat_idx, lon_idx = nearest_pixel(lat, lon, my_lat, my_lon)

fused_val   = fused[lat_idx, lon_idx]
tropomi_val = tropomi_bc[lat_idx, lon_idx]

print(f"Location         : ({my_lat}N, {my_lon}E)")
print(f"Nearest grid     : ({lat[lat_idx]:.2f}N, {lon[lon_idx]:.2f}E)")
print(f"Fused XCH4       : {fused_val:.2f} ppb" if not np.isnan(fused_val)
      else "Fused XCH4       : NaN (no retrieval this day)")
print(f"TROPOMI BC XCH4  : {tropomi_val:.2f} ppb" if not np.isnan(tropomi_val)
      else "TROPOMI BC XCH4  : NaN (no retrieval this day)")

# ========================================================
# EXAMPLE 4 — Monthly mean map
# ========================================================
print("\n" + "=" * 50)
print("EXAMPLE 4 — Monthly mean map")
print("=" * 50)

mpath = load_mean_file(mean_dir, YEAR, MONTH)
lat_m, lon_m, fused_mean, tropomi_mean, n_days_f, n_days_t, yr, mo = load_monthly_mean(mpath)

print(f"Month             : {calendar.month_name[mo]} {yr}")
print(f"Fused mean XCH4   : {np.nanmean(fused_mean):.2f} ppb")
print(f"TROPOMI mean XCH4 : {np.nanmean(tropomi_mean):.2f} ppb")
print(f"Max valid days    : {n_days_f.max()}")

plot_comparison(fused_mean, tropomi_mean, lat_m, lon_m,
                title=f"Monthly Mean — Fused vs TROPOMI Official BC — {calendar.month_name[mo]} {yr}")


# =======================================================
# EXAMPLE 6 — List available days
# ========================================================
print("\n" + "=" * 50)
print("EXAMPLE 6 — List available days")
print("=" * 50)

with h5py.File(fpath, "r") as f:
    days = sorted([k for k in f.keys() if k.isdigit()])
print(f"Available days in {YEAR}/{MONTH:02d}: {len(days)} days")
print(days)