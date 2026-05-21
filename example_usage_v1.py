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
daily_dir  = "/share/air-5/ishi/Fused_xch4"
mean_dir   = "/share/air-5/ishi/Fused_xch4_monthly_mean"
YEAR       = 2021
MONTH      = 1
DAY        = 1
VMIN, VMAX = 1800, 1940
CMAP       = "jet"

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
        lat   = f["lat"][:]
        lon   = f["lon"][:]
        xch4  = f[f"{day_key}/fused_xch4"][:]
        date  = f[f"{day_key}/date"][()].decode()
    return lat, lon, xch4, date

def load_monthly_mean(fpath):
    with h5py.File(fpath, "r") as f:
        lat   = f["lat"][:]
        lon   = f["lon"][:]
        xch4  = f["fused_xch4"][:]
        ndays = f["n_days"][:]
        yr    = int(f["year"][()])
        mo    = int(f["month"][()])
    return lat, lon, xch4, ndays, yr, mo

def nearest_pixel(lat, lon, target_lat, target_lon):
    lat_idx = np.argmin(np.abs(lat - target_lat))
    lon_idx = np.argmin(np.abs(lon - target_lon))
    return lat_idx, lon_idx

def plot_map(xch4, lat, lon, title,
             lon_min=None, lon_max=None, lat_min=None, lat_max=None):
    if lon_min is not None:
        lat_mask = (lat >= lat_min) & (lat <= lat_max)
        lon_mask = (lon >= lon_min) & (lon <= lon_max)
        xch4 = xch4[np.ix_(lat_mask, lon_mask)]
        lat  = lat[lat_mask]
        lon  = lon[lon_mask]
        proj = ccrs.PlateCarree()
        global_view = False
    else:
        proj        = ccrs.Robinson()
        global_view = True

    lo2d, la2d = np.meshgrid(lon, lat)

    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": proj})

    if global_view:
        ax.set_global()
    else:
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax.set_facecolor("white")
    ax.add_feature(cfeature.LAND,      color="white", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, linestyle="--", zorder=2)

    p = ax.pcolormesh(lo2d, la2d, xch4,
                      transform=ccrs.PlateCarree(),
                      cmap=CMAP, vmin=VMIN, vmax=VMAX,
                      shading="auto", zorder=1)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle="--", color="gray")
    gl.top_labels   = False
    gl.right_labels = False
    gl.xformatter   = LONGITUDE_FORMATTER
    gl.yformatter   = LATITUDE_FORMATTER

    cbar = fig.colorbar(p, ax=ax, orientation="vertical", pad=0.02, fraction=0.03)
    cbar.set_label("XCH4 (ppb)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
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
lat, lon, xch4, date = load_day(fpath, YEAR, MONTH, DAY)

print(f"Date              : {date}")
print(f"Valid pixels      : {np.sum(~np.isnan(xch4))}")
print(f"Mean XCH4         : {np.nanmean(xch4):.2f} ppb")

plot_map(xch4, lat, lon, title=f"Fused XCH4 — {date}")

# ========================================================
# EXAMPLE 2 — Regional maps
# ========================================================
print("\n" + "=" * 50)
print("EXAMPLE 2 — Regional maps")
print("=" * 50)

REGIONS = {
    "USA":   [-130, -60, 20, 55],
    "India": [65,  100,  5, 40],
    "China": [70,  135, 15, 55],
}

for region, (lon_min, lon_max, lat_min, lat_max) in REGIONS.items():
    plot_map(xch4, lat, lon,
             title=f"Fused XCH4 — {region} — {date}",
             lon_min=lon_min, lon_max=lon_max,
             lat_min=lat_min, lat_max=lat_max)

# ========================================================
# EXAMPLE 3 — Extract value at a point
# ========================================================
print("\n" + "=" * 50)
print("EXAMPLE 3 — Point extraction")
print("=" * 50)

my_lat, my_lon = 41.85, -87.65   # Chicago, USA
lat_idx, lon_idx = nearest_pixel(lat, lon, my_lat, my_lon)
value = xch4[lat_idx, lon_idx]

print(f"Location          : ({my_lat}N, {my_lon}E)")
print(f"Nearest grid      : ({lat[lat_idx]:.2f}N, {lon[lon_idx]:.2f}E)")
print(f"Fused XCH4        : {value:.2f} ppb" if not np.isnan(value)
      else "Fused XCH4        : NaN (no retrieval this day)")

# ========================================================
# EXAMPLE 4 — Monthly mean map
# ========================================================
print("\n" + "=" * 50)
print("EXAMPLE 4 — Monthly mean map")
print("=" * 50)

mpath = load_mean_file(mean_dir, YEAR, MONTH)
lat_m, lon_m, xch4_mean, ndays, yr, mo = load_monthly_mean(mpath)

print(f"Month             : {calendar.month_name[mo]} {yr}")
print(f"Mean XCH4         : {np.nanmean(xch4_mean):.2f} ppb")
print(f"Max valid days    : {ndays.max()}")

plot_map(xch4_mean, lat_m, lon_m,
         title=f"Monthly Mean Fused XCH4 — {calendar.month_name[mo]} {yr}")

# ========================================================
# EXAMPLE 5 — Time series at a point (full month)
# ========================================================
print("\n" + "=" * 50)
print("EXAMPLE 5 — Time series at a point")
print("=" * 50)

my_lat, my_lon = 41.85, -87.65   # Chicago, USA
dates_ts, values_ts = [], []

with h5py.File(fpath, "r") as f:
    day_keys = sorted([k for k in f.keys() if k.isdigit()])
    lat_ts   = f["lat"][:]
    lon_ts   = f["lon"][:]
    li, loi  = nearest_pixel(lat_ts, lon_ts, my_lat, my_lon)
    for dk in day_keys:
        dates_ts.append(dk)
        values_ts.append(float(f[f"{dk}/fused_xch4"][li, loi]))

dates_ts  = np.array(dates_ts)
values_ts = np.array(values_ts)
valid     = ~np.isnan(values_ts)

print(f"Location          : ({my_lat}N, {my_lon}E)")
print(f"Valid days        : {np.sum(valid)} / {len(dates_ts)}")
if np.sum(valid) > 0:
    print(f"Mean XCH4         : {np.nanmean(values_ts[valid]):.2f} ppb")

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(np.where(valid)[0], values_ts[valid], "o-",
        color="steelblue", markersize=5, linewidth=1.5, label="Fused XCH4")
ax.set_xticks(np.where(valid)[0])
ax.set_xticklabels([dates_ts[j][6:] for j in np.where(valid)[0]],
                   rotation=45, fontsize=8)
ax.set_xlabel("Day", fontsize=11)
ax.set_ylabel("XCH4 (ppb)", fontsize=11)
ax.set_title(f"Daily XCH4 at ({my_lat}N, {my_lon}E) — "
             f"{calendar.month_name[MONTH]} {YEAR}", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
plt.close()

# ========================================================
# EXAMPLE 6 — List available days
# ========================================================
print("\n" + "=" * 50)
print("EXAMPLE 6 — List available days")
print("=" * 50)

with h5py.File(fpath, "r") as f:
    days = sorted([k for k in f.keys() if k.isdigit()])
print(f"Available days in {YEAR}/{MONTH:02d}: {len(days)} days")
print(days)
