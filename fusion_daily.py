##"""@yixi"""
##________Date : 2025/09/07
## Daily Fused XCH4 (2020-2023, Full Outputs)
# Fusion priority: GOSAT-2 -> TROPOMI -> GOSAT
# Saves fused_xch4, fused_time, fused_source, fused_alt

import os
import numpy as np
import pandas as pd
import scipy.io as sio
import pyresample
import h5py

# ===== CONFIG =====
YEARS     = [2020, 2021, 2022, 2023]
path_data = "/share/air-3/XCH4_Estimation/data/"

path_gosat  = os.path.join(path_data, "bias_correction/GOSAT/GOSAT_XGB_BC_G2_XCH42020to2023.csv")
path_gosat2 = os.path.join(path_data, "bias_correction/GOSAT2/RF/GOSAT2_RF_BiasCorrected_XCH42020to2023.csv")
path_trop   = os.path.join(path_data, "bias_correction/TROPOMI/XGB_G2/monthly/")

# ===== GRID & LANDMASK =====
grid = sio.loadmat(os.path.join(path_data, "grid/grid_TROPOMI_01degree.mat"))
lon_tp, lat_tp = grid["lon_tp"], grid["lat_tp"]

landmask = sio.loadmat(os.path.join(path_data, "grid/grid_TROPOMI_01degree_landmask.mat"))["landmask"].astype(float)
landmask[landmask == 0] = np.nan
total_land_pixels = np.sum(~np.isnan(landmask))

targ_def = pyresample.geometry.GridDefinition(lons=lon_tp, lats=lat_tp)
wf = lambda r: 1 / r**2

# ===== INPUT DATA =====
gosat_df  = pd.read_csv(path_gosat,  parse_dates=["time"])
gosat2_df = pd.read_csv(path_gosat2, parse_dates=["time"])

# ===== Function: process one day =====
def process_day(day_str, path_out):
    year, month = day_str[:4], day_str[4:6]

    # --- Daily subsets ---
    gosat_day  = gosat_df[gosat_df["time"].dt.strftime("%Y%m%d") == day_str]
    gosat2_day = gosat2_df[gosat2_df["time"].dt.strftime("%Y%m%d") == day_str]

    trop_file = os.path.join(path_trop, f"TROPOMI_XGB_BC_{year}_{month}.h5")
    if not os.path.exists(trop_file):
        print(f"  Warning: Missing TROPOMI file for {day_str}")
        return

    trop_df = pd.read_hdf(trop_file, key="data")
    trop_df["time"] = pd.to_datetime(trop_df["time"], errors="coerce")
    trop_day = trop_df[trop_df["time"].dt.strftime("%Y%m%d") == day_str]

    # --- Datasets to resample ---
    maps = {
        "GOSAT2_ML_BC":       (gosat2_day, "pred_xch4"),
        "GOSAT2_Uncorrected":  (gosat2_day, "xch4"),
        "GOSAT_ML_BC":        (gosat_day,  "pred_xch4"),
        "GOSAT_Official_BC":  (gosat_day,  "xch4_bc"),
        "GOSAT_Uncorrected":  (gosat_day,  "xch4"),
        "TROPOMI_ML_BC":      (trop_day,   "pred_xch4"),
        "TROPOMI_Official_BC":(trop_day,   "xch4_bc"),
        "TROPOMI_Uncorrected":(trop_day,   "xch4"),
    }

    resampled = {k: np.full(lon_tp.shape, np.nan) for k in maps}
    res_time  = {k: np.full(lon_tp.shape, np.nan) for k in maps}
    res_alt   = {k: np.full(lon_tp.shape, np.nan) for k in maps}

    fused_xch4 = np.full(lon_tp.shape, np.nan)
    fused_time = np.full(lon_tp.shape, np.nan)
    fused_src  = np.full(lon_tp.shape, np.nan)
    fused_alt  = np.full(lon_tp.shape, np.nan)

    # --- Resample all ---
    for name, (df_in, col) in maps.items():
        if df_in.empty:
            continue

        orig_def = pyresample.geometry.SwathDefinition(
            lons=df_in["lon"].values,
            lats=df_in["lat"].values
        )
        vals  = df_in[col].values
        times = df_in["time"].values.astype("datetime64[s]").astype(float)
        alt   = df_in["alt"].values if "alt" in df_in.columns else np.full_like(vals, np.nan)

        resampled[name] = pyresample.kd_tree.resample_custom(
            orig_def, vals, targ_def,
            radius_of_influence=10000, neighbours=1,
            weight_funcs=wf, fill_value=np.nan
        )
        res_time[name] = pyresample.kd_tree.resample_custom(
            orig_def, times, targ_def,
            radius_of_influence=10000, neighbours=1,
            weight_funcs=wf, fill_value=np.nan
        )
        res_alt[name] = pyresample.kd_tree.resample_custom(
            orig_def, alt, targ_def,
            radius_of_influence=10000, neighbours=1,
            weight_funcs=wf, fill_value=np.nan
        )

    # --- Fusion priority ---
    priority = {"GOSAT2_ML_BC": 2, "TROPOMI_ML_BC": 3, "GOSAT_ML_BC": 1}
    for k in ["GOSAT2_ML_BC", "TROPOMI_ML_BC", "GOSAT_ML_BC"]:
        mask = np.isnan(fused_xch4) & ~np.isnan(resampled[k])
        fused_xch4[mask] = resampled[k][mask]
        fused_time[mask] = res_time[k][mask]
        fused_alt[mask]  = res_alt[k][mask]
        fused_src[mask]  = priority[k]

    # --- Apply landmask ---
    fused_xch4 = np.where(~np.isnan(landmask), fused_xch4, np.nan)
    fused_time = np.where(~np.isnan(landmask), fused_time, np.nan)
    fused_alt  = np.where(~np.isnan(landmask), fused_alt, np.nan)
    fused_src  = np.where(~np.isnan(landmask), fused_src, np.nan)
    for k in maps:
        resampled[k] = np.where(~np.isnan(landmask), resampled[k], np.nan)

    # --- Coverage ---
    coverage = {f"coverage_{k}": np.sum(~np.isnan(v)) / total_land_pixels * 100
                for k, v in resampled.items()}
    coverage["coverage_fused"] = np.sum(~np.isnan(fused_xch4)) / total_land_pixels * 100

    # --- Save ---
    out_file = os.path.join(path_out, f"fused_xch4_{day_str}.h5")
    with h5py.File(out_file, "w") as f:
        f.create_dataset("fused_xch4",   data=fused_xch4)
        f.create_dataset("fused_time",   data=fused_time)
        f.create_dataset("fused_alt",    data=fused_alt)
        f.create_dataset("fused_source", data=fused_src)
        for k in maps:
            f.create_dataset(f"{k}_xch4", data=resampled[k])
        for k, v in coverage.items():
            f.create_dataset(k, data=np.array(v))
        f.create_dataset("lon_tp", data=lon_tp)
        f.create_dataset("lat_tp", data=lat_tp)
        f.create_dataset("date",   data=np.bytes_(day_str))

    print(f"  ✅ {day_str} | Fused Coverage: {coverage['coverage_fused']:.2f}%")

# ===== RUN LOOP: 2020-2023 =====
for YEAR in YEARS:
    path_out = os.path.join(path_data, f"Fused_XCH4_{YEAR}")
    os.makedirs(path_out, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing {YEAR}")
    print(f"{'='*60}")

    for date in pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31"):
        process_day(date.strftime("%Y%m%d"), path_out)
