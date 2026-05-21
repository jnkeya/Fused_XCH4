# Fused_XCH4

**Global Daily Fused XCH4 Dataset (2020–2023): Multi-Sensor Integration of GOSAT-2, TROPOMI, and GOSAT at 0.1° Resolution**

---

## Overview

This repository provides code and usage examples for the first global fusion of GOSAT, GOSAT-2, and TROPOMI satellite retrievals, generating a globally consistent daily 0.1° land-only XCH4 product for 2020–2023.

Accurate global monitoring of atmospheric methane (CH4) is essential for tracking progress toward climate mitigation targets such as the Global Methane Pledge (GMP). Ground-based measurement networks are too sparse to provide sufficient spatial coverage, while satellite-derived retrievals are hindered by systematic biases and uncertainties. This dataset addresses these limitations through a three-step machine learning framework:

1. Sensor-specific bias correction using TCCON ground-based observations
2. Cross-sensor harmonization to GOSAT-2 as the reference sensor
3. Priority-based fusion: GOSAT-2 → TROPOMI → GOSAT

Independent validation at withheld TCCON stations demonstrates robust performance (R² = 0.81, RMSE = 10.78 ppb), outperforming standard and operational bias-corrected satellite products.

---


## Dataset

| Version | Zenodo DOI | Notes |
|---------|------------|-------|
| v2 (current) | https://doi.org/10.5281/zenodo.20304047 | Recommended — full variables + CF attributes |
| v1 | https://doi.org/10.5281/zenodo.18714293 | Legacy — fused_xch4 only |

---

## Technical Details

| Parameter | Value |
|-----------|-------|
| Spatial resolution | 0.1° × 0.1° |
| Spatial coverage | 60°S–80°N, global land |
| Temporal coverage | 2020–2023 (daily) |
| Units | ppb (parts per billion) |
| Satellites | GOSAT, GOSAT-2, TROPOMI |
| Fusion priority | GOSAT-2 → TROPOMI → GOSAT |
| Bias correction | ML-based (XGBoost/Random Forest), calibrated against TCCON |
| Validation | R² = 0.81, RMSE = 10.78 ppb (independent TCCON stations) |
| File format | HDF5 (.h5), gzip compression (level 4) |
| Grid reference | WGS84 |

---

## File Structure (v2)

48 monthly HDF5 files (12 months × 4 years):

```
fused_xch4_202001.h5   ← January 2020
fused_xch4_202002.h5   ← February 2020
...
fused_xch4_202312.h5   ← December 2023
```

Each file contains:

```
fused_xch4_YYYYMM.h5
├── lat                         (1400,)        degrees_north
├── lon                         (3600,)        degrees_east
└── YYYYMMDD/                   (one group per day)
    ├── date                    scalar          YYYYMMDD string
    ├── fused_xch4              (1400, 3600)   ppb — daily fused XCH4
    ├── gosat2_ml_bc_xch4       (1400, 3600)   ppb — ML bias-corrected GOSAT-2
    ├── gosat_harmonized_xch4   (1400, 3600)   ppb — GOSAT harmonized to GOSAT-2 scale
    ├── tropomi_harmonized_xch4 (1400, 3600)   ppb — TROPOMI harmonized to GOSAT-2 scale
    └── selected_sensor_id      (1400, 3600)   int8 — 1=GOSAT, 2=GOSAT-2, 3=TROPOMI, -127=ocean/missing
```

---

## Variables

| Variable | Dimensions | Units | Description |
|----------|------------|-------|-------------|
| `fused_xch4` | (1400, 3600) | ppb | Daily fused XCH4 |
| `gosat2_ml_bc_xch4` | (1400, 3600) | ppb | ML bias-corrected GOSAT-2 XCH4 |
| `gosat_harmonized_xch4` | (1400, 3600) | ppb | GOSAT XCH4 harmonized to GOSAT-2 scale |
| `tropomi_harmonized_xch4` | (1400, 3600) | ppb | TROPOMI XCH4 harmonized to GOSAT-2 scale |
| `selected_sensor_id` | (1400, 3600) | — | Sensor ID per pixel (1=GOSAT, 2=GOSAT-2, 3=TROPOMI; -127=ocean/no retrieval) |
| `lat` | (1400,) | ° | 1D latitude array (60°S to 80°N) |
| `lon` | (3600,) | ° | 1D longitude array (180°W to 180°E) |
| `date` | scalar | — | Date string YYYYMMDD |

> NaN values in float variables indicate ocean pixels or land pixels with no valid satellite retrieval for that day.

---

## Usage

### Requirements

```bash
pip install h5py numpy matplotlib cartopy
```

### Quick Start

```python
import h5py
import numpy as np

with h5py.File("fused_xch4_202101.h5", "r") as f:
    lat   = f["lat"][:]
    lon   = f["lon"][:]
    xch4  = f["20210101/fused_xch4"][:]
    date  = f["20210101/date"][()].decode()

print(f"Date: {date}")
print(f"Mean XCH4: {np.nanmean(xch4):.2f} ppb")
```

### Load All Variables for One Day

```python
with h5py.File("fused_xch4_202101.h5", "r") as f:
    grp = f["20210101"]
    fused    = grp["fused_xch4"][:]
    gosat2   = grp["gosat2_ml_bc_xch4"][:]
    gosat    = grp["gosat_harmonized_xch4"][:]
    tropomi  = grp["tropomi_harmonized_xch4"][:]
    sensor   = grp["selected_sensor_id"][:]
```

### Extract Value at a Location

```python
my_lat, my_lon = 41.85, -87.65   # Chicago, USA
lat_idx = np.argmin(np.abs(lat - my_lat))
lon_idx = np.argmin(np.abs(lon - my_lon))
value   = xch4[lat_idx, lon_idx]
print(f"XCH4: {value:.2f} ppb")
```

See `example_usage_v2.py` for complete examples including global and regional maps, point extraction, and time series.

---

## Repository Structure

```
Fused_XCH4/
├── README.md                — this file
├── CHANGELOG.md             — v1 → v2 variable name mapping and changes
├── example_usage_v2.py      — complete usage examples (v2, recommended)
├── example_usage_v1.py      — usage examples for v1
└── Fusion_pipeline/         — fusion processing pipeline scripts
```

---

## Citation

If you use this dataset, please cite:

> [Paper citation — to be added upon publication]
>
> v2 Dataset: https://doi.org/10.5281/zenodo.20304047
> v1 Dataset: https://doi.org/10.5281/zenodo.18714293

---

## License

This dataset is released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

---

## Contact

For questions, please contact jnkeya@unist.ac.kr. Additional processing scripts available upon request.
