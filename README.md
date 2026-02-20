# Fused_XCH4

This repository provides code and usage examples for the first global fusion of GOSAT, GOSAT-2, and TROPOMI satellite retrievals, generating a globally consistent daily 0.1° land-only XCH4 product for 2020–2023.

Accurate global monitoring of atmospheric methane (CH4) is essential for tracking progress toward climate mitigation targets such as the Global Methane Pledge (GMP). Ground-based measurement networks are too sparse to provide sufficient spatial coverage, while satellite-derived retrievals are hindered by systematic biases and uncertainties. This dataset addresses these limitations through a three-step machine learning framework:

1. Sensor-specific bias correction using TCCON ground-based observations
2. Cross-sensor harmonization to GOSAT-2 as the reference sensor
3. Priority-based fusion: GOSAT-2 → TROPOMI → GOSAT
Independent validation at withheld TCCON stations demonstrates robust performance (R² = 0.81, RMSE = 10.78 ppb), outperforming standard and operational bias-corrected satellite products.

### Technical Details

| Parameter | Value |
|-----------|-------|
| Spatial resolution | 0.1° × 0.1° |
| Spatial coverage | 70°S–70°N, global land |
| Temporal coverage | 2020–2023 (daily) |
| Units | ppb (parts per billion) |
| Satellites | GOSAT, GOSAT-2, TROPOMI |
| Fusion priority | GOSAT-2 → TROPOMI → GOSAT |
| Bias correction | ML-based (XGBoost/Random Forest), calibrated against TCCON |
| Validation | R² = 0.81, RMSE = 10.78 ppb |
| File format | HDF5 (.h5) with gzip compression |
| Grid reference | WGS84 |
### File Structure

Files are organized by year, with one HDF5 file per month:

```
Fused_XCH4/
├── 2020/
│   ├── fused_xch4_202001.h5
│   ├── fused_xch4_202002.h5
│   └── ...
├── 2021/
├── 2022/
└── 2023/
```

Each monthly HDF5 file contains:

```
fused_xch4_YYYYMM.h5
├── lat              (1400,)       1D latitude array  (70°S to 70°N)
├── lon              (3600,)       1D longitude array (180°W to 180°E)
├── YYYYMMDD/                      one group per day
│   ├── fused_xch4  (1400, 3600)  daily XCH4 in ppb
│   └── date        scalar         date string (YYYYMMDD)
├── YYYYMMDD/
│   └── ...
```

> **Note:** NaN values indicate ocean pixels or land pixels with no valid satellite retrieval for that day.

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

# Load a single day
with h5py.File("fused_xch4_202101.h5", "r") as f:
    lat  = f["lat"][:]
    lon  = f["lon"][:]
    xch4 = f["20210101/fused_xch4"][:]
    date = f["20210101/date"][()].decode()

print(f"Date: {date}")
print(f"Mean XCH4: {np.nanmean(xch4):.2f} ppb")
```

### Extract Value at a Location

```python
# Nearest grid pixel to your location
my_lat, my_lon = 28.6, 77.2   # New Delhi

lat_idx = np.argmin(np.abs(lat - my_lat))
lon_idx = np.argmin(np.abs(lon - my_lon))
value   = xch4[lat_idx, lon_idx]

print(f"XCH4 at ({my_lat}N, {my_lon}E): {value:.2f} ppb")
```

### Extract Regional Subset

```python
# Crop to a region
lat_mask = (lat >= 5) & (lat <= 40)
lon_mask = (lon >= 65) & (lon <= 100)
xch4_sub = xch4[np.ix_(lat_mask, lon_mask)]
```

See `example_usage.py` for complete examples including global mapping, regional plots, monthly mean computation, and time series extraction.

---

## Repository Structure

```
Fused_XCH4/
├── README.md              — this file
├── example_usage.py       — complete usage examples
└── processing/
    ├── fusion_daily.py    — daily fusion script
    └── convert_monthly.py — conversion to monthly HDF5
```

---


