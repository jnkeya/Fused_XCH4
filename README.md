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

The dataset is available on Zenodo: **[Zenodo DOI — add link]**

### Technical Details

| Parameter | Value |
|-----------|-------|
| Spatial resolution | 0.1° × 0.1° |
| Spatial coverage | 70°S–70°N, global land |
| Temporal coverage | 2020–2023 (daily and monthly mean) |
| Units | ppb (parts per billion) |
| Satellites | GOSAT, GOSAT-2, TROPOMI |
| Fusion priority | GOSAT-2 → TROPOMI → GOSAT |
| Bias correction | ML-based (XGBoost/Random Forest), calibrated against TCCON |
| Validation | R² = 0.81, RMSE = 10.78 ppb (independent TCCON stations) |
| File format | HDF5 (.h5) with internal gzip compression |
| Grid reference | WGS84 |

---

### File Structure

Two datasets are provided — daily and monthly mean:

**Daily Data** (48 HDF5 files, 12 months × 4 years):
```
fused_xch4_202001.h5   ← January 2020, contains all daily data
fused_xch4_202002.h5   ← February 2020, contains all daily data
...
fused_xch4_202312.h5   ← December 2023, contains all daily data
```

Each daily monthly file contains:
```
fused_xch4_YYYYMM.h5
├── lat              (1400,)       1D latitude array  (70°S to 70°N)
├── lon              (3600,)       1D longitude array (180°W to 180°E)
├── YYYYMMDD/                      one group per day
│   ├── fused_xch4  (1400, 3600)  daily fused XCH4 in ppb
│   ├── tropomi_bc  (1400, 3600)  daily TROPOMI official BC XCH4 in ppb
│   └── date        scalar         date string (YYYYMMDD)
├── YYYYMMDD/
│   └── ...
```

> **Note:** NaN values indicate ocean pixels or land pixels with no valid satellite retrieval for that day.

**Monthly Mean Data** (48 HDF5 files, 12 months × 4 years):
```
fused_xch4_202001_mean.h5   ← January 2020 mean
fused_xch4_202002_mean.h5   ← February 2020 mean
...
fused_xch4_202312_mean.h5   ← December 2023 mean
```

Each monthly mean file contains:
```
fused_xch4_YYYYMM_mean.h5
├── fused_xch4      (1400, 3600)  monthly mean fused XCH4 in ppb
├── tropomi_bc      (1400, 3600)  monthly mean TROPOMI official BC in ppb
├── n_days_fused    (1400, 3600)  number of valid days per pixel (fused)
├── n_days_tropomi  (1400, 3600)  number of valid days per pixel (TROPOMI)
├── lat             (1400,)       1D latitude array
├── lon             (3600,)       1D longitude array
├── year            scalar        year
└── month           scalar        month
```

> **Note:** NaN values indicate ocean pixels or pixels with no valid retrieval for that month.

---

## File Compression

All HDF5 files use internal gzip compression (level 4). Files can be read directly — no manual decompression needed:

```python
import h5py
with h5py.File("fused_xch4_202101.h5", "r") as f:
    xch4 = f["20210101/fused_xch4"][:]  # decompresses automatically
```

Compression reduces file size by ~98% compared to uncompressed storage. Total dataset size for 4 years is ~3.3 GB (daily + monthly mean).

---

## Usage

### Requirements

```bash
pip install h5py numpy matplotlib cartopy
```

### Quick Start — Daily Data

```python
import h5py
import numpy as np

with h5py.File("fused_xch4_202101.h5", "r") as f:
    lat        = f["lat"][:]
    lon        = f["lon"][:]
    fused      = f["20210101/fused_xch4"][:]
    tropomi_bc = f["20210101/tropomi_bc"][:]
    date       = f["20210101/date"][()].decode()

print(f"Date: {date}")
print(f"Fused mean XCH4  : {np.nanmean(fused):.2f} ppb")
print(f"TROPOMI mean XCH4: {np.nanmean(tropomi_bc):.2f} ppb")
```

### Quick Start — Monthly Mean

```python
with h5py.File("fused_xch4_202101_mean.h5", "r") as f:
    lat            = f["lat"][:]
    lon            = f["lon"][:]
    fused_mean     = f["fused_xch4"][:]
    tropomi_mean   = f["tropomi_bc"][:]
    n_days_fused   = f["n_days_fused"][:]
    n_days_tropomi = f["n_days_tropomi"][:]
```

### Extract Value at a Location

```python
# Nearest grid pixel to your location
my_lat, my_lon = 35.57, 129.38   # Ulsan, South Korea

lat_idx = np.argmin(np.abs(lat - my_lat))
lon_idx = np.argmin(np.abs(lon - my_lon))
value   = fused[lat_idx, lon_idx]
print(f"XCH4: {value:.2f} ppb")
```

See `example_usage.py` for complete examples including global mapping, regional plots, and time series extraction.

---

## Repository Structure

```
Fused_XCH4/
├── README.md          — this file
└── example_usage.py   — complete usage examples (load, map, extract, time series)
```

---

## Citation

If you use this dataset, please cite:

> [Paper citation — to be added upon publication]
>
> Dataset: [Zenodo DOI — to be added]

---

## License

This dataset is released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

---

## Contact

For questions, please open a GitHub issue or contact the corresponding author.
