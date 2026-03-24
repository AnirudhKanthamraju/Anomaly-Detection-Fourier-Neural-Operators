# Anomaly Detection Pipeline — `Anomaly_etl_v2`

A Python ETL pipeline for **Dual-Duct Air Handling Unit (DDAHU)** fault detection. The pipeline loads HVAC sensor time-series data, segments it by operational mode, and produces labelled kernel objects (sliding windows of feature dictionaries) ready for downstream anomaly detection models.

---

## 📁 Project Structure
```
Anomaly_etl_v2/
│
├── main.py                 # Entry point: orchestrates loading, segmentation, kernelisation
├── create_dataset.py       # `kernel` class — wraps a feature snapshot with its label and source
├── dataset_models.py       # `hvac_dataset` class — loads a CSV and auto-segments on init
├── data_loaders.py         # Cache-first CSV loader and datetime column expander
├── data_transformers.py    # `segment_loaded_data` and `kernalise_segment` / `apply_kernalisation`
│
│ ── IN DEVELOPMENT ──
├── naive_models.py         # Classical Models (One Class SVM , Fast MCD) for baselines
├── deep_models.py          # Deep Algorithms ( Neural Operators , Deep SVDD)
├── requirements.txt        # Python dependencies
├── setup.bat               # One-click venv setup (Command Prompt)
│
│ ── NICE TO HAVE ──
├── Dockerfile
├── docker-compose.yml
└── .dockerignore
│
│ ── GENERATED LOCALLY (not in Git) ──
├── cache/                  # Pickle files, auto-generated on first run
├── Dataset/                # Raw CSVs from LBNL (download separately)
└── .venv/                  # Virtual environment
```

---

## 🔧 Pipeline Overview

### 1. Data Loading — `data_loaders.py`

**`load_dataset(file_name)`** — loads a single CSV with a cache-first strategy:
- Checks `cache/<file>.pkl` first; falls back to the raw CSV if not found
- Writes a `.pkl` to cache after parsing a CSV for the first time
- Automatically expands a `Datetime` column into `Day`, `Month`, `Year`, `Hour`, `Minute` columns via `write_datetime_columns()`

**`load_anomaly_datasets(base_path, use_cache)`** — batch-loads every `.csv` in a directory using the same cache-first strategy, returning `Dict[filename, DataFrame]`.

### 2. Segmentation — `data_transformers.py`

**`segment_loaded_data(DataSet)`** — splits a loaded DataFrame by the `SYS_CTL` column (operational mode: Occupied / Setback / Unoccupied / Emergency) into a nested dictionary:
```
{
  "<SYS_CTL_value>": {
      "external_state":        DataFrame  # Time, outside air conditions
      "control_system_state":  DataFrame  # Setpoints, damper demands, valve demands
      "system_state":          DataFrame  # Sensor readings — temperatures, flows, pressures
  }
}
```

Raises `ValueError` if any expected columns are missing or if the total column count doesn't match the defined state-space (8 + 26 + 84 = 118 columns).

### 3. Kernelisation — `data_transformers.py`

**`kernalise_segment(state_space_representation, kernal_type, num_kernels, mean_kernel_size, st_dev_kernel_size)`**

Generates a list of kernel *snapshots* — each snapshot is a `Dict[state_name, DataFrame]` with rows drawn from the segment. Kernel sizes follow a **normal distribution** parameterised by `mean_kernel_size` and `st_dev_kernel_size`.

Three extraction modes:

| Mode | Description |
|---|---|
| `sliding_window` | Overlapping windows; random start index per kernel |
| `listed_sampling` | Sequential, non-overlapping windows |
| `random_sampling` | Shuffled, non-overlapping windows |

For non-overlapping modes, kernels whose sizes would exceed available rows are trimmed starting from the most extreme value (furthest from the mean), preserving the distributional shape.

### 4. Kernel Objects — `create_dataset.py`

**`kernel(features, anomaly, source)`** — a lightweight data class:

| Attribute | Type | Description |
|---|---|---|
| `features` | `Dict[str, DataFrame]` | State-space snapshot from kernelisation |
| `label` | `int` | `0` = normal, `1` = faulty |
| `source` | `str` | Originating file name for traceability |
| `valid` | `bool` | `True` if all feature DataFrames share the same row count |

### 5. HVAC Dataset Class — `dataset_models.py`

**`hvac_dataset(file_name)`** — convenience wrapper that calls `load_dataset` and `segment_loaded_data` on init:
```python
fault_free = hvac_dataset('DualDuct_FaultFree.csv')
fault_free.data      # raw DataFrame
fault_free.segments  # segmented state-space dict
```

### 6. Model Architecture — `models.py` *(in development)*

Defines the core **Fourier Neural Operator (FNO)** building blocks for anomaly classification:

- **`SpectralConv2d`** — applies `rfft2`, multiplies the lower Fourier modes by learned complex weights, then `irfft2` back to physical space
- **`FourierLayer`** — combines the spectral path (`SpectralConv2d`) with a local `1×1` convolution path; outputs are summed and passed through `GELU` activation

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.11+
- Git

### One-Liner Setup

**Command Prompt:**
```cmd
cd Anomaly_etl_v2 && setup.bat
```

The script:
1. Checks Python is installed and on `PATH`
2. Creates `.venv/`
3. Upgrades `pip` and installs `requirements.txt`
4. Prints the activation command

**Activate and run:**
```cmd
.venv\Scripts\activate
python main.py
```

---

## ⚡ Caching

The pipeline uses a **cache-first** strategy. The `cache/` folder is not stored in Git.

### First-time setup on a new machine:
1. Download the raw dataset from [LBNL Buildings FDD](https://faultdetection.lbl.gov/dataset/simulated-dd-ahu-dataset)
2. Place CSVs at: `Dataset/LBNL_FDD_Data_Sets_DDAHU/`
3. Run `python data_loaders.py` — CSVs are parsed and saved as `.pkl` files in `cache/`
4. All subsequent runs load directly from `.pkl` (significantly faster)

---

## 📋 Example Usage
```python
from dataset_models import hvac_dataset
from data_transformers import kernalise_segment
from create_dataset import kernel

# 1. Load and segment
dataset = hvac_dataset('DualDuct_FaultFree.csv')

# 2. Kernelise each operational mode
all_kernels = []
for mode, state_space in dataset.segments.items():
    snapshots = kernalise_segment(
        state_space_representation=state_space,
        kernal_type='sliding_window',
        num_kernels=5000,
        mean_kernel_size=50,
        st_dev_kernel_size=10
    )
    for features in snapshots:
        all_kernels.append(kernel(features=features, anomoly=0, source=f"FaultFree_{mode}"))

print(f"Total kernels: {len(all_kernels)}")
print(f"Feature keys:  {list(all_kernels[0].features.keys())}")
```

---

## 📊 Dataset Reference

| Field | Details |
|---|---|
| **Source** | Lawrence Berkeley National Laboratory (LBNL) |
| **Dataset** | LBNL FDD Data Sets: Dual-Duct Air Handling Unit (DDAHU) |
| **Access** | [faultdetection.lbl.gov](https://faultdetection.lbl.gov/dataset/simulated-dd-ahu-dataset) |
| **Citation** | Fernandez, N., Katipamula, S., & Wang, W. (LBNL) |
| **Format** | CSV — HVAC sensor time-series (temperature, pressure, flow, damper positions) |

---

## 🔧 Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computation and random sampling |
| `scikit-learn` | OCSVM, MCD, StandardScaler *(planned)* |
| `torch` | Fourier Neural Operator implementation |
| `tensorflow` | Additional model support *(planned)* |
| `neuraloperator` | Neural operator utilities *(planned)* |
| `matplotlib` / `seaborn` | Visualisation *(planned)* |
| `scipy` | Statistical scoring *(planned)* |