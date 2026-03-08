# Anomaly Detection Pipeline (`Anomoly_etl_v2`)

A Python-based anomaly detection ETL pipeline for **Dual-Duct Air Handling Unit (DDAHU)** fault detection data. The pipeline loads the pre cached dataset from the sources, applies statistical and machine learning models (OCSVM, MCD) and Deep Neural operators and identifies anomalous operational windows.

---

## 📁 Project Structure

```
Anomoly_etl_v2/
│IN DEVELOPMENT
├── data_loaders.py         # Core data loading logic (cache-first) ( ACTIVE )
├── models.py               # OCSVM, MCD and Deep Neural Operators ( IN DEV)
├── requirements.txt        # Python dependencies ( ACTIVE )
├── setup.ps1               # ⚡ One-click venv setup (PowerShell) ( ACTIVE )
├── setup.bat               # ⚡ One-click venv setup (Command Prompt) ( ACTIVE )
├── .gitignore              # Files excluded from Git ( ACTIVE )
└── README.md               # This file ( ACTIVE )
│NICE TO HAVE
├── Dockerfile              # Container definition
├── docker-compose.yml      # One-command container runner
|__ .dockerignore           # Files excluded from Docker build

```

> **Generated locally (not in Git):**
> - `cache/` — 19GB pickle files, generated on first run
> - `Dataset/` — raw CSVs, download from LBNL (see Dataset Reference below)
> - `.venv/` — virtual environment, created by `setup.ps1` / `setup.bat`

---


## 🚀 Setup & Installation Sequence

### Prerequisites
- [Python 3.11+](https://www.python.org/downloads/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (required for the Docker workflow)
- [Git](https://git-scm.com/)

---

### Option A: Local Python — One-Liner Setup (Recommended)

After cloning, run **one single command** to set everything up automatically:

**PowerShell:**
```powershell
git clone <your-repo-url>; cd Anomoly_etl_v2; powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

**Command Prompt:**
```cmd
git clone <your-repo-url> && cd Anomoly_etl_v2 && setup.bat
```

The script automatically:
1. ✅ Checks Python is installed
2. ✅ Creates the `.venv` virtual environment
3. ✅ Installs all packages from `requirements.txt`
4. ✅ Tells you exactly how to activate and run

**After setup, activate and run:**
```powershell
# PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
python data_loaders.py
```
```cmd
REM Command Prompt
.venv\Scripts\activate
python data_loaders.py
```

---

### Option B: Docker (Containerized)

Use this to run the pipeline in an isolated container — no manual environment setup needed. **Nice to Have.**

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd Anomoly_etl_v2

# 2. Build and run the container
docker-compose up

# To rebuild after code changes:
docker-compose up --build
```

The container will:
1. Use `python:3.11-slim` as base
2. Install all dependencies from `requirements.txt`
3. Copy the `cache/` pickle files into the container
4. Run `python data_loaders.py` automatically

---

## ⚡ How Caching Works

The pipeline uses a **cache-first** loading strategy to avoid redundant CSV parsing.

> The `cache/` folder is **not stored in Git** (19GB — too large). It is auto-generated on first run.

### First-time setup on a new machine:
1. Download the raw dataset from [LBNL Buildings](https://buildings.lbl.gov/cbs/fdd-datasets)
2. Place it inside `Dataset/LBNL_FDD_Data_Sets_DDAHU_all_3/LBNL_FDD_Data_Sets_DDAHU/`
3. Run `python data_loaders.py` → CSVs are parsed and saved as `.pkl` files in `cache/`
4. On all **subsequent runs** → `.pkl` files load directly (much faster, no CSVs needed)

```python
from data_loaders import load_dataset

# Load a single file (reads from cache automatically after first run)
df = load_dataset("DualDuct_FaultFree.csv")
```

---

## 📊 Dataset Reference

**LBNL Fault Detection and Diagnostics (FDD) Data Sets — Dual-Duct AHU**

| Field        | Details |
|---|---|
| **Source**   | Lawrence Berkeley National Laboratory (LBNL) |
| **Dataset**  | LBNL FDD Data Sets: Dual-Duct Air Handling Unit (DDAHU) |
| **Access**   | [Buildings.lbl.gov](https://buildings.lbl.gov/cbs/fdd-datasets) |
| **Citation** | *Fernandez, N., Katipamula, S., & Wang, W. (LBNL). Fault Detection and Diagnostics Data Sets.* |
| **Format**   | CSV files with sensor time-series (temperature, pressure, flow, damper positions) |
| **Use Case** | Identifying anomalous HVAC operating conditions using OCSVM and Minimum Covariance Determinant (MCD) |

---

## 🔧 Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |
| `scikit-learn` | OCSVM, MCD, StandardScaler |
| `matplotlib` | Visualization |
| `seaborn` | Statistical plots |
| `scipy` | Chi-squared scoring |
| `torch` | Deep learning (Fourier Neural Operator models) |
| `tensorflow` | Additional model support |

---

## ⚙️ Setup Scripts

Two automated setup scripts are included to get your environment running with a **single command**. They handle everything: checking Python, creating the virtual environment, and installing all dependencies.

### `setup.ps1` — PowerShell

Run directly from PowerShell without needing to change system security settings:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

**What it does:**
1. Verifies Python 3.11+ is available on `PATH`
2. Creates `.venv/` (skips if it already exists)
3. Upgrades `pip` inside the environment
4. Runs `pip install -r requirements.txt`
5. Prints the exact activation command to use next

---

### `setup.bat` — Command Prompt

For users who prefer the classic Windows Command Prompt:

```cmd
setup.bat
```

**What it does:** Identical steps to `setup.ps1`, but compatible with `cmd.exe`.

---

### Full One-Liner (Clone + Setup)

**PowerShell:**
```powershell
git clone <your-repo-url>; cd Anomoly_etl_v2; powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

**Command Prompt:**
```cmd
git clone <your-repo-url> && cd Anomoly_etl_v2 && setup.bat
```

---

## 🐳 Docker Notes

- The **raw `Dataset/` folder is excluded** from both Git and Docker (too large).
- The `cache/` volume is mounted so any new cache files generated inside the container are saved to your local machine.
- To run interactively inside the container:
  ```bash
  docker-compose run anomaly-pipeline bash
  ```
