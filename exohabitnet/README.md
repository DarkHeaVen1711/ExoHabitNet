# ExoHabitNet 🪐
### Exoplanet Habitability Classification via NASA Kepler Light-Curve Analysis

**Combined report:** `reports/combined_report.md` — consolidated experiments and reproduction steps

> **Course:** ECSCI24305 | **Phase:** 2 & 3

---

## Project Structure

```
exohabitnet/
├── docs/
│   └── dataset_architecture.md     ← Task 1: Feature definitions, labels, strategy
├── scripts/
│   ├── collect_kepler_data.py      ← Task 2: Automated 500+ sample collection
│   └── preprocessing_pipeline.py  ← Task 3: Clean → Normalize → Phase-Fold → EDA
├── data/
│   ├── raw_fits/                   ← Downloaded .fits light curves (auto-created)
│   ├── processed/                  ← Preprocessed fixed-length sequences
│   ├── data_collection_log_template.csv
│   └── data_collection_log.csv     ← Auto-generated during collection
├── reports/
│   └── eda_charts/                 ← Chart 1, 2, 3 PNG outputs
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect 500+ Kepler light curves
python scripts/collect_kepler_data.py

# 3. Preprocess + split + augment train + generate EDA charts
python scripts/preprocessing_pipeline.py

# 4. Train the CNN model
python scripts/train.py

# 5. Evaluate on holdout test set
python scripts/evaluate.py
```

## Execution Steps (Detailed)

Run all commands from the `exohabitnet/` folder.

### 1) Setup Environment

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional verification:

```bash
python verify_status.py
```

### 2) Collect Kepler Light Curves

```bash
python scripts/collect_kepler_data.py
```

What this does:
- Queries NASA Exoplanet Archive KOI data.
- Assigns labels (`HABITABLE`, `NON_HABITABLE`, `FALSE_POSITIVE`).
- Downloads and stitches Kepler light-curve FITS files.

Main outputs:
- `data/raw_fits/` (downloaded FITS files)
- `data/data_collection_log.csv` (collection status and metadata)
- `data/logs/` (collection logs)

Quick check:

```bash
python check_status.py
```

### 3) Preprocess, Split, and Build Model Inputs

```bash
python scripts/preprocessing_pipeline.py
```

What this does:
- Cleans light curves, normalizes flux, phase-folds, and bins to fixed length.
- Creates a real-only stratified train/test split.
- Applies augmentation to the training split only.
- Generates EDA charts.

Main outputs:
- `data/processed_dataset.csv`
- `data/train_dataset.csv`
- `data/test_dataset.csv`
- `reports/eda_charts/`

### 4) Train ExoHabitNet CNN

```bash
python scripts/train.py
```

What this does:
- Loads `data/train_dataset.csv`.
- Splits into train/validation.
- Trains with class-weighted loss, LR scheduler, and early stopping.
- Saves the best checkpoint.

Main outputs:
- `models/checkpoints/best_model.pth`
- `runs/exohabitnet_experiment/` (TensorBoard logs)

Monitor training (optional):

```bash
tensorboard --logdir runs/exohabitnet_experiment
```

### 5) Evaluate on Holdout Test Set

```bash
python scripts/evaluate.py
```

What this does:
- Loads `models/checkpoints/best_model.pth`.
- Evaluates on `data/test_dataset.csv` (real-only holdout split).
- Produces metrics and visual reports.

Main outputs:
- `reports/model_performance.json`
- `reports/classification_report.md`
- `reports/confusion_matrix.png`
- `reports/roc_curves.png`
- `reports/sample_predictions/`

## Recommended End-to-End Command Order

```bash
pip install -r requirements.txt
python scripts/collect_kepler_data.py
python scripts/preprocessing_pipeline.py
python scripts/train.py
python scripts/evaluate.py
```

## Labels

| ID | Class | Description |
|----|-------|-------------|
| 0  | `HABITABLE` | Confirmed planet in Habitable Zone, rocky (Rp ≤ 1.6 R⊕) |
| 1  | `NON_HABITABLE` | Confirmed planet, outside HZ or gas giant |
| 2  | `FALSE_POSITIVE` | Eclipsing binary, background contamination, or systematic |

## Data Source
- **Mission:** NASA Kepler (DR25)
- **API:** NASA MAST via `lightkurve`
- **Metadata:** NASA Exoplanet Archive KOI Cumulative Table (TAP/ADQL)
- **HZ Model:** Kopparapu et al. (2013)

## Current Workflow Notes
- Preprocessing now splits real samples into train/test before any augmentation is applied.
- Synthetic samples are generated only from the training split.
- The test split remains real-only to keep evaluation honest and avoid inflated accuracy.
- Generated files are `data/processed_dataset.csv`, `data/train_dataset.csv`, and `data/test_dataset.csv`.
