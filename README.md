# ExoHabitNet 🪐
**Exoplanet Habitability Classification via NASA Kepler Light-Curve Analysis**

> **Course:** ECSCI24305 | **Phase:** 2 & 3  
> **Deep Learning Approach:** 1D-CNN for transit signal classification

---

## 🚀 Quick Start for Collaborators

### Current Project Status (March 2026)
**✅ Phases Completed:** 1-3 (Environment Setup → Data Collection → Preprocessing)  
**🔄 Next Phase:** Model Architecture Design (1D-CNN implementation)  
**📊 Dataset Ready:** 653 balanced samples (200 HABITABLE / 222 NON_HABITABLE / 231 FALSE_POSITIVE)

**See [`execution_process.md`](execution_process.md) for detailed progress tracking.**

---

### 1. Clone Repository
```bash
git clone https://github.com/DarkHeaVen1711/ExoHabitNet.git
cd ExoHabitNet
```

### 2. Install Dependencies
```bash
cd exohabitnet
pip install -r requirements.txt
```

**Tested Environment:**
- Python 3.11.9
- All dependencies from requirements.txt installed successfully
- Virtual environment recommended (`.venv/`)

### 3. Download NASA Data
**Important:** Data files are NOT included in this repository (too large for Git). Each collaborator must download their own data from NASA MAST API.

```bash
# This will download 460+ Kepler light curves from NASA
python scripts/collect_kepler_data.py
```

**What this downloads:**
- 231 FALSE_POSITIVE (eclipsing binaries, artifacts)
- 222 NON_HABITABLE confirmed planets
- 7 HABITABLE zone candidates (augmented to 200 during preprocessing)
- Total: 460 samples as `.fits` files (several GB)

**Expected time:** 30-60 minutes depending on internet speed

### 4. Preprocess Data (Already Executed)
```bash
# Clean, normalize, phase-fold, and augment light curves
python scripts/preprocessing_pipeline.py
```

**Output:** `data/processed_dataset.csv` — 653 samples ready for model training

---

## 📊 Project Overview

### Problem Statement
Authenticate and classify exoplanet candidates from false positives using raw transit light-curve data.

### What Has Been Executed (Phases 1-3)

#### ✅ Phase 1: Environment Setup
- Python 3.11.9 virtual environment configured
- All dependencies installed and verified
- NASA MAST API connectivity tested successfully

#### ✅ Phase 2: Data Collection  
- **460 light curves** downloaded from NASA Kepler Mission (DR25)
- Class distribution: 231 FALSE_POSITIVE / 222 NON_HABITABLE / 7 HABITABLE
- **Critical issue identified:** Severe class imbalance (HABITABLE only 1.5%)

#### ✅ Phase 3: Preprocessing Pipeline
- **Cleaning:** Quality flag removal, NaN interpolation, 5σ outlier clipping
- **Normalization:** Z-score normalization for CNN input
- **Phase Folding:** All transits aligned using Kepler's Third Law
- **Binning:** Fixed 1024-timestep sequences for uniform CNN input
- **Augmentation:** HABITABLE class 7→200 samples (28.6x oversampling)
- **Final Dataset:** 653 balanced samples (30.6% / 34.0% / 35.4%)
- **Class Weights:** [1.0883, 0.9805, 0.9423] computed for PyTorch training

**Result:** `data/processed_dataset.csv` ready for model training

**For detailed metrics and verification results, see:**
- [PROGRESS_REPORT.md](exohabitnet/PROGRESS_REPORT.md) — Comprehensive Phase 1-3 summary
- [execution_process.md](execution_process.md) — Progress tracking with completion status

### Data Pipeline
```
NASA Kepler Mission (DR25)
        ↓
MAST API Query (lightkurve)
        ↓
Raw FITS Light Curves (500+ samples)
        ↓
Preprocessing (clean → normalize → phase-fold)
        ↓
Class Balancing (SMOTE + augmentation)
        ↓
1D-CNN Model Training
        ↓
Classification: HABITABLE / NON_HABITABLE / FALSE_POSITIVE
```

### Target Classes
- 🟢 **Class 0 (HABITABLE):** Confirmed rocky planet in habitable zone
- 🔴 **Class 1 (NON_HABITABLE):** Gas giant or orbit too hot/cold
- ⚠️ **Class 2 (FALSE_POSITIVE):** Eclipsing binary or instrument artifact

---

## 📁 Repository Structure

```
ExoHabitNet/
├── README.md                          # This file
├── execution_process.md              # Step-by-step execution guide
├── .gitignore                        # Excludes data files
└── exohabitnet/
    ├── README.md                     # Project details
    ├── requirements.txt              # Python dependencies
    │
    ├── scripts/
    │   ├── collect_kepler_data.py           # ⬇️ Download NASA data
    │   ├── preprocessing_pipeline.py        # 🔧 Clean & augment
    │   ├── balance_dataset.py               # ⚖️ Solve class imbalance
    │   └── test_api.py                      # Test NASA API access
    │
    ├── docs/
    │   ├── dataset_architecture.md          # Feature definitions & labels
    │   └── class_imbalance_solutions.md     # Balancing strategies
    │
    ├── data/
    │   ├── data_collection_log_template.csv # Template (tracked in Git)
    │   ├── raw_fits/                        # ⚠️ NOT IN GIT (download yourself)
    │   ├── processed/                       # ⚠️ NOT IN GIT (generated)
    │   ├── logs/                            # ⚠️ NOT IN GIT (generated)
    │   └── *.csv                            # ⚠️ NOT IN GIT (generated)
    │
    ├── models/
    │   └── checkpoints/                     # ⚠️ NOT IN GIT (trained models)
    │
    └── reports/                             # ⚠️ NOT IN GIT (EDA charts)
```

**Legend:**
- ⬇️ Files you download from NASA
- 🔧 Files generated by scripts
- ⚠️ NOT IN GIT = Ignored by `.gitignore` (collaborators generate locally)

---

## 🔄 Workflow for Collaborators

### Step 1: Environment Setup
```bash
# Verify Python 3.8+
python --version

# Install dependencies
cd exohabitnet
pip install -r requirements.txt

# Test NASA API connection
python scripts/test_api.py
```

### Step 2: Data Collection
```bash
# Download Kepler data from NASA MAST
python scripts/collect_kepler_data.py

# Output:
# - data/raw_fits/HABITABLE/*.fits
# - data/raw_fits/NON_HABITABLE/*.fits
# - data/raw_fits/FALSE_POSITIVE/*.fits
# - data/data_collection_log.csv
```

### Step 3: Preprocessing & Class Balancing
```bash
# Preprocess light curves (clean, normalize, phase-fold)
python scripts/preprocessing_pipeline.py

# Output:
# - data/processed_dataset.csv
# - reports/eda_charts/*.png

# Advanced balancing (recommended)
python scripts/balance_dataset.py --strategy moderate

# Output:
# - data/balanced_dataset.csv (ready for training)
```

### Step 4: Model Training (Future)
```bash
# Train 1D-CNN model
python scripts/train.py --data data/balanced_dataset.csv

# Evaluate
python scripts/evaluate.py
```

---

## 🎯 Key Features

### Class Imbalance Solution
The dataset has severe imbalance (HABITABLE ~1.5%). We solve this with:
1. **Relaxed HZ criteria** to collect more HABITABLE samples
2. **Advanced augmentation** (SMOTE, noise injection, time shifting)
3. **Class-weighted loss** during training

See [`docs/class_imbalance_solutions.md`](exohabitnet/docs/class_imbalance_solutions.md) for details.

### Preprocessing Pipeline
- **Cleaning:** Remove quality-flagged cadences, sigma-clip outliers
- **Normalization:** Z-score (local) and min-max (global) methods
- **Phase Folding:** Align transit events using Kepler's Third Law
- **Binning:** Fixed 1024-timestep sequences for CNN input

### Data Augmentation
- Gaussian noise injection (σ=0.012)
- Circular time shifting (±50 timesteps)
- SMOTE-like interpolation between real samples
- Amplitude scaling (±3%)

---

## 📖 Documentation

- **[execution_process.md](execution_process.md)** — Complete step-by-step guide (63 steps, 9 phases) with progress tracking
- **[PROGRESS_REPORT.md](exohabitnet/PROGRESS_REPORT.md)** — Detailed Phase 1-3 execution summary with metrics
- **[dataset_architecture.md](exohabitnet/docs/dataset_architecture.md)** — Feature definitions, labeling schema, HZ model
- **[class_imbalance_solutions.md](exohabitnet/docs/class_imbalance_solutions.md)** — Balancing strategies and techniques
- **[README.md](exohabitnet/README.md)** — Project structure and quick start

---

## 🤝 Collaboration Tips

### For New Collaborators
1. **Do NOT commit data files** — They are gitignored for a reason (too large)
2. **Run data collection yourself** — Each collaborator downloads from NASA independently
3. **Share only code changes** — Scripts, docs, model architecture (not trained weights)
4. **Use branches** — Create feature branches for experiments

### Sharing Results
```bash
# Share code changes
git add exohabitnet/scripts/new_feature.py
git commit -m "Add new feature"
git push origin your-branch

# Share trained model (use external service)
# Upload to Google Drive, Hugging Face, or similar
# Do NOT push .pth/.h5 files to GitHub
```

### Git Branch Strategy
- `main` — Stable, working code
- `preprocess` — Preprocessing experiments
- `model-dev` — Model architecture experiments
- `feature/*` — New features

---

## 🐛 Troubleshooting

### "NASA API returns 0 results"
- Check internet connection
- Verify `scripts/test_api.py` works
- NASA MAST sometimes has rate limits (wait 5 minutes, retry)

### "Only 7 HABITABLE samples collected"
- This is normal! Habitable zone planets are extremely rare (~1%)
- Run `balance_dataset.py` to augment to 200+ samples
- Or adjust `ROCKY_PLANET_MAX_RE` in `collect_kepler_data.py` (currently 3.0 R⊕)

### "Import error: lightkurve not found"
```bash
pip install --upgrade lightkurve astropy pandas numpy
```

### "Git push rejected (file too large)"
- Check if you accidentally committed data files
- Remove from git: `git rm --cached data/raw_fits/* -r`
- Add to `.gitignore` if not already there

---

## 📊 Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Overall Accuracy | ≥ 75% | Competitive with Kepler pipeline |
| HABITABLE F1 | ≥ 0.65 | Most critical metric |
| FALSE_POSITIVE Precision | ≥ 0.80 | Minimize false discoveries |
| Macro F1 | ≥ 0.70 | Accounts for class imbalance |

---

## 📚 References

1. **NASA Exoplanet Archive:** https://exoplanetarchive.ipac.caltech.edu/
2. **Lightkurve Library:** https://docs.lightkurve.org/
3. **Kopparapu et al. (2013):** "Habitable Zones Around Main-Sequence Stars"
4. **Shallue & Vanderburg (2018):** "Identifying Exoplanets with Deep Learning"

---

## 👥 Team

**Project:** ExoHabitNet  
**Course:** ECSCI24305  
**Phase:** 2 & 3  
**Last Updated:** March 10, 2026

---

## 📝 License

Educational project for course ECSCI24305. Data sourced from NASA Kepler Mission (public domain).

---

**Happy planet hunting! 🪐🔭**
