# ExoHabitNet 🪐
**Exoplanet Habitability Classification via NASA Kepler Light-Curve Analysis**

> **Course:** ECSCI24305 | **Phase:** 2 & 3  
> **Deep Learning Approach:** 1D-CNN for transit signal classification

---

## 📊 Results Snapshot

**Latest Evaluation:**
- **Holdout Test Accuracy:** 75.0%
- **Macro F1-Score:** 0.5006
- **Best Model:** `models/checkpoints/best_model.pth`
- **Leakage Status:** real-only holdout split with train-only augmentation

**Key Outcome:** The model now evaluates on untouched real data after the leakage issue was removed.

---

## 🚀 Quick Start for Collaborators

### Current Project Status (April 2026)
**✅ Phases Completed:** 1-7 (Environment Setup → Preprocessing → Model Training → Evaluation)  
**🔄 Next Phase:** Phase 8 (Visualization & Reporting) and Model Optimization  
**📊 Dataset Ready:** real-only holdout split plus train-only augmentation for leak-safe evaluation

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

### 7. Preprocess Data (Already Executed)
```bash
# Clean, normalize, phase-fold, split, and augment only the training split
python scripts/preprocessing_pipeline.py
```

**Output:** `data/processed_dataset.csv`, `data/train_dataset.csv`, `data/test_dataset.csv`

---

## 📊 Project Overview

### Problem Statement
Authenticate and classify exoplanet candidates from false positives using raw transit light-curve data.

### What Has Been Executed (Phases 1-7)

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
- **Splitting:** Real samples are split before augmentation to prevent leakage
- **Augmentation:** HABITABLE class is generated only from the training split
- **Final Outputs:** `processed_dataset.csv`, `train_dataset.csv`, `test_dataset.csv`
- **Class Weights:** Computed dynamically from the final train split

#### ✅ Phase 4: Model Architecture Development
- Developed PyTorch `ExoHabitNetCNN` model
- Implemented 1D-CNN with 3 Convolutional blocks, Batch Normalization, and Dropout
- Global Average Pooling into a 3-class linear classifier

#### ✅ Phase 5: Training Pipeline
- Created a robust stratified data splitting mechanism (Train/Val)
- Overcame class imbalance with weighted CrossEntropyLoss
- Implemented LR Scheduling (ReduceLROnPlateau) and Early Stopping

#### ✅ Phase 6: Model Training
- Trained on the leak-safe augmented training split
- Saved checkpoints for the best validation macro F1-score model
- Maintained detailed TensorBoard logs of performance 

#### ✅ Phase 7: Model Evaluation
- Holdout test accuracy of **75.0%** achieved on real-only data.
- Macro F1-Score of **0.5006** on the holdout test split.
- The earlier 100% accuracy leakage issue was removed.

**Result:** `models/checkpoints/best_model.pth` ready and detailed metrics saved to `reports/model_performance.json`.

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
Preprocessing (clean → normalize → phase-fold → real-only split)
        ↓
Train-only augmentation (SMOTE-like oversampling, noise, time shift)
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
# - data/train_dataset.csv and data/test_dataset.csv from the split-safe pipeline
```

### Step 4: Model Training (Future)
```bash
# Train 1D-CNN model
python scripts/train.py

# Evaluate
python scripts/evaluate.py
```

---

## 🎯 Key Features

### Class Imbalance Solution
The dataset has severe imbalance (HABITABLE ~1.5%). We solve this with:
1. **Relaxed HZ criteria** to collect more HABITABLE samples
2. **Train-only augmentation** after a stratified real-data split
3. **Class-weighted loss** computed from the final train split

See [`docs/class_imbalance_solutions.md`](exohabitnet/docs/class_imbalance_solutions.md) for details.

### Preprocessing Pipeline
- **Cleaning:** Remove quality-flagged cadences, sigma-clip outliers
- **Normalization:** Z-score (local) and min-max (global) methods
- **Phase Folding:** Align transit events using Kepler's Third Law
- **Binning:** Fixed 1024-timestep sequences for CNN input
- **Splitting:** Real samples are split into train/test before any synthetic generation

### Data Augmentation
- Gaussian noise injection on training samples only
- Circular time shifting on training samples only
- SMOTE-like interpolation between real training samples
- Amplitude scaling used sparingly on the training split

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
| Overall Accuracy | ≥ 75% | Current holdout result: 75.0% |
| HABITABLE F1 | ≥ 0.65 | Holdout split has only 1 HABITABLE sample, so this metric is unstable |
| FALSE_POSITIVE Precision | ≥ 0.80 | Model Achieved (0.7529 on current holdout) |
| Macro F1 | ≥ 0.70 | Current holdout result: 0.5006 |

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
**Phase:** 1 - 7 Completed  
**Last Updated:** April 02, 2026

---

## 📝 License

Educational project for course ECSCI24305. Data sourced from NASA Kepler Mission (public domain).

---

**Happy planet hunting! 🪐🔭**
