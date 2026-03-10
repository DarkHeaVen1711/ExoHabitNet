# ExoHabitNet 🪐
**AI-Powered Exoplanet Habitability Classification Using NASA Kepler Data**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NASA Kepler](https://img.shields.io/badge/Data-NASA%20Kepler-orange.svg)](https://archive.stsci.edu/kepler/)

> 🌍 Detecting potentially habitable worlds through deep learning analysis of stellar light curves  
> 🔬 **Course Project:** ECSCI24305 | **Status:** Phases 1-3 Complete (Ready for Model Training)

---

## 🎯 Project Overview

ExoHabitNet is a deep learning system designed to authenticate and classify exoplanet candidates from NASA's Kepler Mission, focusing on identifying potentially habitable worlds. Using **1D Convolutional Neural Networks (CNN)**, the system analyzes raw transit light curves to distinguish between:

- 🟢 **Habitable Zone Planets** — Rocky planets in the "Goldilocks zone" where liquid water could exist
- 🔴 **Non-Habitable Planets** — Gas giants or planets in extreme orbits
- ⚠️ **False Positives** — Eclipsing binary stars and instrumental artifacts

### Why This Matters

Only **~5,000 confirmed exoplanets** exist out of 200+ billion stars in our galaxy. Of these, fewer than **60 are potentially habitable**. Traditional methods require extensive follow-up observations. ExoHabitNet accelerates discovery by automating the classification of candidate signals, helping astronomers prioritize which targets deserve further study.

---

## 🚀 Current Project Status (March 2026)

### ✅ **Phases 1-3: Complete**
- **Environment Setup** — Python 3.11.9, all dependencies configured
- **Data Collection** — 460 light curves from NASA Kepler Mission DR25
- **Preprocessing & Augmentation** — Class imbalance solved (1.5% → 30.6%), **653 balanced samples ready**

### 🔄 **Phase 4: In Progress**
- 1D-CNN model architecture design
- Training pipeline implementation

### 📊 **Dataset Statistics**
| Class | Samples | Percentage |
|-------|---------|------------|
| HABITABLE | 200 | 30.6% |
| NON_HABITABLE | 222 | 34.0% |
| FALSE_POSITIVE | 231 | 35.4% |

**See the [`preprocess`](https://github.com/DarkHeaVen1711/ExoHabitNet/tree/preprocess) branch for full implementation.**

---

## 🔬 Technical Approach

---

## 🔬 Technical Approach

### Data Source
- **NASA Kepler Mission DR25** — Final data release with 460 candidate light curves
- **MAST Archive** — Downloaded via `lightkurve` Python library
- **Time-series data** — ~62,000 flux measurements per star over 4 years

### Machine Learning Pipeline

```
Raw FITS Files (NASA Kepler)
        ↓
Quality Filtering & Outlier Removal
        ↓
Z-Score Normalization
        ↓
Phase Folding (Kepler's 3rd Law)
        ↓
Fixed-Length Binning (1024 timesteps)
        ↓
Class Balancing (SMOTE-like augmentation)
        ↓
1D-CNN Deep Learning Model
        ↓
3-Class Classification Output
```

### Key Innovation: Solving Class Imbalance
Habitable planets are extremely rare (~1.5% of dataset). We solved this with:
- **Gaussian noise injection** (σ=0.012)
- **SMOTE-like interpolation** between real samples
- **28.6x augmentation** (7 real → 200 synthetic HABITABLE samples)
- **Class-weighted loss function** for training

**Result:** Balanced 30.6% / 34.0% / 35.4% distribution

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- 8GB RAM minimum
- ~10GB disk space for NASA data

### Installation

```bash
# Clone repository
git clone https://github.com/DarkHeaVen1711/ExoHabitNet.git
cd ExoHabitNet

# Install dependencies (use preprocess branch for full code)
git checkout preprocess
cd exohabitnet
pip install -r requirements.txt
```

### Download NASA Data

```bash
# Downloads 460 light curves from NASA MAST API (~30-60 minutes)
python scripts/collect_kepler_data.py
```

### Run Preprocessing

```bash
# Clean, normalize, phase-fold, and augment dataset
python scripts/preprocessing_pipeline.py
```

**Output:** `data/processed_dataset.csv` — 653 samples ready for model training

---

## 📁 Repository Structure

## 📁 Repository Structure

```
ExoHabitNet/
├── README.md                    # Project overview (you are here)
├── execution_process.md         # 63-step execution guide (9 phases)
├── .gitignore                   # Excludes large data files
│
└── exohabitnet/                 # Main project directory
    ├── scripts/                 # Python scripts (see preprocess branch)
    ├── docs/                    # Technical documentation
    ├── data/                    # Data files (gitignored, download locally)
    ├── models/                  # Model architecture & checkpoints
    └── reports/                 # EDA visualizations & results
```

### 🌿 Branch Strategy

| Branch | Purpose | Status |
|--------|---------|--------|
| **`main`** | Documentation & project overview | ✅ Current |
| **`preprocess`** | Full implementation (scripts, pipeline, data) | ✅ Complete (Phases 1-3) |
| **`model-dev`** | Model architecture experiments | 🔄 Coming soon |

**⚠️ Important:** All Python scripts and implementation code are in the **`preprocess`** branch. The `main` branch contains only documentation.

---

## 📖 Documentation

Comprehensive guides available in the repository:

- **[execution_process.md](execution_process.md)** — Complete 63-step guide with progress tracking
- **[EXECUTION_SUMMARY.md](https://github.com/DarkHeaVen1711/ExoHabitNet/blob/preprocess/EXECUTION_SUMMARY.md)** (preprocess branch) — Phases 1-3 detailed summary
- **[PROGRESS_REPORT.md](https://github.com/DarkHeaVen1711/ExoHabitNet/blob/preprocess/exohabitnet/PROGRESS_REPORT.md)** (preprocess branch) — Execution metrics & verification
- **[dataset_architecture.md](https://github.com/DarkHeaVen1711/ExoHabitNet/blob/preprocess/exohabitnet/docs/dataset_architecture.md)** (preprocess branch) — Feature definitions & labeling schema
- **[class_imbalance_solutions.md](https://github.com/DarkHeaVen1711/ExoHabitNet/blob/preprocess/exohabitnet/docs/class_imbalance_solutions.md)** (preprocess branch) — Augmentation strategies

---

## 🎯 Project Achievements

### ✅ Completed Milestones

- [x] **Environment Setup** — Python 3.11.9 with all dependencies
- [x] **NASA Data Access** — Successfully queried MAST API
- [x] **Data Collection** — 460 exoplanet light curves downloaded
- [x] **Class Imbalance Solution** — HABITABLE 1.5% → 30.6% (balanced)
- [x] **Preprocessing Pipeline** — 100% success rate (460/460 samples)
- [x] **Data Augmentation** — 193 synthetic HABITABLE samples generated
- [x] **Dataset Preparation** — 653 samples ready for training
- [x] **Bug Fixes** — 2 critical preprocessing bugs resolved
- [x] **Documentation** — Complete execution guide & progress reports

### 📊 Key Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **Data Collection Success Rate** | 97.3% (460/473) | ✅ Excellent |
| **Preprocessing Success Rate** | 100% (460/460) | ✅ Perfect |
| **Final Dataset Size** | 653 samples | ✅ Ready |
| **Class Balance** | 30.6% / 34.0% / 35.4% | ✅ Optimal |
| **HABITABLE Augmentation** | 28.6x (7 → 200) | ✅ Solved |

---

## 🛠️ Technologies Used

**Data Science Stack:**
- `lightkurve` — NASA Kepler data access & manipulation
- `astropy` — FITS file handling & astronomical calculations
- `pandas` & `numpy` — Data processing & numerical operations
- `scipy` — Signal processing (phase folding, binning)

**Visualization:**
- `matplotlib` & `seaborn` — EDA charts & class distribution plots

**Deep Learning (Phase 4):**
- `PyTorch` — 1D-CNN model implementation (upcoming)
- `TensorBoard` — Training visualization (upcoming)

---

## 🚦 Roadmap

### Phase 4: Model Architecture (Current)
- [ ] Design 1D-CNN architecture with 3-5 convolutional blocks
- [ ] Implement class-weighted loss function
- [ ] Set up training pipeline with validation

### Phase 5: Training Pipeline
- [ ] 70/15/15 train/val/test stratified split
- [ ] Adam optimizer with learning rate scheduling
- [ ] Early stopping and model checkpointing
- [ ] TensorBoard logging

### Phase 6: Model Evaluation
- [ ] Confusion matrix & per-class metrics
- [ ] ROC curves & AUC scores
- [ ] Target: >75% accuracy, HABITABLE F1 >0.60

### Phase 7-9: Refinement & Deployment
- [ ] Hyperparameter tuning
- [ ] Ensemble modeling (optional)
- [ ] Final model evaluation on test set
- [ ] Documentation & results visualization

---

## 🤝 Contributing

This is an academic course project for ECSCI24305. Collaboration guidelines:

1. **Fork the repository** and work on feature branches
2. **Use the `preprocess` branch** for implementation code
3. **Do NOT commit data files** — They're gitignored (download from NASA)
4. **Share models externally** — Use Google Drive/Hugging Face for trained weights
5. **Document changes** — Update relevant .md files

### For Collaborators

```bash
# Get full implementation code
git checkout preprocess
cd exohabitnet

# Setup your environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download data (required, not in Git)
python scripts/collect_kepler_data.py
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Data Attribution:**
- NASA Kepler Mission DR25 data is publicly available via MAST Archive
- [NASA Exoplanet Archive](https://exoplanetarchive.ipasa.nasa.gov/)

---

## 📚 References

1. **Kepler Mission:** [NASA Kepler/K2](https://www.nasa.gov/mission_pages/kepler/main/index.html)
2. **Lightkurve Library:** [Lightkurve Documentation](https://docs.lightkurve.org/)
3. **Habitable Zone Calculations:** Kopparapu et al. (2013) - "Habitable Zones Around Main-Sequence Stars"
4. **SMOTE Augmentation:** Chawla et al. (2002) - "SMOTE: Synthetic Minority Over-sampling Technique"

---

## 📞 Contact

**Project Repository:** [https://github.com/DarkHeaVen1711/ExoHabitNet](https://github.com/DarkHeaVen1711/ExoHabitNet)

**Course:** ECSCI24305  
**Date:** March 2026

---

<div align="center">

### 🌌 Searching for Habitable Worlds, One Light Curve at a Time 🔭

**Made with ❤️ using Python, NASA data, and Deep Learning**

[![GitHub stars](https://img.shields.io/github/stars/DarkHeaVen1711/ExoHabitNet?style=social)](https://github.com/DarkHeaVen1711/ExoHabitNet/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/DarkHeaVen1711/ExoHabitNet?style=social)](https://github.com/DarkHeaVen1711/ExoHabitNet/network/members)

</div>

