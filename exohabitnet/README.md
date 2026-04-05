# ExoHabitNet 🪐
### Exoplanet Habitability Classification via NASA Kepler Light-Curve Analysis

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

# 2. Collect 500+ Kepler light curves (Phase 2)
python scripts/collect_kepler_data.py

# 3. Run preprocessing pipeline + generate EDA charts (Phase 3)
python scripts/preprocessing_pipeline.py
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
