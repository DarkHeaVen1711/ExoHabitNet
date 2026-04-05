# Exoplanet Habitability Classification — Dataset Architecture
## Project: ExoHabitNet | Phase 2 Deliverable

---

## Task 1A: Technical Attributes (Feature Signal Analysis)

These are the raw and derived features extracted from each Kepler light-curve observation.  
Each feature acts as a **signal dimension** that the deep learning model uses to discriminate between classes.

| # | Feature Name | Source | Unit | DL Signal Role |
|---|---|---|---|---|
| 1 | **PDCSAP Flux** | Kepler `.fits` SAP_FLUX (corrected) | e⁻/s | Primary time-series. Transit dips are the key classification signal. |
| 2 | **Normalized Flux** | Derived from PDCSAP | Dimensionless (0–1) | Removes stellar brightness bias, enabling cross-star comparison. |
| 3 | **Orbital Period (P)** | Derived via BLS/phase-folding | Days | Determines if orbit falls within Habitable Zone (HZ). |
| 4 | **Stellar Effective Temperature (Teff)** | NASA Exoplanet Archive | Kelvin | Used to compute HZ boundaries (Kopparapu 2013 model). |
| 5 | **Planet-to-Star Radius Ratio (Rp/R\*)** | Derived from transit depth (ΔF = (Rp/R\*)²) | Dimensionless | Determines planet size; small Rp/R\* → potentially rocky. |
| 6 | **Transit Duration (T14)** | Derived from light-curve | Hours | Distinguishes short stellar flares from long planetary transits. |
| 7 | **Centroid Motion** | Kepler Pixel Response Function | Arcsec | Identifies if the transit dip originates from the target star (FP check). |
| 8 | **SAP Quality Flags** | Kepler `.fits` header | Bitmask | Quality filter; used during data cleaning to remove bad cadences. |

### How Features Form the DL Signal

```
Raw Light Curve (PDCSAP Flux time series)
        │
        ▼
[Sequence of N flux values over time T]
        │
        ▼
Deep Learning Model sees:
  ├─ 1D-CNN / LSTM layer → learns transit SHAPE (dip depth, duration)
  ├─ Global features (Teff, Period) injected via dense branch
  └─ Combined → classification head → 3-class softmax output
```

A transit event causes a **periodic, symmetric dip** in normalized flux. The model learns:
- **Depth** of dip → `Rp/R*` → planet size
- **Period** of recurrence → orbital distance → HZ check
- **Shape symmetry** → distinguishes planet transit from eclipsing binary (EB) or stellar noise

---

## Task 1B: Labeling Schema (3-Class Definition)

| Label ID | Class Name | Code | Definition & Criteria |
|---|---|---|---|
| **0** | Confirmed Habitable | `HABITABLE` | KOI confirmed as planet (`koi_disposition = CONFIRMED`) + orbital period places it within the conservative HZ (Kopparapu 2013: 0.99–1.70 AU for G-star) + Rp/R* consistent with rocky/super-Earth radius (< 1.6 R⊕). |
| **1** | Non-Habitable Exoplanet | `NON_HABITABLE` | KOI confirmed as planet but fails HZ check (too hot/cold) or is a gas giant (Rp > 4 R⊕). Includes hot Jupiters and cold ice giants. |
| **2** | Stellar Artifact / False Positive | `FALSE_POSITIVE` | KOI flagged as `FALSE POSITIVE` by the Kepler pipeline. Caused by eclipsing binaries, background stars, or instrumental systematics. Centroid motion and odd/even transit depth differences are distinguishing signals. |

### Labeling Decision Tree

```
KOI Record
    │
    ├─ koi_disposition == "FALSE POSITIVE" ─────────────────► Label 2: FALSE_POSITIVE
    │
    ├─ koi_disposition == "CONFIRMED"
    │       │
    │       ├─ Rp > 4 R⊕  ──────────────────────────────────► Label 1: NON_HABITABLE
    │       │
    │       ├─ Period outside HZ boundaries for given Teff ──► Label 1: NON_HABITABLE
    │       │
    │       └─ Rp ≤ 1.6 R⊕ AND Period within HZ ────────────► Label 0: HABITABLE
    │
    └─ koi_disposition == "CANDIDATE" ──────────────────────► EXCLUDE (ambiguous)
```

> **Viva Note:** Labels are grounded in the peer-reviewed Kopparapu et al. (2013) habitable zone model, not arbitrary thresholds. This gives the labeling scientific validity.

---

## Task 1C: Collection Strategy (lightkurve + NASA MAST API)

### Library Stack

```bash
pip install lightkurve astropy pandas numpy
```

### Step-by-Step Pipeline

```
Step 1: Query NASA Exoplanet Archive
   └─ GET https://exoplanetarchive.ipac.caltech.edu/TAP/sync
      └─ ADQL query for all KOIs with disposition, period, Teff, Rp

Step 2: Filter KOI List
   └─ Keep only CONFIRMED + FALSE POSITIVE entries
   └─ Assign label from decision tree above

Step 3: For each KOI → fetch light curve via lightkurve
   └─ lk.search_lightcurve(target="KIC {kid}", mission="Kepler")
   └─ .download_all() → returns LightCurveCollection of .fits files
   └─ Stitch all quarters → single continuous time series

Step 4: Save raw .fits artifacts + metadata CSV
   └─ raw_fits/{KIC_ID}/
   └─ data_collection_log.csv (Source ID, Quality Flags, Label)

Step 5: Hand off to Preprocessing Pipeline (Phase 3)
```

## Task 1D: Leakage-Safe Split Policy

The current pipeline splits real preprocessed samples into train and test sets before any synthetic augmentation is created.

### Policy
- Real samples are split with stratified sampling.
- Synthetic HABITABLE samples are generated only from the training split.
- The test split remains real-only to preserve a meaningful holdout evaluation.

### Output Artifacts
- `data/processed_dataset.csv` — real-only preprocessed samples
- `data/train_dataset.csv` — training split with synthetic augmentation
- `data/test_dataset.csv` — real-only holdout split

### Why `.fits` Files?
FITS (Flexible Image Transport System) files from MAST contain:
- `TIME` column (BJD - 2454833)
- `PDCSAP_FLUX` (Pre-search Data Conditioning Simple Aperture Photometry)
- `SAP_QUALITY` bitmask (quality flags per cadence)
- Stellar metadata in FITS header (Teff, log g, RA, Dec)

This is the **same raw data used by NASA scientists** — making the dataset original and scientifically credible.

---
*Phase 2 | ExoHabitNet | Last updated: 2026-02-25*
