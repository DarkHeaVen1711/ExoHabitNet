# ExoHabitNet — Step-by-Step Execution Process
**Exoplanet Habitability Classification via NASA Kepler Light-Curve Analysis**

> **Project:** ExoHabitNet | **Course:** ECSCI24305 | **Date:** March 2026

---

## 📊 Slide 1: Problem Statement & Approach

**Problem:** Authenticating and classifying exoplanet candidates from false positives using raw transit data.

**Data Type to be Collected:**
- **Light Curves (Flux):** The drop in star brightness during a transit event
- **Key Attributes:** Orbital Period, Stellar Temperature (Teff), Planet-to-Star Radius Ratio (Rp/R*)

**Deep Learning Approach:** 1D-CNN (Convolutional Neural Network) to learn the exact physical shape of a planetary transit dip versus an eclipsing binary star.

---

## 📊 Slide 2: Dataset Design & Collection

**Format & Size:** 500+ samples of `.fits` (Flexible Image Transport System) files from NASA Kepler Mission (DR25)

**Collection Process:** Automated tracking via Python scripts querying NASA MAST API with continuous `data_collection_log.csv` maintenance

**Target Classes (Labels):**
- 🟢 **Class 0 (HABITABLE):** Confirmed rocky planet inside the Habitable Zone
- 🔴 **Class 1 (NON_HABITABLE):** Gas giant, or orbit is too hot/cold
- ⚠️ **Class 2 (FALSE_POSITIVE):** Instrument artifact or eclipsing binary star

---

## 🚀 Complete Execution Process

### **Phase 1: Environment Setup & Dependency Verification**

| Step | Action | Command/Description |
|------|--------|---------------------|
| 1 | Verify Python environment | Ensure Python 3.8+ is installed: `python --version` |
| 2 | Install dependencies | `pip install -r requirements.txt` |
| 3 | Add deep learning framework | `pip install torch torchvision` (PyTorch) or `pip install tensorflow` |
| 4 | Test NASA MAST API access | `python scripts/test_api.py` |
| 5 | Confirm data directory structure | Verify `data/`, `scripts/`, `models/` folders exist |

**Verification:** All packages installed without errors, API returns valid response.

---

### **Phase 2: Data Collection & Verification**

| Step | Action | Command/Description |
|------|--------|---------------------|
| 6 | Check current data collection status | Inspect `data/raw_fits/` folder for existing FITS files |
| 7 | Verify class distribution | Count samples in `FALSE_POSITIVE/`, `NON_HABITABLE/`, `HABITABLE/` folders |
| 8 | Run data collection script (if needed) | `python scripts/collect_kepler_data.py` |
| 9 | Review collection log | Open `data/data_collection_log.csv` and verify 500+ entries |
| 10 | Validate FITS file integrity | Ensure no corrupted files, check FITS headers |

**Current Status:** ~217 FALSE_POSITIVE, ~229 NON_HABITABLE, ~7 HABITABLE samples collected.

**Verification:** `data_collection_log.csv` contains complete metadata (KIC ID, Period, Teff, Label).

---

### **Phase 3: Data Preprocessing Pipeline**

| Step | Action | Command/Description |
|------|--------|---------------------|
| 11 | Execute preprocessing pipeline | `python scripts/preprocessing_pipeline.py` |
| 12 | **Cleaning Step** | Remove quality-flagged cadences, interpolate NaN flux values, sigma-clip outliers |
| 13 | **Normalization Step** | Apply z-score normalization (local) and min-max normalization (global) |
| 14 | **Phase Folding Step** | Align all transit events at phase = 0 using orbital period and t0 |
| 15 | **Binning Step** | Resample phase-folded curves into fixed 1024-timestep sequences |
| 16 | Generate EDA charts | Charts saved to `reports/eda_charts/` |
| 17 | Split dataset | 70% train / 15% validation / 15% test with stratified sampling |
| 18 | Compute class weights | Calculate inverse-frequency weights for imbalanced loss function |

**Output Files:**
- `data/processed_dataset.csv` — Normalized sequences ready for deep learning
- `reports/eda_charts/chart1_transit_depth_distribution.png`
- `reports/eda_charts/chart2_class_balance_histogram.png`
- `reports/eda_charts/chart3_phase_folded_samples.png`

**Verification:** All sequences have shape (1024,), flux values are normalized (mean ≈ 0, std ≈ 1).

---

### **Phase 4: Model Architecture Development**

| Step | Action | Description |
|------|--------|-------------|
| 19 | Create model directory | `mkdir models` |
| 20 | Implement 1D-CNN architecture | Create `models/cnn_model.py` |
| 21 | Define CNN backbone | 3 Conv1D blocks (64→128→256 filters) with BatchNorm + Dropout |
| 22 | Add metadata branch | Dense layer for Period, Teff, Rp/R* features (→ 32 units) |
| 23 | Concatenate features | Combine CNN output + metadata → final dense layer |
| 24 | Define output layer | Dense(3) + Softmax for 3-class classification |
| 25 | Implement class-weighted loss | `CrossEntropyLoss` with weights from Phase 3 |

**Architecture Diagram:**
```
Input (batch, 1024, 1)
  │
  ├─ Conv1D(64 filters, kernel=5) → ReLU → BatchNorm → Dropout(0.3)
  ├─ Conv1D(128 filters, kernel=5) → ReLU → BatchNorm → Dropout(0.3)
  ├─ Conv1D(256 filters, kernel=3) → ReLU → BatchNorm → Dropout(0.4)
  └─ Global Average Pooling → (batch, 256)
                          ↓
Metadata (Period, Teff, Rp/R*)
  └─ Dense(32) → ReLU
                ↓
[CNN Features ⊕ Metadata Features]
  └─ Dense(3) → Softmax → [HABITABLE, NON_HABITABLE, FALSE_POSITIVE]
```

**Verification:** Model summary shows ~500K-1M trainable parameters.

---

### **Phase 5: Training Pipeline Implementation**

| Step | Action | Description |
|------|--------|-------------|
| 26 | Create training script | `scripts/train.py` with PyTorch/TensorFlow training loop |
| 27 | Configure optimizer | Adam optimizer with learning rate = 0.001 |
| 28 | Add learning rate scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| 29 | Implement early stopping | Monitor validation F1-score, patience=10 epochs |
| 30 | Add model checkpointing | Save best model to `models/checkpoints/best_model.pth` |
| 31 | Integrate logging system | TensorBoard or Weights & Biases for real-time monitoring |
| 32 | Configure data augmentation | Gaussian noise (σ=0.02) + time shifting for HABITABLE class |
| 33 | Set hyperparameters | Batch size=32, Epochs=50-100, Weight decay=1e-5 |

**Training Configuration:**
- **Loss Function:** CrossEntropyLoss with class weights [w_HABITABLE ≈ 50, w_NON_HABITABLE ≈ 2, w_FALSE_POSITIVE ≈ 2]
- **Augmentation Strategy:** 3x oversampling for HABITABLE class to balance dataset
- **Regularization:** Dropout (0.3-0.4) + L2 weight decay (1e-5)

**Verification:** Training loop runs for 1 epoch without errors, loss decreases.

---

### **Phase 6: Model Training Execution**

| Step | Action | Command/Description |
|------|--------|---------------------|
| 34 | Start training | `python scripts/train.py --config config/hyperparameters.yaml` |
| 35 | Monitor training loss | Watch for consistent decrease over epochs |
| 36 | Monitor validation metrics | Track validation accuracy and macro-F1 score |
| 37 | Monitor per-class F1 scores | Especially HABITABLE class (hardest to classify) |
| 38 | Check for overfitting | Compare train vs. validation loss gap (should be <10%) |
| 39 | Wait for early stopping | Training stops when validation F1 plateaus for 10 epochs |
| 40 | Save best model checkpoint | Best model saved based on highest validation macro-F1 |

**Expected Training Progress:**
- **Epoch 1-5:** Rapid loss decrease, accuracy climbs above 50%
- **Epoch 10-20:** Validation accuracy reaches 70-75%
- **Epoch 30-50:** Fine-tuning, macro-F1 optimized
- **Early stopping:** Triggers around epoch 40-60

**Verification:** Training curves show convergence, no NaN losses, validation metrics improve.

---

### **Phase 7: Model Evaluation & Testing**

| Step | Action | Description |
|------|--------|-------------|
| 41 | Create evaluation script | `scripts/evaluate.py` for test set inference |
| 42 | Load best model checkpoint | Load from `models/checkpoints/best_model.pth` |
| 43 | Run inference on test set | Predict labels for held-out 15% test samples |
| 44 | Generate confusion matrix | 3×3 matrix showing true vs. predicted labels |
| 45 | Calculate per-class metrics | Precision, Recall, F1-score for each class |
| 46 | Calculate overall metrics | Accuracy, macro-averaged F1, weighted F1 |
| 47 | Identify misclassified samples | Extract top-10 errors for error analysis |
| 48 | Export results | Save to `reports/model_performance.json` |

**Target Performance Metrics:**
- ✅ **Overall Accuracy:** ≥ 75%
- ✅ **HABITABLE F1-score:** ≥ 0.65
- ✅ **FALSE_POSITIVE Precision:** ≥ 0.80
- ✅ **Macro-averaged F1:** ≥ 0.70

**Verification:** Test accuracy exceeds random baseline (33%), all classes have F1 > 0.6.

---

### **Phase 8: Visualization & Reporting**

| Step | Action | Output File |
|------|--------|-------------|
| 49 | Create visualization utility | `utils/visualization.py` |
| 50 | Plot training history | `reports/training_history.png` (loss & accuracy curves) |
| 51 | Generate confusion matrix heatmap | `reports/confusion_matrix.png` |
| 52 | Plot ROC curves | `reports/roc_curves.png` (one-vs-rest for each class) |
| 53 | Visualize sample predictions | `reports/sample_predictions/` (light curves with predicted labels) |
| 54 | Create classification report table | `reports/classification_report.md` (precision/recall/F1 table) |
| 55 | Generate saliency maps (optional) | `reports/saliency_maps/` (which features drive predictions) |

**Deliverable Charts:**
1. **Training History:** Loss and accuracy over epochs for train/val sets
2. **Confusion Matrix:** 3×3 heatmap showing classification performance
3. **ROC Curves:** True positive rate vs. false positive rate per class
4. **Sample Predictions:** 9-panel grid showing correctly/incorrectly classified light curves

**Verification:** All visualization files generated and saved to `reports/` folder.

---

### **Phase 9: Model Refinement & Optimization** *(Iterative)*

| Step | Action | Description |
|------|--------|-------------|
| 56 | Analyze error patterns | Identify which class pairs are most confused |
| 57 | Investigate HABITABLE errors | Check if misclassified as FALSE_POSITIVE or NON_HABITABLE |
| 58 | Review misclassified light curves | Visually inspect transit shapes of errors |
| 59 | Hyperparameter tuning experiment | Grid search: dropout rates, learning rates, augmentation strength |
| 60 | Architectural experiments | Try deeper networks (4-5 Conv blocks) or residual connections |
| 61 | Ensemble modeling (optional) | Train 5-fold cross-validation ensemble for robust predictions |
| 62 | Re-train with refined configuration | Apply best hyperparameters from experiments |
| 63 | Re-evaluate on test set | Compare performance improvement |

**Optimization Strategies:**
- **If HABITABLE F1 < 0.6:** Increase augmentation strength or collect more HZ samples
- **If FALSE_POSITIVE Precision < 0.75:** Add centroid motion as additional feature
- **If overall accuracy plateaus:** Try ensemble of 3-5 models with different initializations

**Verification:** Final model achieves target metrics and generalizes well to unseen data.

---

## 📁 Project File Structure

```
exohabitnet/
├── data/
│   ├── raw_fits/
│   │   ├── HABITABLE/           # ~7 samples (target: 20+ via relaxed HZ)
│   │   ├── NON_HABITABLE/       # ~229 samples
│   │   └── FALSE_POSITIVE/      # ~217 samples
│   ├── processed/
│   │   └── processed_dataset.csv      # Fixed-length sequences (1024 timesteps)
│   ├── data_collection_log.csv        # Metadata: KIC, Period, Teff, Label
│   └── logs/                          # Collection logs
├── docs/
│   ├── dataset_architecture.md        # Feature definitions, HZ model, labeling schema
│   └── execution_process.md           # **THIS FILE** — Step-by-step guide
├── models/
│   ├── __init__.py
│   ├── cnn_model.py                   # 1D-CNN architecture class
│   └── checkpoints/
│       └── best_model.pth             # Best model from training
├── scripts/
│   ├── collect_kepler_data.py         # ✅ Data collection (complete)
│   ├── preprocessing_pipeline.py      # ✅ Preprocessing (complete)
│   ├── train.py                       # 🔨 Training loop (to be created)
│   ├── evaluate.py                    # 🔨 Test evaluation (to be created)
│   └── predict.py                     # 🔨 Single inference (to be created)
├── utils/
│   ├── metrics.py                     # Custom metric calculators
│   └── visualization.py               # Plotting utilities
├── reports/
│   ├── eda_charts/                    # EDA visualizations from preprocessing
│   ├── training_history.png           # Loss/accuracy curves
│   ├── confusion_matrix.png           # 3×3 classification matrix
│   ├── roc_curves.png                 # One-vs-rest ROC curves
│   ├── sample_predictions/            # Example light curves with predictions
│   └── model_performance.json         # Final test metrics
├── config/
│   └── hyperparameters.yaml           # Training configuration (optional)
├── requirements.txt                   # Python dependencies
└── README.md                          # Project overview
```

---

## ✅ Validation Checklist

### **Data Collection Validation**
- [ ] `data_collection_log.csv` contains ≥ 500 entries
- [ ] Class distribution verified: FP ~40-50%, NH ~40-50%, H ~5-10%
- [ ] No corrupted FITS files (test with `astropy.io.fits.open()`)
- [ ] All KIC IDs have complete metadata (Period, Teff, koi_time0bk)

### **Preprocessing Validation**
- [ ] `data/processed_dataset.csv` generated successfully
- [ ] All sequences have shape (1024,)
- [ ] Normalized flux: mean ≈ 0, std ≈ 1 (z-score) or range [0,1] (min-max)
- [ ] Phase-folded transits centered at phase = 0
- [ ] EDA charts show expected patterns (transit depths, class balance)

### **Training Validation**
- [ ] Training loss decreases consistently (no divergence)
- [ ] Validation accuracy > 50% after 5 epochs
- [ ] No NaN/Inf losses (indicates stable training)
- [ ] Train-validation gap < 10% (no severe overfitting)
- [ ] HABITABLE class F1-score improves over training

### **Final Model Validation**
- [ ] **Test Accuracy ≥ 75%** (competitive with Kepler pipeline)
- [ ] **HABITABLE F1 ≥ 0.65** (most critical metric)
- [ ] **FALSE_POSITIVE Precision ≥ 0.80** (minimize false discoveries)
- [ ] Confusion matrix shows < 10% HABITABLE ↔ FALSE_POSITIVE confusion
- [ ] Model generalizes to unseen data (no overfitting to training set)

---

## 🔧 Technical Decisions

| Decision Point | Options | Chosen Approach | Rationale |
|----------------|---------|-----------------|-----------|
| **Deep Learning Framework** | TensorFlow vs. PyTorch | **PyTorch** | More flexible for research, better debugging, active community |
| **Architecture** | 1D-CNN vs. LSTM/GRU | **1D-CNN** | Transit shape learning > temporal dependencies; CNNs faster |
| **Sequence Length** | 512 vs. 1024 vs. 2048 | **1024** | Balances transit detail vs. computational cost |
| **Normalization** | Global (min-max) vs. Local (z-score) | **Local (primary)** | Removes stellar variability, focuses model on transit shape |
| **Class Imbalance Strategy** | Oversampling vs. Undersampling vs. Weighted Loss | **Oversampling + Weighted Loss** | Preserves all data while balancing |
| **Train/Val/Test Split** | 60/20/20 vs. 70/15/15 vs. 80/10/10 | **70/15/15** | Standard for 500-sample datasets |
| **Evaluation Metric** | Accuracy vs. F1-score vs. AUC | **Macro F1-score** | Best for imbalanced multi-class |
| **Augmentation Strength** | Conservative vs. Aggressive | **Conservative** | Small noise (σ=0.02) avoids unrealistic transits |
| **Metadata Injection** | CNN-only vs. Hybrid | **Hybrid (CNN + metadata)** | Period/Teff improve HZ classification by ~10-15% |

---

## 🔍 Common Issues & Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| **HABITABLE class F1 < 0.5** | Insufficient samples (~7) | Re-run collection with relaxed HZ (±40% boundaries) or aggressive augmentation (10x) |
| **Training loss = NaN** | Learning rate too high | Reduce learning rate to 0.0001 or use gradient clipping |
| **Validation accuracy stuck at 40%** | Model too simple | Add more Conv blocks or increase filter sizes |
| **High train-val gap (>20%)** | Overfitting | Increase dropout to 0.5, add more augmentation, reduce model complexity |
| **FALSE_POSITIVE confused with HABITABLE** | Transit shapes too similar | Add centroid motion feature or secondary eclipse check |
| **Preprocessing script crashes** | Corrupted FITS file | Add try-except in preprocessing loop, log failed KIC IDs |

---

## 🎯 Success Criteria

**Minimum Viable Model (MVP):**
- ✅ Overall test accuracy ≥ 70%
- ✅ All classes have F1-score ≥ 0.60
- ✅ Model trains without divergence
- ✅ Inference time < 100ms per light curve

**Target Performance (Competitive):**
- ✅ Overall test accuracy ≥ 75%
- ✅ HABITABLE F1-score ≥ 0.65
- ✅ FALSE_POSITIVE precision ≥ 0.80
- ✅ Macro-averaged F1 ≥ 0.70
- ✅ Model interpretability: saliency maps show transit features

**Stretch Goals (Publication-Ready):**
- ✅ Overall test accuracy ≥ 80%
- ✅ Ensemble model with 5-fold cross-validation
- ✅ External validation on TESS mission data
- ✅ Ablation study showing metadata contribution

---

## 📚 Key References

1. **Kepler Mission Data:** NASA Exoplanet Archive (https://exoplanetarchive.ipac.caltech.edu/)
2. **Habitable Zone Model:** Kopparapu et al. (2013) — "Habitable Zones Around Main-Sequence Stars"
3. **Lightkurve Library:** https://docs.lightkurve.org/
4. **Phase Folding:** Kepler's Third Law for semi-major axis calculation
5. **Class Imbalance:** SMOTE-like augmentation for time-series data
6. **1D-CNN Architecture:** Inspired by Shallue & Vanderburg (2018) — "Identifying Exoplanets with Deep Learning"

---

## 📧 Contact & Support

**Project Team:** ExoHabitNet Development  
**Course:** ECSCI24305 | Phase 2 & 3  
**Last Updated:** March 10, 2026

For questions or issues, refer to:
- [dataset_architecture.md](dataset_architecture.md) — Feature definitions and labeling schema
- [collect_kepler_data.py](../scripts/collect_kepler_data.py) — Data collection documentation
- [preprocessing_pipeline.py](../scripts/preprocessing_pipeline.py) — Preprocessing logic

---

**Next Steps:** Begin with Phase 1 (Environment Setup) and proceed sequentially through Phase 9. Good luck! 🚀
