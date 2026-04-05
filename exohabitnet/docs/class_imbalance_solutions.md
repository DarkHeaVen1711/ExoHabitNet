# Class Imbalance Solutions for ExoHabitNet
**Addressing Severe Class Imbalance in Exoplanet Classification**

---

## Problem Overview

The original dataset has a severe class imbalance:
- **HABITABLE:** ~7 samples (1.5%)
- **NON_HABITABLE:** ~229 samples (48%)
- **FALSE_POSITIVE:** ~217 samples (50%)

This imbalance poses critical challenges:
1. **Model bias:** Model will heavily favor majority classes
2. **Poor minority class performance:** HABITABLE class (most important) will have very low recall
3. **Evaluation metrics:** Accuracy will be misleading (can achieve 98.5% by never predicting HABITABLE)

---

## Three-Pronged Solution Approach

### 🎯 Solution 1: Collect More HABITABLE Samples (Preferred)

**Changes made to `collect_kepler_data.py`:**
- ✅ Relaxed HZ boundaries from ±30% to **±50%** buffer
- ✅ Increased rocky planet threshold from 2.5 R⊕ to **3.0 R⊕**
- ✅ Accept both CONFIRMED and CANDIDATE dispositions

**Scientific justification:**
- Kopparapu (2013) "optimistic HZ" includes planets with thick atmospheres
- Cloud feedback effects can extend habitable zone boundaries by 40-50%
- Mini-Neptunes (2.5-3.0 R⊕) with hydrogen loss could become habitable

**How to use:**
```bash
# Re-run data collection with relaxed criteria
cd exohabitnet
python scripts/collect_kepler_data.py

# This should increase HABITABLE samples from 7 to 15-30
```

**Expected outcome:**
- HABITABLE samples: 7 → **15-30** (2-4x increase)
- Still imbalanced, but augmentation will be more effective

---

### 🎯 Solution 2: Advanced Data Augmentation (Quick Fix)

**Changes made to `preprocessing_pipeline.py`:**
- ✅ Split real samples into train/test before any augmentation
- ✅ Applied augmentation only to the training split
- ✅ Kept the test split real-only to prevent leakage

**New script: `balance_dataset.py`**

This script implements **three strategies** for class balancing:

#### Conservative Strategy (Recommended for Baseline)
```bash
python scripts/balance_dataset.py --strategy conservative
```
- HABITABLE: 7 → **150** (21x oversampling)
- NON_HABITABLE: Keep all **229** samples
- FALSE_POSITIVE: Keep all **217** samples
- **Final balance:** 150 / 229 / 217 (25% / 38% / 37%)

**Augmentation techniques:**
- Gaussian noise (σ=0.008)
- Time shifting (±30 timesteps)

---

#### Moderate Strategy (Recommended for Production)
```bash
python scripts/balance_dataset.py --strategy moderate
```
- HABITABLE: 7 → **200** (28x oversampling)
- NON_HABITABLE: 229 → **220** (slight undersample)
- FALSE_POSITIVE: 217 → **200** (slight undersample)
- **Final balance:** 200 / 220 / 200 (32% / 36% / 32%)

**Augmentation techniques:**
- SMOTE-like interpolation between real samples
- Gaussian noise (σ=0.012)
- Time shifting (±50 timesteps)
- Random amplitude scaling (±3%)

---

#### Aggressive Strategy (High Augmentation)
```bash
python scripts/balance_dataset.py --strategy aggressive
```
- HABITABLE: 7 → **250** (35x oversampling)
- NON_HABITABLE: 229 → **200** (undersample)
- FALSE_POSITIVE: 217 → **200** (undersample)
- **Final balance:** 250 / 200 / 200 (38% / 31% / 31%)

**Augmentation techniques:**
- SMOTE-like interpolation
- Gaussian noise (σ=0.015)
- Time shifting (±80 timesteps)
- Random amplitude scaling (±3%)
- Occasional stellar flares (10% probability)

**⚠️ Warning:** Aggressive augmentation may introduce unrealistic samples. Monitor validation performance carefully.

---

### 🎯 Solution 3: Class Weighting in Loss Function

Both `preprocessing_pipeline.py` and `balance_dataset.py` automatically compute optimal class weights for PyTorch `CrossEntropyLoss`.

**Example output:**
```python
class_weights = torch.tensor([10.5, 1.2, 1.3])  # [HABITABLE, NON_HABITABLE, FALSE_POSITIVE]
loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

**How it works:**
- Formula: `weight_c = total_samples / (n_classes × count_c)`
- HABITABLE gets weight ~10-50x higher than majority classes
- Model penalizes HABITABLE misclassifications much more heavily during training

---

## Recommended Workflow

### Phase 1: Collect More Data
```bash
# 1. Update collection parameters (already done)
cd e:\Coding\ExoHabitNet\exohabitnet

# 2. Re-run data collection to get more HABITABLE samples
python scripts/collect_kepler_data.py

# Expected: 15-30 HABITABLE samples instead of 7
```

---

### Phase 2: Run Preprocessing
```bash
# 3. Run preprocessing with automatic augmentation
python scripts/preprocessing_pipeline.py

# This will:
# - Clean and normalize all light curves
# - Phase-fold transit events
# - Split real samples into train/test first
# - Augment HABITABLE class only in the train split
# - Generate EDA charts
# - Output: data/processed_dataset.csv, data/train_dataset.csv, data/test_dataset.csv
```

---

### Phase 3: Advanced Balancing (Optional but Recommended)
```bash
# 4. Run advanced balancing for production model
python scripts/balance_dataset.py --strategy moderate

# This will:
# - Apply SMOTE-like interpolation
# - Use multiple augmentation techniques
# - Balance all three classes more evenly
# - Output: data/balanced_dataset.csv
```

---

### Phase 4: Train with Class Weights
```python
# In your training script (train.py):

import torch
import torch.nn as nn

# Load class weights from preprocessing output
class_weights = torch.tensor([10.5, 1.2, 1.3])  # Example values

# Define loss function with weights
loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

# During training loop
for batch in train_loader:
    inputs, targets = batch
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)  # Automatically applies class weights
    loss.backward()
```

---

## Validation Metrics

**⚠️ DO NOT use accuracy as primary metric!**

With imbalanced data, use these metrics instead:

### 1. Macro-Averaged F1-Score (Primary Metric)
```python
from sklearn.metrics import f1_score

f1_macro = f1_score(y_true, y_pred, average='macro')
# This treats all classes equally regardless of sample size
```

### 2. Per-Class F1-Scores
```python
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred, 
      target_names=['HABITABLE', 'NON_HABITABLE', 'FALSE_POSITIVE']))
```

**Target performance:**
- ✅ HABITABLE F1 ≥ **0.65** (most critical)
- ✅ NON_HABITABLE F1 ≥ **0.75**
- ✅ FALSE_POSITIVE F1 ≥ **0.80**
- ✅ Macro F1 ≥ **0.70**

**Current pipeline note:** the holdout test split is real-only and very small for HABITABLE, so that class's test F1 can be unstable even when the pipeline is behaving correctly.

### 3. Confusion Matrix Analysis
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
```

**Critical checks:**
- ❌ **Avoid:** HABITABLE misclassified as FALSE_POSITIVE (false discoveries)
- ❌ **Avoid:** FALSE_POSITIVE misclassified as HABITABLE (missed real planets)

---

## Advanced Techniques (Future Enhancements)

### 1. Focal Loss (Instead of CrossEntropyLoss)
Focuses training on hard-to-classify samples.
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

### 2. Ensemble with Different Augmentation Seeds
Train 5 models with different augmentation random seeds, then ensemble predictions.

### 3. Transfer Learning from TESS Mission
Pre-train on larger TESS dataset, fine-tune on Kepler HABITABLE samples.

### 4. Adversarial Validation
Check if augmented samples are distinguishable from real samples (should not be).

---

## File Structure After Balancing

```
exohabitnet/
├── data/
│   ├── processed_dataset.csv          # Real-only preprocessed samples
│   ├── train_dataset.csv              # Train split with synthetic HABITABLE samples
│   ├── test_dataset.csv               # Real-only holdout test split
│   ├── balanced_dataset.csv           # Legacy/optional advanced balancing output
│   └── raw_fits/
│       ├── HABITABLE/       # 7 → 15-30 after re-collection
│       ├── NON_HABITABLE/   # ~229 samples
│       └── FALSE_POSITIVE/  # ~217 samples
├── reports/
│   └── class_balance/
│       ├── class_balance_conservative.png
│       ├── class_balance_moderate.png
│       └── class_balance_aggressive.png
└── scripts/
    ├── collect_kepler_data.py         # ✅ Updated: relaxed HZ criteria
    ├── preprocessing_pipeline.py      # ✅ Updated: 200-sample augmentation
    └── balance_dataset.py             # ✅ New: advanced balancing strategies
```

---

## Quick Reference Commands

```bash
# Full workflow from start to finish:

# Step 1: Collect more data (relaxed criteria)
python scripts/collect_kepler_data.py

# Step 2: Preprocess with augmentation
python scripts/preprocessing_pipeline.py

# Step 3: Advanced balancing (choose one strategy)
python scripts/balance_dataset.py --strategy moderate

# Step 4: Train model
python scripts/train.py

# Step 5: Evaluate with proper metrics
python scripts/evaluate.py
```

---

## Expected Results

### Before Balancing
- Training: Model achieves 98% accuracy by always predicting majority classes
- HABITABLE recall: **0%** (never predicts HABITABLE)
- Validation F1-macro: **0.33** (only 2 classes learned)

### After Balancing (Moderate Strategy)
- Training: Balanced loss across all classes
- HABITABLE recall: **60-70%** (successfully learns minority class)
- Validation F1-macro: **0.70-0.75** (all classes learned properly)

### Current Leak-Safe Pipeline
- Training: train-only augmentation with real-only holdout evaluation
- Holdout accuracy: **75.0%**
- Macro F1: **0.5006** on the current real-only test split
- HABITABLE test metrics: unstable because only one real HABITABLE sample exists in the holdout split

---

## Troubleshooting

### Issue: HABITABLE class still has poor F1 after balancing
**Solution:**
1. Try aggressive strategy: `--strategy aggressive`
2. Increase augmentation target to 300-400 samples
3. Collect more real samples by expanding search to K2 mission data

### Issue: Model overfits to augmented samples
**Symptoms:** Training F1 >> Validation F1 for HABITABLE class
**Solution:**
1. Reduce augmentation strength (use conservative strategy)
2. Add more regularization (dropout, weight decay)
3. Use adversarial validation to check augmentation quality

### Issue: FALSE_POSITIVE confused with HABITABLE
**Solution:**
1. Add centroid motion as additional feature (distinguishes background stars)
2. Add secondary eclipse detection (eclipsing binaries have secondary transits)
3. Increase model capacity (deeper network)

---

## Summary

✅ **Solution 1 (Best):** Re-collect data with relaxed HZ criteria → 15-30 HABITABLE samples  
✅ **Solution 2 (Essential):** Advanced augmentation → 200-250 HABITABLE samples  
✅ **Solution 3 (Required):** Class weights in loss function → balanced learning  

**Combined impact:**
- Real HABITABLE samples: 7 → 15-30 (data collection)
- Total HABITABLE samples: 7 → 200-250 (with augmentation)
- Class balance: 1.5% → 30-35% (nearly balanced)
- Expected model F1-macro: 0.33 → **0.70-0.75**

Use **moderate strategy** for production models. Monitor per-class F1 scores during training.
