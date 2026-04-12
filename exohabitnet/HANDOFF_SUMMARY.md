# ExoHabitNet — Handoff Summary

**Date:** 2026-04-12

## Overview

This document summarizes the project state, key artifacts, reproduction steps, and recommended next steps for handing the ExoHabitNet work to a collaborator.

## Key artifacts & locations

- **Model checkpoints:**
  - models/checkpoints/best_model.pth — canonical best single checkpoint
  - models/checkpoints/best_ensemble.pth — ensembled checkpoint
  - models/checkpoints/kfold/fold_{1..5}_best.pth — per-fold checkpoints
- **Reports & metrics:**
  - reports/classification_report.md
  - reports/model_performance.json
  - reports/oof_probs.npy, reports/oof_labels.npy
  - reports/oof_calibration_results.json
  - reports/test_calibrated_results.json
  - reports/pipeline_diagram_highres.png, reports/pipeline_diagram.svg
- **Scripts / entrypoints:** see scripts/ and exohabitnet/scripts/ for the following main scripts:
  - preprocessing_pipeline.py
  - kfold_finetune.py / train.py
  - oof_aggregator.py, oof_calibrate.py, apply_calibrator_to_test.py
  - evaluate.py
  - create_ensemble_checkpoint.py, ensemble_checkpoints.py
- **Data:** data/processed_dataset.csv, data/train_dataset.csv, data/test_dataset.csv
- **Pruning log:** exohabitnet/PRUNED_FILES.md (lists files removed during cleanup)

## Environment & quick setup

Recommended: use the Python environment described in exohabitnet/requirements.txt.

Quick install and smoke-run (PowerShell):

```powershell
pip install -r exohabitnet/requirements.txt
python scripts/preprocessing_pipeline.py
python scripts/kfold_finetune.py --folds 5
python scripts/oof_aggregator.py
python scripts/oof_calibrate.py
python scripts/apply_calibrator_to_test.py
python scripts/evaluate.py
```

Note: CLI flags/config names may vary between scripts. Inspect the header or --help for each script to pass dataset paths, hyperparameters, and device options.

## Model & training summary

- Backbone: ExoHabitNet CNN (see exohabitnet/models/cnn_model.py) — 1D conv blocks + global pooling.
- Cross-validation: Stratified K-Fold (N=5).
- Sampling & augmentation: WeightedRandomSampler used; recommended sampler_scale=1.5. HAB augmentation target was set to ≈50 in the best runs.
- Loss & balancing: focal loss was used in sweeps (best gamma=2.0) with a HAB-class weight multiplier ≈3.0.
- Optimizer & schedulers: standard Adam/AdamW with early stopping and ReduceLROnPlateau in training loop.

## Evaluation & calibration

- OOF aggregation produced reports/oof_probs.npy and reports/oof_labels.npy and the PR/threshold sweep (reports/oof_threshold_result.json).
- Calibration: both Platt (logistic) and isotonic were tried on OOF; chosen calibrator and fit metadata saved in reports/oof_calibration_results.json and as pickles in reports/.
- Calibrator applied to the holdout test and saved in reports/test_calibrated_results.json.

## Ensemble

- Two ensemble helper scripts exist: exohabitnet/scripts/create_ensemble_checkpoint.py and exohabitnet/scripts/ensemble_checkpoints.py. Both were preserved by request; the canonical ensembled output is models/checkpoints/best_ensemble.pth.

## Data considerations & caveats

- HAB examples (positives) are extremely rare in this dataset (≈7 true HAB examples). Training relied on targeted augmentation and sampler tuning to mitigate extreme class imbalance — this affects reproducibility and evaluation variability.
- The canonical processed data lives in data/processed/ and the CSVs in data/.

## Git branches & recent commits

- Cleanup (pruning old artifacts) committed on branch: cleanup/prune_old_reports_2026-04-12.
- Earlier packaging branch (artifact bundle): results/package_artifacts_2026-04-12 (kept for history; large archive removed per pruning).

