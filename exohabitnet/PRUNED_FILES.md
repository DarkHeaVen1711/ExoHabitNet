Pruned files log (performed on 2026-04-12)
=========================================

Purpose: remove older grid/focus run artifacts and logs to simplify repository for handoff.

Kept: core scripts, the final trained checkpoints, and canonical reports (e.g., `model_performance.json`, `classification_report.md`, `grid_summary.json`, `grid_recommendation.json`, `oof_*`, `test_calibrated_results.json`, `pipeline_diagram*`).

Removed files (older experiments / logs):

- reports/kfold_finetune_report_gamma1.0_habwt1.0_n5_e10.json
- reports/kfold_finetune_report_gamma1.0_habwt2.0_n5_e10.json
- reports/kfold_finetune_report_gamma1.0_habwt3.0_n5_e10.json
- reports/kfold_finetune_report_gamma2.0_habwt1.0_n5_e10.json
- reports/kfold_finetune_report_gamma2.0_habwt2.0_n5_e10.json
- reports/kfold_finetune_report_gamma3.0_habwt1.0_n5_e10.json
- reports/kfold_finetune_report_gamma3.0_habwt2.0_n5_e10.json
- reports/kfold_finetune_report_gamma3.0_habwt3.0_n5_e10.json

- reports/kfold_finetune_report_scale1.0_aug25_n5_e4.json
- reports/kfold_finetune_report_scale1.0_aug50_n5_e4.json
- reports/kfold_finetune_report_scale1.5_aug25_n5_e4.json
- reports/kfold_finetune_report_scale2.0_aug25_n5_e4.json
- reports/kfold_finetune_report_scale2.0_aug50_n5_e4.json
- reports/kfold_finetune_report_scale_1.0.json
- reports/kfold_finetune_report_scale_2.0.json
- reports/kfold_finetune_report_scale_3.0.json

- reports/kfold_focal_gamma1.0_habwt1.0_n5_e10.log
- reports/kfold_focal_gamma1.0_habwt2.0_n5_e10.log
- reports/kfold_focal_gamma1.0_habwt3.0_n5_e10.log
- reports/kfold_focal_gamma2.0_habwt1.0_n5_e10.log
- reports/kfold_focal_gamma2.0_habwt2.0_n5_e10.log
- reports/kfold_focal_gamma2.0_habwt3.0_n5_e10.log
- reports/kfold_focal_gamma3.0_habwt1.0_n5_e10.log
- reports/kfold_focal_gamma3.0_habwt2.0_n5_e10.log
- reports/kfold_focal_gamma3.0_habwt3.0_n5_e10.log

- reports/kfold_grid_scale1.0_aug25_n5_e4.log
- reports/kfold_grid_scale1.0_aug50_n5_e4.log
- reports/kfold_grid_scale1.5_aug25_n5_e4.log
- reports/kfold_grid_scale1.5_aug50_n5_e4.log
- reports/kfold_grid_scale2.0_aug25_n5_e4.log
- reports/kfold_grid_scale2.0_aug50_n5_e4.log

- reports/kfold_sampler_scale_1.0.log
- reports/kfold_sampler_scale_1.5.log
- reports/kfold_sampler_scale_2.0.log
- reports/kfold_sampler_scale_3.0.log
- reports/kfold_weighted_sampler_run.log

Notes:
- Retained the recommended run JSONs and final artifacts (e.g., `kfold_finetune_report_scale1.5_aug50_n5_e4.json`, `kfold_finetune_report_gamma2.0_habwt3.0_n5_e10.json`).
- If you'd like more aggressive pruning (remove older OOF arrays, etc.) say so and I'll prepare a list for confirmation.

Branch & commit: these deletions and documentation updates will be committed on a cleanup branch named `cleanup/prune_old_reports_2026-04-12`.
