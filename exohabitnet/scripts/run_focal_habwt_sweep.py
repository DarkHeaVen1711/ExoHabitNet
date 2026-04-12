"""
run_focal_habwt_sweep.py
------------------------
Run a focused sweep over focal loss gamma and HAB weight multiplier
on the recommended sampler/augmentation combo. Saves per-combo JSON
reports and logs under `reports/`.

Usage:
    cd exohabitnet
    python -u scripts/run_focal_habwt_sweep.py

Adjust `GAMMAS` and `HAB_WTS` inside the script for different ranges.
"""
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / 'reports'
REPORTS.mkdir(parents=True, exist_ok=True)

# Sweep parameters
GAMMAS = [1.0, 2.0, 3.0]
HAB_WTS = [1.0, 2.0, 3.0]

# Recommended combo overrides
SAMPLER_SCALE = 1.5
AUG_TARGET = 50
N_SPLITS = 5
EPOCHS = 10
BATCH_SIZE = 32

PY = sys.executable

def run_combo(gamma, hab_wt):
    name_tag = f"gamma{gamma}_habwt{hab_wt}_n{N_SPLITS}_e{EPOCHS}"
    log_file = REPORTS / f"kfold_focal_{name_tag}.log"
    out_json = REPORTS / f"kfold_finetune_report_{name_tag}.json"

    cmd = [PY, '-u', 'scripts/kfold_finetune.py',
           '--loss', 'focal',
           '--focal-gamma', str(gamma),
           '--hab-weight-multiplier', str(hab_wt),
           '--use-weighted-sampler',
           '--sampler-scale', str(SAMPLER_SCALE),
           '--aug-target', str(AUG_TARGET),
           '--n-splits', str(N_SPLITS),
           '--epochs', str(EPOCHS),
           '--batch-size', str(BATCH_SIZE)
           ]

    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT)

    print(f"Running combo: gamma={gamma}, hab_wt={hab_wt} -> log: {log_file.name}")
    with open(log_file, 'wb') as lf:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)

    # rename default report if present
    default_report = REPORTS / 'kfold_finetune_report.json'
    if default_report.exists():
        try:
            default_report.rename(out_json)
            print(f"Saved report: {out_json.name}")
        except Exception as e:
            print(f"Failed renaming report for {name_tag}: {e}")
    else:
        print(f"Warning: expected report not found for combo {name_tag}")


def main():
    for g in GAMMAS:
        for w in HAB_WTS:
            run_combo(g, w)

    print('Sweep finished. Per-combo logs and JSON reports are under reports/.')


if __name__ == '__main__':
    main()
