"""
run_sampler_aug_grid.py
-----------------------
Simple runner to sweep sampler-scale × augmentation-target values by
invoking `kfold_finetune.py` and saving per-run reports/logs.

Usage (defaults):
    python -u exohabitnet/scripts/run_sampler_aug_grid.py

Customize:
    --sampler-scales 1.0,1.5,2.0 --aug-targets 25,50 --epochs 4
"""
import subprocess
import shlex
import itertools
from pathlib import Path
import argparse

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_combo(sampler_scale, aug_target, epochs=4, n_splits=5, batch_size=32):
    # Use an absolute path to the kfold_finetune script to avoid CWD path duplication
    base_dir = Path(__file__).resolve().parent
    kfold_script = (base_dir / "kfold_finetune.py").resolve()
    cmd = f"python -u \"{kfold_script}\" --n-splits {n_splits} --epochs {epochs} --batch-size {batch_size} --use-weighted-sampler --sampler-scale {sampler_scale} --aug-target {aug_target} --aug-noise-level 0.008"
    stamp = f"scale{sampler_scale}_aug{aug_target}_n{n_splits}_e{epochs}"
    log_file = REPORTS_DIR / f"kfold_grid_{stamp}.log"
    print(f"Running: {cmd}\nLogging to {log_file}")
    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.run(shlex.split(cmd), stdout=lf, stderr=subprocess.STDOUT)

    # rename output JSON if present
    out_json = REPORTS_DIR / "kfold_finetune_report.json"
    if out_json.exists():
        dest = REPORTS_DIR / f"kfold_finetune_report_{stamp}.json"
        if dest.exists():
            dest.unlink()
        out_json.rename(dest)
        print(f"Saved report {dest}")
    else:
        print("Warning: report not found after run. Check log for errors.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampler-scales', type=str, default="1.0,1.5,2.0")
    parser.add_argument('--aug-targets', type=str, default="25,50")
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    scales = [float(s) for s in args.sampler_scales.split(',') if s.strip()]
    targets = [int(t) for t in args.aug_targets.split(',') if t.strip()]

    for s, t in itertools.product(scales, targets):
        run_combo(s, t, epochs=args.epochs, n_splits=args.n_splits, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
