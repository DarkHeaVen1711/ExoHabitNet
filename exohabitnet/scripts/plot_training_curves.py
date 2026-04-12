"""
plot_training_curves.py
======================
Read TensorBoard event files from the experiment run directory and save
a PNG with training and validation loss curves (and a small JSON summary).

Usage:
    python scripts/plot_training_curves.py --logdir runs/exohabitnet_experiment

If no --logdir is provided the default above is used.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_scalars(logdir: Path):
    # Search recursively for event files in the logdir and its immediate children.
    train_events_all = []
    val_events_all = []
    found_tags = set()

    # candidate dirs: the top-level logdir and its subdirectories
    candidate_dirs = [logdir] + [p for p in logdir.iterdir() if p.is_dir()]

    for run_dir in candidate_dirs:
        # check for event files in this directory
        event_files = list(run_dir.glob('events.out.tfevents.*'))
        if not event_files:
            continue

        ea = EventAccumulator(str(run_dir), size_guidance={'scalars': 0})
        try:
            ea.Reload()
        except Exception:
            continue

        tags = ea.Tags().get('scalars', [])
        found_tags.update(tags)

        # try to find train/val loss tags here
        train_tag = None
        val_tag = None
        for t in tags:
            low = t.lower()
            if 'loss' in low and 'train' in low:
                train_tag = t
            if 'loss' in low and 'val' in low:
                val_tag = t

        if train_tag is None and 'Loss/train' in tags:
            train_tag = 'Loss/train'
        if val_tag is None and 'Loss/val' in tags:
            val_tag = 'Loss/val'

        # If tags not explicitly carrying 'train'/'val', infer from directory name
        dirname = run_dir.name.lower()
        if (train_tag is None or val_tag is None) and 'loss' in ' '.join([t.lower() for t in tags]):
            # find the scalar tag that contains 'loss' (usually just 'Loss')
            loss_tag = next((t for t in tags if 'loss' in t.lower()), None)
            if loss_tag:
                if 'train' in dirname:
                    train_tag = loss_tag
                if 'val' in dirname:
                    val_tag = loss_tag

        # If both tags found in this run_dir, extract scalars
        if train_tag and val_tag:
            train_events_all.extend(ea.Scalars(train_tag))
            val_events_all.extend(ea.Scalars(val_tag))
        else:
            # If only one of them is present, add it accordingly (handles Loss_train / Loss_val folders)
            loss_tag = next((t for t in tags if 'loss' in t.lower()), None)
            if loss_tag:
                if 'train' in dirname:
                    train_events_all.extend(ea.Scalars(loss_tag))
                if 'val' in dirname:
                    val_events_all.extend(ea.Scalars(loss_tag))

    # If nothing collected, show discovered tags for debugging
    if not train_events_all or not val_events_all:
        raise RuntimeError(f"Unable to find train/val loss scalar tags under {logdir}. Found tags: {sorted(list(found_tags))}")

    # Combine and sort by step
    def to_arrays(events):
        if not events:
            return np.array([]), np.array([])
        steps = np.array([e.step for e in events])
        vals = np.array([e.value for e in events])
        order = np.argsort(steps)
        return steps[order], vals[order]

    train_steps, train_vals = to_arrays(train_events_all)
    val_steps, val_vals = to_arrays(val_events_all)

    # Use tag names from any of the found scalars (best effort)
    sample_tag = next(iter(found_tags)) if found_tags else 'Loss'
    train_tag_name = 'Loss/train'
    val_tag_name = 'Loss/val'

    return (train_steps, train_vals, train_tag_name), (val_steps, val_vals, val_tag_name)


def plot_and_save(train, val, out_png: Path, out_json: Path):
    (t_steps, t_vals, t_tag) = train
    (v_steps, v_vals, v_tag) = val

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    plt.plot(t_steps, t_vals, marker='o', label='Train Loss', color='#1f77b4')
    plt.plot(v_steps, v_vals, marker='o', label='Val Loss', color='#ff7f0e')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    # Small JSON summary
    best_idx = int(np.argmin(v_vals)) if len(v_vals) > 0 else None
    summary = {
        'train_steps': t_steps.tolist(),
        'train_loss_last': float(t_vals[-1]) if len(t_vals) > 0 else None,
        'val_steps': v_steps.tolist(),
        'val_loss_last': float(v_vals[-1]) if len(v_vals) > 0 else None,
        'val_loss_min': float(v_vals.min()) if len(v_vals) > 0 else None,
        'val_loss_min_epoch': int(v_steps[best_idx]) if best_idx is not None else None,
        'train_tag': t_tag,
        'val_tag': v_tag,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='runs/exohabitnet_experiment')
    parser.add_argument('--out', type=str, default='reports/training_loss.png')
    parser.add_argument('--summary', type=str, default='reports/training_loss_summary.json')
    args = parser.parse_args()

    logdir = Path(args.logdir)
    if not logdir.exists():
        raise FileNotFoundError(f"Log directory not found: {logdir}")

    train, val = extract_scalars(logdir)
    summary = plot_and_save(train, val, Path(args.out), Path(args.summary))
    print(f"Saved loss plot to: {args.out}")
    print(f"Saved summary to: {args.summary}")


if __name__ == '__main__':
    main()
