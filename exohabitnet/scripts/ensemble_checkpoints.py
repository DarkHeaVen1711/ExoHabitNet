"""
ensemble_checkpoints.py
------------------------
Average per-fold `fold_*_best.pth` state_dicts into a single ensemble
checkpoint saved as `models/checkpoints/best_model.pth`.

Usage:
    cd exohabitnet
    python -u scripts/ensemble_checkpoints.py
"""
from pathlib import Path
import argparse
import sys
import torch
import numpy as np


def load_state_dict(path: Path):
    st = torch.load(path, map_location="cpu")
    # unwrap common checkpoint wrappers
    if isinstance(st, dict) and ("model_state_dict" in st or "state_dict" in st):
        st = st.get("model_state_dict", st.get("state_dict"))
    if not isinstance(st, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {path}")
    # convert tensors to CPU float32
    return {k: v.clone().detach().cpu().to(torch.float32) for k, v in st.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, default="models/checkpoints/kfold")
    parser.add_argument("--out", type=str, default="models/checkpoints/best_model.pth")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    out_path = Path(args.out)
    ckpts = sorted(ckpt_dir.glob("fold_*_best.pth"))
    if len(ckpts) == 0:
        print(f"No per-fold checkpoints found in {ckpt_dir}")
        sys.exit(1)

    print(f"Found {len(ckpts)} fold checkpoints. Loading...")
    states = [load_state_dict(p) for p in ckpts]

    # ensure consistent keys
    ref_keys = list(states[0].keys())
    for i, st in enumerate(states[1:], start=1):
        if set(st.keys()) != set(ref_keys):
            print(f"Key mismatch between {ckpts[0]} and {ckpts[i]}")
            sys.exit(1)

    # average parameters
    avg_state = {}
    n = len(states)
    for k in ref_keys:
        arrs = [s[k].cpu().numpy().astype(np.float64) for s in states]
        stacked = np.stack(arrs, axis=0)
        mean_arr = stacked.mean(axis=0)
        avg_state[k] = torch.from_numpy(mean_arr.astype(np.float32))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(avg_state, out_path)
    print(f"Saved ensembled checkpoint to {out_path} (averaged {n} folds)")


if __name__ == '__main__':
    main()
