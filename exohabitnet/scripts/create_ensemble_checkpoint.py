"""
create_ensemble_checkpoint.py
-----------------------------
Average model weights from per-fold checkpoints saved under
`models/checkpoints/kfold/` and write an ensembled checkpoint to
`models/checkpoints/best_ensemble.pth` and `models/checkpoints/best_hab_detector.pth`.

This script is conservative: it averages only keys present in all
fold checkpoints and preserves integer dtypes by rounding.
"""
from pathlib import Path
import torch
import sys


ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / 'models' / 'checkpoints' / 'kfold'
OUT_DIR = ROOT / 'models' / 'checkpoints'
OUT_DIR.mkdir(parents=True, exist_ok=True)
ENSEMBLE_PATH = OUT_DIR / 'best_ensemble.pth'
ALIAS_PATH = OUT_DIR / 'best_hab_detector.pth'


def load_state(ckpt_path: Path):
    state = torch.load(str(ckpt_path), map_location='cpu')
    # unwrap common checkpoint wrappers
    if isinstance(state, dict):
        if 'model_state_dict' in state:
            return state['model_state_dict']
        if 'state_dict' in state:
            return state['state_dict']
        # otherwise assume it's already a state_dict
        return state
    raise RuntimeError(f'Unrecognized checkpoint format: {ckpt_path}')


def main():
    ckpts = sorted(CKPT_DIR.glob('fold_*_best.pth'))
    if not ckpts:
        print(f'No fold checkpoints found in {CKPT_DIR}', file=sys.stderr)
        sys.exit(1)

    states = [load_state(p) for p in ckpts]
    keys = set(states[0].keys())
    for s in states[1:]:
        keys &= set(s.keys())

    if not keys:
        print('No common parameter keys found across checkpoints', file=sys.stderr)
        sys.exit(1)

    if len(keys) != len(states[0].keys()):
        print('Warning: not all keys present in every checkpoint; averaging intersection of keys')

    avg_state = {}
    for k in sorted(keys):
        vals = []
        for s in states:
            v = s[k]
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            vals.append(v.float())
        stacked = torch.stack(vals, dim=0)
        mean = stacked.mean(dim=0)

        orig = states[0][k]
        if orig.dtype.is_floating_point:
            avg_state[k] = mean.to(orig.dtype)
        else:
            # integer/bool buffers (e.g. num_batches_tracked) -> round and cast
            avg_state[k] = mean.round().to(orig.dtype)

    torch.save(avg_state, str(ENSEMBLE_PATH))
    # also write with alias for downstream scripts
    torch.save(avg_state, str(ALIAS_PATH))

    print(f'Saved ensemble checkpoint: {ENSEMBLE_PATH}')
    print(f'Also saved alias: {ALIAS_PATH}')


if __name__ == '__main__':
    main()
