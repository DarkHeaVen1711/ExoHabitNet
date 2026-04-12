"""
evaluate_hab_samples.py
=======================
Run the trained `best_model.pth` on all HABITABLE samples from
`data/processed_dataset.csv` and save a JSON report and optional plots.

Usage:
  python scripts/evaluate_hab_samples.py [--data data/processed_dataset.csv] [--model models/checkpoints/best_model.pth] [--out reports/hab_predictions.json] [--plot]

"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# allow importing models from project root
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.cnn_model import ExoHabitNetCNN


DEFAULT_DATA = Path("data/processed_dataset.csv")
DEFAULT_MODEL = Path("models/checkpoints/best_model.pth")
DEFAULT_OUT = Path("reports/hab_predictions.json")

CLASSES = ["HABITABLE", "NON_HABITABLE", "FALSE_POSITIVE"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=str(DEFAULT_DATA))
    parser.add_argument('--model', type=str, default=str(DEFAULT_MODEL))
    parser.add_argument('--out', type=str, default=str(DEFAULT_OUT))
    parser.add_argument('--plot', action='store_true', help='Save small plots for each sample')
    args = parser.parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model)
    out_path = Path(args.out)

    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}")
        return
    if not model_path.exists():
        print(f"ERROR: model checkpoint not found: {model_path}")
        return

    df = pd.read_csv(data_path)

    # Select HABITABLE rows
    if 'label' in df.columns:
        df_hab = df[df['label'].str.upper() == 'HABITABLE']
    elif 'label_id' in df.columns:
        df_hab = df[df['label_id'] == 0]
    else:
        raise RuntimeError('No label column found in dataset')

    if df_hab.empty:
        print('No HABITABLE samples found in', data_path)
        return

    flux_cols = [c for c in df_hab.columns if c.startswith('flux_')]
    X = df_hab[flux_cols].values.astype(np.float32)
    fits_paths = df_hab['fits_path'].fillna('').tolist() if 'fits_path' in df_hab.columns else [''] * len(df_hab)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ExoHabitNetCNN().to(device)
    state = torch.load(model_path, map_location=device)
    # If the checkpoint contains more than state_dict (e.g., dict with keys), try common keys
    if isinstance(state, dict) and any(k.startswith('epoch') or k in ('model_state_dict','state_dict') for k in state.keys()):
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        elif 'state_dict' in state:
            state = state['state_dict']

    model.load_state_dict(state)
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        logits = model(X_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1).tolist()

    entries = []
    for i, idx in enumerate(df_hab.index.tolist()):
        entries.append({
            'original_index': int(idx),
            'fits_path': fits_paths[i],
            'predicted_id': int(preds[i]),
            'predicted_label': CLASSES[int(preds[i])],
            'probs': probs[i].tolist(),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        'n_samples': len(entries),
        'predictions': entries,
    }
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # print concise summary
    from collections import Counter
    counts = Counter([e['predicted_label'] for e in entries])
    print('Predicted label counts on HAB samples:', dict(counts))
    print(f"Wrote predictions to: {out_path}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            out_dir = Path('reports/hab_samples')
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, row in enumerate(entries):
                flux = X[i]
                plt.figure(figsize=(6, 2.5))
                plt.plot(flux, color='black')
                plt.title(f"True: HABITABLE | Pred: {row['predicted_label']}")
                plt.xlabel('phase bin')
                plt.ylabel('flux')
                plt.tight_layout()
                plt.savefig(out_dir / f"hab_{i}_{row['predicted_label']}.png", dpi=150)
                plt.close()
            print(f"Saved per-sample plots to: {out_dir}")
        except Exception as e:
            print('Plotting failed:', e)


if __name__ == '__main__':
    main()
