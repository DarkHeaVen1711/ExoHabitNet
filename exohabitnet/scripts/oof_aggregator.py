"""
oof_aggregator.py
-----------------
Generate out-of-fold (OOF) probabilities by loading per-fold best
checkpoints saved by `kfold_finetune.py` and evaluating each fold's
validation indices. Produces PR curve and recommended threshold based on
the pooled OOF probabilities for the HAB class.

Outputs:
- reports/oof_probs.npy
- reports/oof_labels.npy
- reports/oof_pr_curve.png
- reports/oof_threshold_result.json

Run (after training chosen combo with kfold_finetune to create per-fold ckpts):
    python -u exohabitnet/scripts/oof_aggregator.py --n-splits 5
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold

from models.cnn_model import ExoHabitNetCNN

DATA_PATH = Path('data/processed_dataset.csv')
CKPT_DIR = Path('models/checkpoints/kfold')
REPORTS = Path('reports')
REPORTS.mkdir(exist_ok=True)


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed data not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    flux_cols = [c for c in df.columns if c.startswith('flux_')]
    X = df[flux_cols].values
    y = df['label_id'].values
    return X, y


def run_oof(n_splits=5, random_state=42, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y = load_data()
    n = len(y)
    oof_probs = np.zeros((n, 3), dtype=float)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        fold_id = fold_idx + 1
        ckpt_path = CKPT_DIR / f'fold_{fold_id}_best.pth'
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

        model = ExoHabitNetCNN().to(device)
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and ('model_state_dict' in state or 'state_dict' in state):
            state = state.get('model_state_dict', state.get('state_dict'))
        model.load_state_dict(state)
        model.eval()

        # evaluate val indices in batches
        with torch.no_grad():
            for i in range(0, len(val_idx), batch_size):
                batch_inds = val_idx[i:i+batch_size]
                Xb = torch.tensor(X[batch_inds], dtype=torch.float32).unsqueeze(1).to(device)
                logits = model(Xb)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                oof_probs[batch_inds] = probs

    # Save arrays
    np.save(REPORTS / 'oof_probs.npy', oof_probs)
    np.save(REPORTS / 'oof_labels.npy', y)
    print('Saved OOF arrays to reports/')

    # PR curve for HAB class (index 0)
    y_hab = (y == 0).astype(int)
    prob_hab = oof_probs[:, 0]
    precision, recall, thresholds = precision_recall_curve(y_hab, prob_hab)
    avg_prec = average_precision_score(y_hab, prob_hab)

    # Compute best threshold by F1
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
    if len(thresholds) > 0:
        f1_for_thresholds = f1_scores[1:]
        best_idx = int(np.nanargmax(f1_for_thresholds))
        best_threshold = float(thresholds[best_idx])
        best_f1 = float(f1_for_thresholds[best_idx])
        best_precision = float(precision[best_idx + 1])
        best_recall = float(recall[best_idx + 1])
    else:
        best_threshold = 0.5
        best_f1 = float(f1_scores[0])
        best_precision = float(precision[0])
        best_recall = float(recall[0])

    # Save PR plot
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, lw=2, label=f'AP = {avg_prec:.3f}')
    plt.scatter([best_recall], [best_precision], color='red', zorder=5, label=f'Best F1={best_f1:.3f}\nthr={best_threshold:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('OOF Precision-Recall Curve (HAB)')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(REPORTS / 'oof_pr_curve.png', dpi=150)
    plt.close()

    # Create result JSON
    prec_array, rec_array, f1_array, support = precision_recall_fscore_support(y, (prob_hab >= best_threshold).astype(int), zero_division=0)
    overall_acc = accuracy_score(y, np.argmax(oof_probs, axis=1))
    result = {
        'average_precision': float(avg_prec),
        'best_threshold': best_threshold,
        'best_f1_on_pr': best_f1,
        'precision_at_best': best_precision,
        'recall_at_best': best_recall,
        'accuracy_at_best': float(overall_acc),
        'per_class': {
            'HABITABLE': {'precision': float(prec_array[0]), 'recall': float(rec_array[0]), 'f1': float(f1_array[0]), 'support': int(support[0])},
            'NON_HABITABLE': {'precision': float(prec_array[1]), 'recall': float(rec_array[1]), 'f1': float(f1_array[1]), 'support': int(support[1])},
            'FALSE_POSITIVE': {'precision': float(prec_array[2]), 'recall': float(rec_array[2]), 'f1': float(f1_array[2]), 'support': int(support[2])}
        }
    }
    with open(REPORTS / 'oof_threshold_result.json', 'w') as f:
        json.dump(result, f, indent=2)

    print('Saved OOF threshold result to reports/oof_threshold_result.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    run_oof(n_splits=args.n_splits, random_state=args.random_state, batch_size=args.batch_size)
