"""
apply_calibrator_to_test.py
--------------------------
Load the recommended calibrator from `reports/oof_calibration_results.json`,
run the trained model on the holdout `data/test_dataset.csv`, and save the
calibrated HAB probabilities to `reports/test_calibrated_probs.npy`.

Also writes `reports/test_probs_raw.npy`, `reports/test_labels.npy`, and
`reports/test_calibrated_results.json` with summary metrics.

Run from project root (exohabitnet/exohabitnet):
    $env:PYTHONPATH = "<repo>/exohabitnet"; python -u scripts/apply_calibrator_to_test.py
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_recall_fscore_support, accuracy_score

from models.cnn_model import ExoHabitNetCNN

REPORTS = Path('reports')
REPORTS.mkdir(exist_ok=True)

TEST_DATA = Path('data/test_dataset.csv')
MODEL_PATH = Path('models/checkpoints/best_model.pth')


def load_calibration_choice():
    cfg = REPORTS / 'oof_calibration_results.json'
    if not cfg.exists():
        raise FileNotFoundError('Missing reports/oof_calibration_results.json — run oof_calibrate.py first')
    j = json.loads(cfg.read_text())
    sel = j.get('selected', {})
    method = sel.get('method', 'raw')
    details = sel.get('details', {})
    return method, details


def infer_test_probs(device, batch_size=64):
    if not TEST_DATA.exists():
        raise FileNotFoundError('Test dataset not found: data/test_dataset.csv')
    df = pd.read_csv(TEST_DATA)
    flux_cols = [c for c in df.columns if c.startswith('flux_')]
    X = df[flux_cols].values
    y = df['label_id'].values

    model = ExoHabitNetCNN().to(device)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Model checkpoint not found: {MODEL_PATH}')
    state = torch.load(MODEL_PATH, map_location=device)
    if isinstance(state, dict) and ('model_state_dict' in state or 'state_dict' in state):
        state = state.get('model_state_dict', state.get('state_dict'))
    model.load_state_dict(state)
    model.eval()

    probs = np.zeros((len(X), 3), dtype=float)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).unsqueeze(1).to(device)
            logits = model(xb)
            p = F.softmax(logits, dim=1).cpu().numpy()
            probs[i:i+len(p)] = p

    return probs, y


def compute_best_threshold(probs, y):
    precision, recall, thresholds = precision_recall_curve(y, probs)
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    if thresholds.size > 0:
        best_idx = int(np.nanargmax(f1[1:]))
        thr = float(thresholds[best_idx])
        best_f1 = float(f1[best_idx + 1])
        best_prec = float(precision[best_idx + 1])
        best_rec = float(recall[best_idx + 1])
    else:
        thr = 0.5
        best_f1 = float(f1[0])
        best_prec = float(precision[0])
        best_rec = float(recall[0])
    return {'threshold': thr, 'f1': best_f1, 'precision': best_prec, 'recall': best_rec}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=str(MODEL_PATH))
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    method, details = load_calibration_choice()

    # Inference
    probs_raw, labels = infer_test_probs(device, batch_size=args.batch_size)
    np.save(REPORTS / 'test_probs_raw.npy', probs_raw)
    np.save(REPORTS / 'test_labels.npy', labels)

    # HAB probabilities
    prob_hab = probs_raw[:, 0]
    prob_hab = np.clip(prob_hab, 1e-8, 1 - 1e-8)

    # Apply calibrator
    if method == 'raw':
        calibrated = prob_hab.copy()
        used = 'raw'
    elif method == 'platt':
        pfile = REPORTS / 'oof_calibrator_platt.pkl'
        if not pfile.exists():
            raise FileNotFoundError('Platt calibrator not found: reports/oof_calibrator_platt.pkl')
        with open(pfile, 'rb') as f:
            platt = pickle.load(f)
        calibrated = platt.predict_proba(prob_hab.reshape(-1, 1))[:, 1]
        used = 'platt'
    elif method == 'isotonic':
        pfile = REPORTS / 'oof_calibrator_isotonic.pkl'
        if not pfile.exists():
            raise FileNotFoundError('Isotonic calibrator not found: reports/oof_calibrator_isotonic.pkl')
        with open(pfile, 'rb') as f:
            iso = pickle.load(f)
        calibrated = iso.predict(prob_hab)
        used = 'isotonic'
    else:
        raise ValueError(f'Unknown calibration method: {method}')

    np.save(REPORTS / 'test_calibrated_probs.npy', calibrated)

    # Evaluate calibrated scores on test labels
    y_hab = (labels == 0).astype(int)
    ap = float(average_precision_score(y_hab, calibrated))
    best = compute_best_threshold(calibrated, y_hab)

    # Confusion / classification at threshold
    preds_bin = (calibrated >= best['threshold']).astype(int)
    prec_array, rec_array, f1_array, support = precision_recall_fscore_support(y_hab, preds_bin, zero_division=0)
    acc = float(accuracy_score(y_hab, preds_bin))

    out = {
        'method': used,
        'average_precision': ap,
        'selected_threshold': best['threshold'],
        'selected_f1': best['f1'],
        'precision_at_selected': best['precision'],
        'recall_at_selected': best['recall'],
        'binary_accuracy_at_selected': acc,
        'support': int(support.sum())
    }

    with open(REPORTS / 'test_calibrated_results.json', 'w') as f:
        json.dump(out, f, indent=2)

    print('Saved calibrated HAB probabilities to reports/test_calibrated_probs.npy')
    print('Saved test calibration summary to reports/test_calibrated_results.json')


if __name__ == '__main__':
    main()
