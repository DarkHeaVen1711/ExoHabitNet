"""
oof_calibrate.py
----------------
Fit Platt (logistic) and Isotonic calibrators on OOF HAB probabilities,
compare PR performance, and save calibrated probabilities, pickles, and
summary JSON + plots.

Outputs (saved under `reports/`):
- oof_calibration_results.json
- oof_probs_calibrated_platt.npy
- oof_probs_calibrated_isotonic.npy
- oof_calibrator_platt.pkl
- oof_calibrator_isotonic.pkl
- oof_calibration_pr_comparison.png

Run:
    cd exohabitnet
    python -u scripts/oof_calibrate.py
"""
import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score


REPORTS = Path('reports')
REPORTS.mkdir(exist_ok=True)


def compute_best_threshold(probs, y):
    precision, recall, thresholds = precision_recall_curve(y, probs)
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    if thresholds.size > 0:
        # thresholds aligns with precision[1:] / recall[1:]
        best_idx = int(np.nanargmax(f1[1:]))
        best_thr = float(thresholds[best_idx])
        best_f1 = float(f1[best_idx + 1])
        best_prec = float(precision[best_idx + 1])
        best_rec = float(recall[best_idx + 1])
    else:
        best_thr = 0.5
        best_f1 = float(f1[0])
        best_prec = float(precision[0])
        best_rec = float(recall[0])
    return {'threshold': best_thr, 'f1': best_f1, 'precision': best_prec, 'recall': best_rec}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports-dir', type=str, default='reports')
    args = parser.parse_args()

    reports = Path(args.reports_dir)
    probs_path = reports / 'oof_probs.npy'
    labels_path = reports / 'oof_labels.npy'
    if not probs_path.exists() or not labels_path.exists():
        raise FileNotFoundError('OOF arrays not found in reports/; run oof_aggregator first')

    oof_probs = np.load(probs_path)
    y = np.load(labels_path)
    # HAB class index 0
    y_hab = (y == 0).astype(int)
    prob_raw = oof_probs[:, 0]
    prob_raw = np.clip(prob_raw, 1e-6, 1 - 1e-6)

    results = {}

    # raw performance
    results['raw'] = {}
    results['raw']['average_precision'] = float(average_precision_score(y_hab, prob_raw))
    results['raw'].update(compute_best_threshold(prob_raw, y_hab))

    # Platt scaling (LogisticRegression on prob -> calibrated prob)
    platt = LogisticRegression(max_iter=2000)
    X = prob_raw.reshape(-1, 1)
    platt.fit(X, y_hab)
    platt_probs = platt.predict_proba(X)[:, 1]
    np.save(reports / 'oof_probs_calibrated_platt.npy', platt_probs)
    with open(reports / 'oof_calibrator_platt.pkl', 'wb') as f:
        pickle.dump(platt, f)
    results['platt'] = {}
    results['platt']['average_precision'] = float(average_precision_score(y_hab, platt_probs))
    results['platt'].update(compute_best_threshold(platt_probs, y_hab))

    # Isotonic regression
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(prob_raw, y_hab)
    iso_probs = iso.predict(prob_raw)
    np.save(reports / 'oof_probs_calibrated_isotonic.npy', iso_probs)
    with open(reports / 'oof_calibrator_isotonic.pkl', 'wb') as f:
        pickle.dump(iso, f)
    results['isotonic'] = {}
    results['isotonic']['average_precision'] = float(average_precision_score(y_hab, iso_probs))
    results['isotonic'].update(compute_best_threshold(iso_probs, y_hab))

    # Pick better calibrator by best F1 on PR
    best_name = max(('raw', 'platt', 'isotonic'), key=lambda k: results[k]['f1'])
    results['selected'] = {'method': best_name, 'details': results[best_name]}

    # Save JSON
    with open(reports / 'oof_calibration_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # PR comparison plot
    plt.figure(figsize=(7, 6))
    for name, probs, color in (('raw', prob_raw, 'C0'), ('platt', platt_probs, 'C1'), ('isotonic', iso_probs, 'C2')):
        prec, rec, _ = precision_recall_curve(y_hab, probs)
        ap = average_precision_score(y_hab, probs)
        plt.plot(rec, prec, label=f'{name} (AP={ap:.3f})', color=color)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('OOF PR: raw vs Platt vs Isotonic (HAB)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(reports / 'oof_calibration_pr_comparison.png', dpi=150)
    plt.close()

    print('Saved calibration results and pickles to', reports)


if __name__ == '__main__':
    main()
