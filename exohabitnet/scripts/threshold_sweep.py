"""
threshold_sweep.py
==================
Run a precision-recall threshold sweep for the `HABITABLE` class on the held-out
test set using a saved checkpoint. Saves a PR plot, recommended threshold and
metrics to the `reports/` directory.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_recall_fscore_support, accuracy_score

# Ensure project imports resolve
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.cnn_model import ExoHabitNetCNN

# CONFIG
TEST_DATA_PATH = Path("data/test_dataset.csv")
MODEL_PATH = Path("models/checkpoints/best_model.pth")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_IDX = 0  # HABITABLE is class 0
CLASSES = ["HABITABLE", "NON_HABITABLE", "FALSE_POSITIVE"]


def load_test_set(path):
    import pandas as pd
    if not path.exists():
        raise FileNotFoundError(f"Test dataset not found at {path}")
    df = pd.read_csv(path)
    flux_cols = [c for c in df.columns if c.startswith('flux_')]
    X = df[flux_cols].values
    y = df['label_id'].values
    return X, y


def run_threshold_sweep(model_path=MODEL_PATH, test_path=TEST_DATA_PATH, class_idx=CLASS_IDX):
    print(f"Device: {DEVICE}")
    X_test, y_test = load_test_set(test_path)
    X_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    model = ExoHabitNetCNN().to(DEVICE)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    state = torch.load(model_path, map_location=DEVICE)
    if isinstance(state, dict) and ('model_state_dict' in state or 'state_dict' in state):
        state = state.get('model_state_dict', state.get('state_dict'))
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(X_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds_argmax = np.argmax(probs, axis=1)

    y_true = y_test

    # Binarize for the HAB class
    y_hab = (y_true == class_idx).astype(int)
    prob_hab = probs[:, class_idx]

    # Compute PR curve
    precision, recall, thresholds = precision_recall_curve(y_hab, prob_hab)
    avg_prec = average_precision_score(y_hab, prob_hab)

    # Compute F1 for each threshold (precision and recall arrays are len = len(thresholds)+1)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
    # Map threshold index to f1 index: threshold i corresponds to precision[i+1], recall[i+1]
    if len(thresholds) > 0:
        f1_for_thresholds = f1_scores[1:]
        best_idx = int(np.nanargmax(f1_for_thresholds))
        best_threshold = float(thresholds[best_idx])
        best_f1 = float(f1_for_thresholds[best_idx])
        best_precision = float(precision[best_idx + 1])
        best_recall = float(recall[best_idx + 1])
    else:
        # Edge case: no threshold variations (all probs identical)
        best_threshold = float(0.5)
        best_f1 = float(f1_scores[0])
        best_precision = float(precision[0])
        best_recall = float(recall[0])

    # Build adjusted multi-class predictions using HAB threshold
    adjusted_preds = np.argmax(probs, axis=1)
    mask_hab = prob_hab >= best_threshold
    adjusted_preds[mask_hab] = class_idx
    # For non-HAB where argmax was HAB but prob < threshold, pick best of remaining classes
    non_hab_idx = np.where(~mask_hab)[0]
    if len(non_hab_idx) > 0:
        adjusted_preds[non_hab_idx] = np.argmax(probs[non_hab_idx, :][:, [i for i in range(probs.shape[1]) if i != class_idx]], axis=1)
        # remap indices {0,1} -> {1,2} if class_idx==0
        if class_idx == 0:
            adjusted_preds[non_hab_idx] = adjusted_preds[non_hab_idx] + 1

    # Compute new metrics at chosen threshold
    acc_new = float(accuracy_score(y_true, adjusted_preds))
    prec_array, rec_array, f1_array, support = precision_recall_fscore_support(y_true, adjusted_preds, zero_division=0)
    macro_f1_new = float(np.mean(f1_array))

    # Save PR plot and results
    pr_png = REPORTS_DIR / f"pr_curve_class_{CLASSES[class_idx].lower()}.png"
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, lw=2, label=f'AP = {avg_prec:.3f}')
    plt.scatter([best_recall], [best_precision], color='red', zorder=5, label=f'Best F1={best_f1:.3f}\nthr={best_threshold:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({CLASSES[class_idx]})')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(pr_png, dpi=150)
    plt.close()

    result = {
        "model_path": str(model_path),
        "test_dataset": str(test_path),
        "class": CLASSES[class_idx],
        "average_precision": float(avg_prec),
        "best_threshold": best_threshold,
        "best_f1_on_pr": best_f1,
        "precision_at_best": best_precision,
        "recall_at_best": best_recall,
        "accuracy_at_best": acc_new,
        "macro_f1_at_best": macro_f1_new,
        "per_class": {
            CLASSES[i]: {
                "precision": float(prec_array[i]),
                "recall": float(rec_array[i]),
                "f1": float(f1_array[i]),
                "support": int(support[i])
            } for i in range(len(CLASSES))
        }
    }

    out_json = REPORTS_DIR / "threshold_sweep_result.json"
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Saved PR curve to {pr_png}")
    print(f"Saved threshold sweep results to {out_json}")
    print(f"Best threshold for {CLASSES[class_idx]}: {best_threshold:.4f} (F1={best_f1:.4f})")


if __name__ == '__main__':
    run_threshold_sweep()
