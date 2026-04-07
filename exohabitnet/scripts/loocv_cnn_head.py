"""
loocv_cnn_head.py
-----------------
Extract features from a pretrained CNN encoder and run LOOCV using a lightweight
multinomial logistic regression head on those features.

Input: `data/processed_dataset.csv` (real processed samples)
Pretrained encoder: `models/checkpoints/encoder_pretrained.pth` (created by pretrain_encoder.py)

Outputs: prints metrics and writes a JSON report to `reports/loocv_cnn_head_report.json`.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, recall_score, classification_report, confusion_matrix

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.cnn_model import ExoHabitNetCNN

DATA_PATH = Path("data/processed_dataset.csv")
ENCODER_PATH = Path("models/checkpoints/encoder_pretrained.pth")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = REPORT_DIR / "loocv_cnn_head_report.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_features(model, X_array, device=DEVICE, batch_size=64):
    """Run the CNN up to the global-average-pool and return 2D features (N, 256)."""
    model.eval()
    features = []
    with torch.no_grad():
        for i in range(0, len(X_array), batch_size):
            batch = X_array[i:i+batch_size]
            t = torch.tensor(batch, dtype=torch.float32).unsqueeze(1).to(device)  # (B,1,1024)
            # replicate forward until adaptive_pool
            x = model.conv1(t)
            x = model.bn1(x)
            x = torch.relu(x)
            x = model.pool1(x)
            x = model.dropout1(x)

            x = model.conv2(x)
            x = model.bn2(x)
            x = torch.relu(x)
            x = model.pool2(x)
            x = model.dropout2(x)

            x = model.conv3(x)
            x = model.bn3(x)
            x = torch.relu(x)
            x = model.dropout3(x)

            # adaptive pool
            x = model.adaptive_pool(x).squeeze(-1)  # (B, 256)
            features.append(x.cpu().numpy())
    return np.vstack(features)


def main():
    if not DATA_PATH.exists():
        print(f"ERROR: Processed data not found at {DATA_PATH}. Run preprocessing first.")
        sys.exit(1)
    if not ENCODER_PATH.exists():
        print(f"ERROR: Pretrained encoder not found at {ENCODER_PATH}. Run pretrain_encoder.py first.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    flux_cols = [c for c in df.columns if c.startswith('flux_')]
    X = df[flux_cols].values
    y = df['label_id'].values

    # Load model and pretrained weights
    model = ExoHabitNetCNN().to(DEVICE)
    model.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    print("Loaded pretrained encoder from:", ENCODER_PATH)

    # Extract features
    print("Extracting CNN encoder features for all samples...")
    feats = extract_features(model, X, device=DEVICE)
    print("Features shape:", feats.shape)

    # LOOCV with logistic regression head
    loo = LeaveOneOut()
    preds = np.zeros(len(y), dtype=int)

    for train_idx, test_idx in loo.split(feats):
        X_train, X_test = feats[train_idx], feats[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight='balanced', max_iter=1000)
        clf.fit(X_train, y_train)
        p = clf.predict(X_test)
        preds[test_idx] = p

    # Metrics
    macro_f1 = f1_score(y, preds, average='macro')
    # recall for HAB class (label 0)
    recall_per_class = recall_score(y, preds, average=None, labels=[0,1,2])
    hab_recall = float(recall_per_class[0])

    report = classification_report(y, preds, target_names=['HABITABLE','NON_HABITABLE','FALSE_POSITIVE'], output_dict=True)
    cm = confusion_matrix(y, preds).tolist()

    summary = {
        'n_samples': int(len(y)),
        'macro_f1': float(macro_f1),
        'hab_recall': hab_recall,
        'classification_report': report,
        'confusion_matrix': cm
    }

    print('\nLOOCV results:')
    print(f"  Samples: {len(y)}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  HAB Recall: {hab_recall:.4f}")
    print('\nPer-class report:')
    print(classification_report(y, preds, target_names=['HABITABLE','NON_HABITABLE','FALSE_POSITIVE']))

    with open(OUT_JSON, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved LOOCV report to {OUT_JSON}")

if __name__ == '__main__':
    main()
