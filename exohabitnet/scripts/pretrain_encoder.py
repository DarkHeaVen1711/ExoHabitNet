"""
pretrain_encoder.py
-------------------
Quick supervised pretraining of the ExoHabitNet CNN on `data/train_dataset.csv`.
Saves the model weights to `models/checkpoints/encoder_pretrained.pth`.

This is intentionally short (few epochs) to provide a reasonable encoder for LOOCV head-only experiments.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.cnn_model import ExoHabitNetCNN

# Config
TRAIN_DATA_PATH = Path("data/train_dataset.csv")
CHECKPOINT_DIR = Path("models/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = CHECKPOINT_DIR / "encoder_pretrained.pth"

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-5
RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KeplerFluxDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,1024)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def compute_class_weights(y):
    counts = np.bincount(y, minlength=3)
    total = counts.sum()
    weights = [total / (3 * max(int(c), 1)) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def main():
    print(f"Using device: {DEVICE}")
    if not TRAIN_DATA_PATH.exists():
        print(f"ERROR: Train data not found at {TRAIN_DATA_PATH}. Run preprocessing first.")
        sys.exit(1)

    df = pd.read_csv(TRAIN_DATA_PATH)
    flux_cols = [c for c in df.columns if c.startswith('flux_')]
    X = df[flux_cols].values
    y = df['label_id'].values

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )

    train_loader = DataLoader(KeplerFluxDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(KeplerFluxDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = ExoHabitNetCNN().to(DEVICE)

    class_weights = compute_class_weights(y_tr).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_f1 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(batch_y.cpu().numpy())

        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        val_loss /= len(val_loader.dataset)

        # compute metrics
        try:
            from sklearn.metrics import f1_score
            val_f1 = f1_score(val_targets, val_preds, average='macro')
        except Exception:
            val_f1 = 0.0

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1:.4f}")

        # save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), OUT_PATH)
            print(f"  [*] Saved encoder checkpoint ({OUT_PATH})")

    print("Pretraining complete. Best Val Macro-F1:", best_val_f1)

if __name__ == '__main__':
    main()
