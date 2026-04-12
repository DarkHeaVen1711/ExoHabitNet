"""
kfold_finetune.py
------------------
Run Stratified K-Fold fine-tuning of the full ExoHabitNet CNN.

- Uses `data/processed_dataset.csv` (real samples only).
- For each fold: augment HAB samples in the training fold (safe dynamic target),
  train the full CNN (optionally initialize from pretrained encoder), and
  evaluate on the fold's validation set.
- Saves per-fold checkpoints to `models/checkpoints/kfold/` and a JSON report
  to `reports/kfold_finetune_report.json`.

This script is intentionally conservative with augmentation and early stopping.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, classification_report

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.cnn_model import ExoHabitNetCNN
from utils.losses import FocalLoss

# Config
DATA_PATH = Path("data/processed_dataset.csv")
PRETRAIN_PATH = Path("models/checkpoints/encoder_pretrained.pth")
CHECKPOINT_DIR = Path("models/checkpoints/kfold")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = REPORT_DIR / "kfold_finetune_report.json"

N_SPLITS = 5
EPOCHS = 20
PATIENCE = 7
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-5
RANDOM_STATE = 42
NOISE_LEVEL = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_WEIGHTED_SAMPLER = False
SAMPLER_SCALE = 1.0

# Augmentation overrides (can be set via CLI)
AUG_TARGET = None
AUG_NOISE_LEVEL = None

# Loss options (can be overridden via CLI args at runtime)
LOSS_TYPE = 'ce'            # 'ce' or 'focal'
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = None
HAB_WEIGHT_MULTIPLIER = 1.0

class KeplerFluxDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,1024)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def augment_habitable(X_train, y_train, target_count, noise_level=NOISE_LEVEL, rng_seed=RANDOM_STATE):
    rng = np.random.default_rng(rng_seed)
    hab_idx = np.where(y_train == 0)[0]
    n_real = len(hab_idx)
    n_needed = max(0, target_count - n_real)
    if n_real == 0 or n_needed == 0:
        return X_train, y_train

    synthetic = []
    for i in range(n_needed):
        src = X_train[hab_idx[i % n_real]]
        noise = rng.normal(0, noise_level, size=src.shape)
        synthetic.append(src + noise)

    if synthetic:
        X_aug = np.vstack([X_train, np.vstack(synthetic)])
        y_aug = np.concatenate([y_train, np.zeros(len(synthetic), dtype=int)])
    else:
        X_aug, y_aug = X_train, y_train

    # Shuffle
    perm = np.random.default_rng(rng_seed + 1).permutation(len(y_aug))
    return X_aug[perm], y_aug[perm]


def compute_class_weights(y):
    counts = np.bincount(y, minlength=3)
    total = counts.sum()
    weights = [total / (3 * max(int(c), 1)) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def train_one_fold(fold_id, X_train, y_train, X_val, y_val, init_weights_path=None, aug_target_override=None, aug_noise_override=None):
    print(f"\n=== Fold {fold_id+1} / {N_SPLITS} ===")
    # Compute augmentation target based on HAB in this fold (or use override)
    n_hab_train = int((y_train == 0).sum())
    if aug_target_override is not None:
        target_count = int(aug_target_override)
    else:
        target_count = int(min(50, max(20, n_hab_train * 5)))
    print(f"Train HAB (real): {n_hab_train} | Augmentation target: {target_count}")

    # decide noise level (override or default)
    noise_level = aug_noise_override if aug_noise_override is not None else NOISE_LEVEL
    X_tr_aug, y_tr_aug = augment_habitable(X_train, y_train, target_count=target_count, noise_level=noise_level)
    print(f"After augmentation - Train shape: {X_tr_aug.shape} | HAB count: {(y_tr_aug==0).sum()}")

    # Dataloaders
    # Optionally use WeightedRandomSampler to oversample HAB class
    if USE_WEIGHTED_SAMPLER:
        class_counts = np.bincount(y_tr_aug, minlength=3)
        sample_weights = np.array([1.0 / max(int(class_counts[int(lbl)]), 1) for lbl in y_tr_aug], dtype=np.float64)
        if SAMPLER_SCALE != 1.0:
            sample_weights = sample_weights * (np.where(y_tr_aug == 0, float(SAMPLER_SCALE), 1.0))
        sample_weights_t = torch.tensor(sample_weights, dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights_t, num_samples=len(sample_weights_t), replacement=True)
        print(f"[kfold] Using WeightedRandomSampler (scale={SAMPLER_SCALE}) | sample_weights min={sample_weights.min():.4f} max={sample_weights.max():.4f}")
        train_loader = DataLoader(KeplerFluxDataset(X_tr_aug, y_tr_aug), batch_size=BATCH_SIZE, sampler=sampler)
    else:
        train_loader = DataLoader(KeplerFluxDataset(X_tr_aug, y_tr_aug), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(KeplerFluxDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = ExoHabitNetCNN().to(DEVICE)
    if init_weights_path and init_weights_path.exists():
        try:
            model.load_state_dict(torch.load(init_weights_path, map_location=DEVICE))
            print("Loaded pretrained weights into model.")
        except Exception as e:
            print("Warning: failed to load pretrained weights:", e)

    class_weights = compute_class_weights(y_tr_aug).to(DEVICE)
    # apply optional HAB weight multiplier
    if HAB_WEIGHT_MULTIPLIER != 1.0:
        class_weights[0] = class_weights[0] * float(HAB_WEIGHT_MULTIPLIER)

    if LOSS_TYPE == 'ce':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        alpha_tensor = None
        if FOCAL_ALPHA is not None:
            alpha_vec = torch.ones_like(class_weights)
            alpha_vec[0] = float(FOCAL_ALPHA)
            alpha_tensor = alpha_vec.to(DEVICE)
        criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights, alpha=alpha_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_f1 = -1.0
    epochs_no_improve = 0
    best_state = None

    for epoch in range(EPOCHS):
        # Train
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

        # Val
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
        try:
            val_f1 = f1_score(val_targets, val_preds, average='macro')
        except Exception:
            val_f1 = 0.0

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1:.4f}")

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, CHECKPOINT_DIR / f"fold_{fold_id+1}_best.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (no improvement)")
                break

    # Load best state for evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on val set
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(DEVICE)
            logits = model(batch_X)
            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(batch_y.numpy())

    macro_f1 = float(f1_score(val_targets, val_preds, average='macro'))
    recall_per_class = recall_score(val_targets, val_preds, average=None, labels=[0,1,2])
    hab_recall = float(recall_per_class[0])
    report = classification_report(val_targets, val_preds, target_names=['HABITABLE','NON_HABITABLE','FALSE_POSITIVE'], output_dict=True)

    print(f"Fold {fold_id+1} | Val Macro-F1: {macro_f1:.4f} | HAB Recall: {hab_recall:.4f}")

    return {
        'fold': int(fold_id+1),
        'n_train': int(len(y_tr_aug)),
        'n_val': int(len(y_val)),
        'train_hab_real': int(n_hab_train),
        'train_hab_total': int((y_tr_aug==0).sum()),
        'val_hab': int((y_val==0).sum()),
        'macro_f1': macro_f1,
        'hab_recall': hab_recall,
        'classification_report': report
    }


def main():
    if not DATA_PATH.exists():
        print(f"ERROR: Processed data not found at {DATA_PATH}. Run preprocessing first.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    flux_cols = [c for c in df.columns if c.startswith('flux_')]
    X = df[flux_cols].values
    y = df['label_id'].values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        res = train_one_fold(
            fold_id,
            X_train,
            y_train,
            X_val,
            y_val,
            init_weights_path=PRETRAIN_PATH if PRETRAIN_PATH.exists() else None,
            aug_target_override=AUG_TARGET,
            aug_noise_override=AUG_NOISE_LEVEL,
        )
        fold_results.append(res)

    # Aggregate
    macro_f1s = [r['macro_f1'] for r in fold_results]
    hab_recalls = [r['hab_recall'] for r in fold_results]

    summary = {
        'n_folds': N_SPLITS,
        'macro_f1_mean': float(np.mean(macro_f1s)),
        'macro_f1_std': float(np.std(macro_f1s)),
        'hab_recall_mean': float(np.mean(hab_recalls)),
        'hab_recall_std': float(np.std(hab_recalls)),
        'folds': fold_results
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(summary, f, indent=2)

    print('\nK-Fold fine-tuning complete.')
    print(f"Results saved: {OUT_JSON}")
    print(f"Macro-F1 mean: {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f}")
    print(f"HAB recall mean: {summary['hab_recall_mean']:.4f} ± {summary['hab_recall_std']:.4f}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Stratified K-Fold fine-tune (with optional focal loss)")
    parser.add_argument('--loss', choices=['ce', 'focal'], default='ce', help='Loss function to use')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--focal-alpha', type=float, default=None, help='Focal alpha scaling for HAB class (optional)')
    parser.add_argument('--hab-weight-multiplier', type=float, default=1.0, help='Multiply HAB class weight by this factor')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Epochs per fold')
    parser.add_argument('--n-splits', type=int, default=N_SPLITS, help='Number of StratifiedKFold splits')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--use-weighted-sampler', action='store_true', help='Use WeightedRandomSampler to oversample classes in the training loader')
    parser.add_argument('--sampler-scale', type=float, default=1.0, help='Multiplier applied to HAB sample weights when using the weighted sampler')
    parser.add_argument('--aug-target', type=int, default=None, help='Override augmentation target_count per fold (int)')
    parser.add_argument('--aug-noise-level', type=float, default=None, help='Override augmentation noise level (float)')
    args = parser.parse_args()

    # Override module-level settings from CLI
    LOSS_TYPE = args.loss
    FOCAL_GAMMA = args.focal_gamma
    FOCAL_ALPHA = args.focal_alpha
    HAB_WEIGHT_MULTIPLIER = args.hab_weight_multiplier
    EPOCHS = args.epochs
    N_SPLITS = args.n_splits
    BATCH_SIZE = args.batch_size
    USE_WEIGHTED_SAMPLER = bool(args.use_weighted_sampler)
    SAMPLER_SCALE = float(args.sampler_scale)
    AUG_TARGET = args.aug_target
    AUG_NOISE_LEVEL = args.aug_noise_level

    main()
