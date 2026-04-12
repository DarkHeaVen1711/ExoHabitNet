"""
train.py
========
PyTorch Training Pipeline for ExoHabitNet

Features:
- Stratified Train/Val/Test split via scikit-learn
- Class-weighted CrossEntropyLoss
- Early Stopping
- ReduceLROnPlateau
- TensorBoard logging
- Best model checkpointing
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path so we can import models
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.cnn_model import ExoHabitNetCNN
from utils.losses import FocalLoss

# Pretraining output
PRETRAIN_PATH = Path("models/checkpoints/encoder_pretrained.pth")


def pretrain_encoder(train_csv: Path = Path("data/train_dataset.csv"), epochs: int = 10):
    """
    Short supervised pretraining of the CNN encoder on `data/train_dataset.csv`.
    Saves weights to `models/checkpoints/encoder_pretrained.pth`.
    """
    from sklearn.model_selection import train_test_split

    print(f"Running encoder pretraining (epochs={epochs}) using {train_csv}...")
    if not train_csv.exists():
        print(f"ERROR: Train data not found at {train_csv}. Run preprocessing_pipeline.py first.")
        return

    df = pd.read_csv(train_csv)
    flux_cols = [c for c in df.columns if c.startswith('flux_')]
    X = df[flux_cols].values
    y = df['label_id'].values

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

    class KeplerFluxDatasetLocal(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
            self.y = torch.tensor(y, dtype=torch.long)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_loader = DataLoader(KeplerFluxDatasetLocal(X_tr, y_tr), batch_size=32, shuffle=True)
    val_loader = DataLoader(KeplerFluxDatasetLocal(X_val, y_val), batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExoHabitNetCNN().to(device)

    # class weights
    counts = np.bincount(y_tr, minlength=3)
    total = counts.sum()
    class_weights = [total / (3 * max(int(c), 1)) for c in counts]
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val_f1 = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
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
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
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

        print(f"Pretrain Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # save encoder weights
            PRETRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), PRETRAIN_PATH)
            print(f"  [*] Saved encoder checkpoint ({PRETRAIN_PATH})")

    print("Encoder pretraining complete. Best Val Macro-F1:", best_val_f1)


# ── CONFIGURATION ─────────────────────────────────────────────────────────────
TRAIN_DATA_PATH = Path("data/train_dataset.csv")
CHECKPOINT_DIR = Path("models/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("runs/exohabitnet_experiment")

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
PATIENCE = 10
RANDOM_STATE = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── DATASET CLASS ─────────────────────────────────────────────────────────────
class KeplerFluxDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1) # Shape: (N, 1, 1024)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ── TRAINING LOOP ─────────────────────────────────────────────────────────────
def train_model(args):
    print(f"Using device: {device}")
    
    # 1. Load Data
    print(f"Loading training dataset from {TRAIN_DATA_PATH}...")
    if not TRAIN_DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {TRAIN_DATA_PATH}. Run preprocessing_pipeline.py first.")
        sys.exit(1)
        
    df = pd.read_csv(TRAIN_DATA_PATH)
    
    # Extract features and labels
    flux_cols = [c for c in df.columns if c.startswith('flux_')]
    X = df[flux_cols].values
    y = df['label_id'].values
    
    # 2. Stratified Split (Train dataset -> Train/Val)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    
    print(f"Dataset split sizes - Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Dataloaders
    # allow overriding batch size via CLI
    batch_size = getattr(args, 'batch_size', BATCH_SIZE)
    # Optional: use a WeightedRandomSampler to oversample minority classes (HAB)
    use_sampler = bool(getattr(args, 'use_weighted_sampler', False))
    sampler_scale = float(getattr(args, 'sampler_scale', 1.0))
    if use_sampler:
        # per-sample weights = 1 / class_count[label]
        class_counts = np.bincount(y_train, minlength=3)
        sample_weights = np.array([1.0 / max(int(class_counts[int(lbl)]), 1) for lbl in y_train], dtype=np.float64)
        # amplify HAB samples if requested (label_id == 0)
        if sampler_scale != 1.0:
            sample_weights = sample_weights * (np.where(y_train == 0, float(sampler_scale), 1.0))
        sample_weights_t = torch.tensor(sample_weights, dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights_t, num_samples=len(sample_weights_t), replacement=True)
        print(f"Using WeightedRandomSampler (sampler_scale={sampler_scale}) | sample_weights summary: min={sample_weights.min():.4f}, max={sample_weights.max():.4f}")
        train_loader = DataLoader(KeplerFluxDataset(X_train, y_train), batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(KeplerFluxDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(KeplerFluxDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    
    # 3. Model, Loss, Optimizer
    model = ExoHabitNetCNN().to(device)
    # If encoder pretraining exists, initialize model weights from it
    if PRETRAIN_PATH.exists():
        try:
            model.load_state_dict(torch.load(PRETRAIN_PATH, map_location=device))
            print(f"Loaded pretrained encoder weights from {PRETRAIN_PATH}")
        except Exception as e:
            print(f"Warning: failed to load pretrained weights: {e}")

    class_counts = np.bincount(y_train, minlength=3)
    total = class_counts.sum()
    class_weights = [total / (3 * max(c, 1)) for c in class_counts]
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    # optionally amplify HAB class weight
    hab_mult = getattr(args, 'hab_weight_multiplier', 1.0)
    if hab_mult != 1.0:
        weights[0] = weights[0] * float(hab_mult)

    print(f"Class weights (from train split, post-multiplier): {[round(float(w), 4) for w in weights.tolist()]}")

    # Select loss: cross-entropy or focal
    loss_choice = getattr(args, 'loss', 'ce')
    if loss_choice == 'ce':
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        focal_gamma = float(getattr(args, 'focal_gamma', 2.0))
        focal_alpha = getattr(args, 'focal_alpha', None)
        alpha_tensor = None
        if focal_alpha is not None:
            alpha_tensor = torch.ones_like(weights)
            alpha_tensor[0] = float(focal_alpha)
            alpha_tensor = alpha_tensor.to(device)
        criterion = FocalLoss(gamma=focal_gamma, weight=weights, alpha=alpha_tensor)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    writer = SummaryWriter(LOG_DIR)
    
    # Training state
    best_val_f1 = 0.0
    epochs_no_improve = 0
    
    print("Starting training...")
    epochs = getattr(args, 'epochs', EPOCHS)
    for epoch in range(epochs):
        # ── TRAIN MODE ──
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
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
        train_f1 = f1_score(train_targets, train_preds, average='macro')
        
        # ── EVAL MODE ──
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        val_acc = accuracy_score(val_targets, val_preds)
        
        # Logging
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalar('F1/val_macro', val_f1, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")
        
        # Scheduler steps on Validation F1 (we want to maximize it)
        # However, ReduceLROnPlateau might output warnings if optimizer states don't change
        # PyTorch 2.0+ expects patience behavior, so we pass val_f1
        scheduler.step(val_f1)
        
        # Early Stopping & Checkpointing
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")
            print(f"  [*] New best model saved! (Val F1: {best_val_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs due to no improvement in validation F1.")
                break
                
    writer.close()
    print("Training complete!")
    print(f"Best Validation Macro-F1: {best_val_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ExoHabitNet model")
    parser.add_argument("--pretrain", action="store_true", help="Run encoder pretraining before main training")
    parser.add_argument("--pretrain-epochs", type=int, default=10, help="Number of epochs for encoder pretraining")
    parser.add_argument("--loss", choices=['ce', 'focal'], default='ce', help="Loss function to use: 'ce' or 'focal'")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma parameter")
    parser.add_argument("--focal-alpha", type=float, default=None, help="Focal loss alpha scaling for HAB class (optional)")
    parser.add_argument("--hab-weight-multiplier", type=float, default=1.0, help="Multiply HAB class weight by this factor")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--use-weighted-sampler", action="store_true", help="Use WeightedRandomSampler to oversample classes in the training loader")
    parser.add_argument("--sampler-scale", type=float, default=1.0, help="Multiplier applied to HAB sample weights when using the weighted sampler")
    args = parser.parse_args()

    if args.pretrain:
        pretrain_encoder(epochs=args.pretrain_epochs)

    train_model(args)
