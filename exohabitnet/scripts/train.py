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
def train_model():
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
    train_loader = DataLoader(KeplerFluxDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(KeplerFluxDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Model, Loss, Optimizer
    model = ExoHabitNetCNN().to(device)

    class_counts = np.bincount(y_train, minlength=3)
    total = class_counts.sum()
    class_weights = [total / (3 * max(c, 1)) for c in class_counts]
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights (from train split): {[round(w, 4) for w in class_weights]}")
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    writer = SummaryWriter(LOG_DIR)
    
    # Training state
    best_val_f1 = 0.0
    epochs_no_improve = 0
    
    print("Starting training...")
    for epoch in range(EPOCHS):
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
    train_model()
