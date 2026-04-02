"""
evaluate.py
===========
Phase 7 & 8 Test Evaluation Pipeline for ExoHabitNet

Executes exactly what was established in `train.py` to isolate the 15% unseen
test distribution, calculates final mathematical performance, saves a markdown
report, and calls the visualization tools to render results.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Ensure imports work from project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.cnn_model import ExoHabitNetCNN
from utils.visualization import plot_confusion_matrix, plot_roc_curves, plot_sample_predictions

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
DATA_PATH = Path("data/balanced_dataset.csv")
MODEL_PATH = Path("models/checkpoints/best_model.pth")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CLASSES = ["HABITABLE", "NON_HABITABLE", "FALSE_POSITIVE"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model():
    print(f"Using device: {device}")
    print(f"Loading data from {DATA_PATH}...")
    if not DATA_PATH.exists():
        print("ERROR: balanced_dataset.csv not found!")
        sys.exit(1)
        
    df = pd.read_csv(DATA_PATH)
    flux_cols = [c for c in df.columns if c.startswith('flux_')]
    X = df[flux_cols].values
    y = df['label_id'].values
    
    # EXACT same split as `train.py` (Isolate the 15% Test Split)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )
    
    print(f"Isolated EXACT test set: {len(X_test)} unseen samples.")
    
    # 2. Format as tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    # 3. Load Model
    model = ExoHabitNetCNN().to(device)
    if not MODEL_PATH.exists():
        print(f"ERROR: No trained model checkpoint found at {MODEL_PATH}!")
        sys.exit(1)
        
    # Since model parameters map directly, we use load_state_dict with weights_only=True
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    print(f"Loaded trained checkpoint from {MODEL_PATH}.")
    
    # 4. Run Inference
    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        
    y_true = y_test_tensor.cpu().numpy()
    
    # 5. Metrics Calculation
    acc = accuracy_score(y_true, preds)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, preds, zero_division=0)
    macro_f1 = f1.mean()
    
    print(f"\nEvaluating final Testing Logic:")
    print(f"Overall Test Accuracy: {acc:.4f}")
    print(f"Macro-F1 Score:        {macro_f1:.4f}")
    
    # Create JSON performance log
    performance_log = {
        "overall_accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "class_metrics": {
            CLASSES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i])
            } for i in range(3)
        }
    }
    
    with open(REPORTS_DIR / "model_performance.json", "w") as f:
        json.dump(performance_log, f, indent=4)
        
    # Create Markdown Classification Report
    md_report = f"# ExoHabitNet Final Model Evaluation Report\n\n"
    md_report += f"**Overall Status**: Tested on {len(X_test)} Phase-Folded Light Curves\n\n"
    md_report += f"- **Accuracy**: {acc*100:.2f}%\n"
    md_report += f"- **Macro-F1**: {macro_f1:.4f}\n\n"
    md_report += "### Class Breakdown:\n\n"
    md_report += "| Class | Precision | Recall | F1-Score | Samples |\n"
    md_report += "|-------|-----------|--------|----------|---------|\n"
    
    for i in range(3):
        md_report += f"| {CLASSES[i]} | {precision[i]:.4f} | {recall[i]:.4f} | {f1[i]:.4f} | {int(support[i])} |\n"
        
    with open(REPORTS_DIR / "classification_report.md", "w") as f:
        f.write(md_report)
        
    print(f"\nWritten {REPORTS_DIR}/classification_report.md")
        
    # 6. Generate Phase 8 Graphics
    print("\nGenerating Visualizations (Phase 8)...")
    plot_confusion_matrix(y_true, preds, REPORTS_DIR / "confusion_matrix.png")
    plot_roc_curves(y_true, probs, REPORTS_DIR / "roc_curves.png")
    plot_sample_predictions(X_test, y_true, preds, REPORTS_DIR / "sample_predictions")
    
    print("Evaluation Complete! All reports and charts saved to reports/ directory.")

if __name__ == "__main__":
    evaluate_model()
