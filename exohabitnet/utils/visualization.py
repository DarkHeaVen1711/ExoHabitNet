"""
visualization.py
================
Visualization toolkit for ExoHabitNet evaluation metrics.
Includes Confusion Matrix, ROC curves, and sample prediction plotting.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from pathlib import Path

# Setup class mappings
CLASSES = ["HABITABLE", "NON_HABITABLE", "FALSE_POSITIVE"]
sns.set_theme(style="whitegrid")

def plot_confusion_matrix(y_true, y_pred, output_path):
    """Generates a 3x3 heatmap of true vs. predicted classes."""
    cm = confusion_matrix(y_true, y_pred)
    # Normalize for percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES,
                annot_kws={"size": 14})
    
    plt.title('Normalized Confusion Matrix\n(Values show raw counts)', fontsize=14, fontweight='bold')
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_roc_curves(y_true, y_prob, output_path):
    """
    Plots One-vs-Rest ROC curves for multi-class evaluation.
    y_prob is shape (N, 3) 
    """
    # Binarize labels
    y_bin = np.zeros((len(y_true), 3))
    for i, label in enumerate(y_true):
        y_bin[i, label] = 1

    plt.figure(figsize=(8, 6))
    colors = ['#66BB6A', '#EF5350', '#FFA726']
    
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'{CLASSES[i]} (AUC = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-Class ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
def plot_sample_predictions(x_test, y_true, y_pred, output_dir, num_samples=9):
    """
    Plots a 3x3 grid of random light curves from the test set.
    Shows the true label vs predicted label.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Pick random indices
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        flux = x_test[idx] # Shape (1, 1024) or (1024,)
        if len(flux.shape) > 1:
            flux = flux.squeeze()
            
        ax.plot(flux, color='black', alpha=0.7, lw=1)
        
        true_lbl = CLASSES[y_true[idx]]
        pred_lbl = CLASSES[y_pred[idx]]
        
        # Color title based on correctness
        color = 'green' if true_lbl == pred_lbl else 'red'
        ax.set_title(f"True: {true_lbl}\nPred: {pred_lbl}", color=color, fontsize=10, fontweight='bold')
        ax.set_xticks([]) # Hide x-ticks to clean up
        
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'test_sample_grid.png', dpi=150)
    plt.close()
