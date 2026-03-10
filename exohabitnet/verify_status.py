"""
ExoHabitNet Status Verification Script
Shows current project status and dataset statistics
"""

import pandas as pd
from pathlib import Path
from collections import Counter

print("=" * 60)
print("ExoHabitNet — Project Status Verification")
print("=" * 60)

# Check environment
print("\n✅ PHASE 1: Environment Setup")
try:
    import lightkurve
    import astropy
    import numpy as np
    import matplotlib
    print("  ✓ All dependencies installed")
    print(f"  ✓ Python environment: Active")
except ImportError as e:
    print(f"  ✗ Missing dependency: {e}")

# Check data collection
print("\n✅ PHASE 2: Data Collection")
raw_fits_path = Path("data/raw_fits")
collection_log = Path("data/data_collection_log.csv")

if collection_log.exists():
    log_df = pd.read_csv(collection_log)
    print(f"  ✓ Collection log: {len(log_df)} records")
    
    # Count by class
    class_counts = log_df['label'].value_counts().to_dict()
    print(f"  ✓ Raw class distribution:")
    for label, count in sorted(class_counts.items()):
        print(f"    - {label}: {count} samples")
else:
    print("  ✗ Collection log not found")

# Check preprocessing
print("\n✅ PHASE 3: Data Preprocessing")
processed_path = Path("data/processed_dataset.csv")

if processed_path.exists():
    df = pd.read_csv(processed_path)
    print(f"  ✓ Processed dataset: {len(df)} samples")
    print(f"  ✓ Dataset shape: {df.shape}")
    
    # Class distribution
    label_counts = df['label'].value_counts().sort_index()
    print(f"\n  Final Class Distribution (After Augmentation):")
    
    total = len(df)
    for label_idx in sorted(label_counts.index):
        count = label_counts[label_idx]
        percentage = (count / total) * 100
        
        # Map label to name
        label_names = {0: "HABITABLE", 1: "NON_HABITABLE", 2: "FALSE_POSITIVE"}
        name = label_names.get(label_idx, f"Class {label_idx}")
        
        print(f"    {name:>15}: {count:>3} samples ({percentage:>5.1f}%)")
    
    # Calculate class weights
    counts = [label_counts.get(i, 0) for i in sorted(label_counts.index)]
    if all(c > 0 for c in counts):
        max_count = max(counts)
        weights = [max_count / c for c in counts]
        print(f"\n  Class Weights for Training:")
        for i, w in enumerate(weights):
            label_names = {0: "HABITABLE", 1: "NON_HABITABLE", 2: "FALSE_POSITIVE"}
            name = label_names.get(i, f"Class {i}")
            print(f"    {name:>15}: {w:.4f}")
    
    print(f"\n  ✅ Dataset ready for model training!")
    
else:
    print("  ✗ Processed dataset not found")
    print("  → Run: python scripts/preprocessing_pipeline.py")

# Next steps
print("\n" + "=" * 60)
print("🔄 NEXT PHASE: Model Architecture Design")
print("=" * 60)
print("\nPending Tasks:")
print("  1. Create models/cnn_model.py with 1D-CNN architecture")
print("  2. Create scripts/train.py with training loop")
print("  3. Implement class-weighted loss function")
print("  4. Set up TensorBoard logging")
print("\nSee execution_process.md for detailed steps.")
print("=" * 60)
