"""
balance_dataset.py
==================
ExoHabitNet — Class Imbalance Solution Script
Addresses the severe class imbalance (HABITABLE ~1.5%, NON_HABITABLE ~48%, FALSE_POSITIVE ~50%)

Three-pronged approach:
    1. Aggressive augmentation with advanced techniques
    2. SMOTE-like synthetic oversampling for time-series
    3. Dynamic class weight computation

Usage:
    python balance_dataset.py --strategy aggressive
    python balance_dataset.py --strategy moderate
    python balance_dataset.py --strategy conservative

Author: ExoHabitNet Team
Date: March 2026
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
RAW_FITS_DIR = Path("data/raw_fits")
COLLECTION_LOG = Path("data/data_collection_log.csv")
PROCESSED_CSV = Path("data/processed_dataset.csv")
BALANCED_CSV = Path("data/balanced_dataset.csv")
REPORTS_DIR = Path("reports/class_balance")

SEQUENCE_LENGTH = 1024
RANDOM_STATE = 42

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Augmentation strategies
STRATEGIES = {
    "conservative": {
        "habitable_target": 150,
        "non_habitable_target": 229,  # Keep original
        "false_positive_target": 217,  # Keep original
        "noise_level": 0.008,
        "time_shift_range": 30,
        "use_smote": False
    },
    "moderate": {
        "habitable_target": 200,
        "non_habitable_target": 220,  # Slight undersample
        "false_positive_target": 200,  # Slight undersample
        "noise_level": 0.012,
        "time_shift_range": 50,
        "use_smote": True
    },
    "aggressive": {
        "habitable_target": 250,
        "non_habitable_target": 200,  # Undersample majority
        "false_positive_target": 200,  # Undersample majority
        "noise_level": 0.015,
        "time_shift_range": 80,
        "use_smote": True
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED AUGMENTATION TECHNIQUES
# ─────────────────────────────────────────────────────────────────────────────

def add_gaussian_noise(flux: np.ndarray, noise_level: float, rng) -> np.ndarray:
    """Add Gaussian noise to flux sequence."""
    noise = rng.normal(0, noise_level, size=len(flux))
    return flux + noise


def time_shift(flux: np.ndarray, shift_range: int, rng) -> np.ndarray:
    """
    Circular time shift to simulate different transit phases.
    Physics: Transit can occur at any point in the observation window.
    """
    shift = rng.integers(-shift_range, shift_range + 1)
    return np.roll(flux, shift)


def scale_amplitude(flux: np.ndarray, scale_range: tuple, rng) -> np.ndarray:
    """
    Scale transit depth slightly to simulate stellar variability.
    Physics: Same planet around slightly different stars or different viewing angles.
    Scale range: e.g., (0.95, 1.05) = ±5% variation
    """
    scale = rng.uniform(*scale_range)
    mean = np.nanmean(flux)
    return mean + (flux - mean) * scale


def add_stellar_flare(flux: np.ndarray, flare_prob: float, rng) -> np.ndarray:
    """
    Add occasional stellar flare (sharp positive spike).
    Physics: Real Kepler data contains flares; makes augmented data more realistic.
    """
    if rng.random() < flare_prob:
        # Random flare position and amplitude
        pos = rng.integers(100, len(flux) - 100)
        amplitude = rng.uniform(0.1, 0.3)
        width = rng.integers(5, 15)
        
        # Gaussian flare profile
        x = np.arange(len(flux))
        flare = amplitude * np.exp(-((x - pos) ** 2) / (2 * width ** 2))
        return flux + flare
    return flux


def synthetic_interpolation(flux1: np.ndarray, flux2: np.ndarray, alpha: float) -> np.ndarray:
    """
    SMOTE-like interpolation between two real samples.
    Physics: Linear combination of two similar transits from different planets.
    """
    return alpha * flux1 + (1 - alpha) * flux2


def augment_sample_advanced(
    flux: np.ndarray,
    noise_level: float,
    time_shift_range: int,
    rng,
    apply_scale: bool = True,
    apply_flare: bool = False
) -> np.ndarray:
    """
    Apply multiple augmentation techniques in sequence.
    """
    aug_flux = flux.copy()
    
    # Always apply noise and time shift
    aug_flux = add_gaussian_noise(aug_flux, noise_level, rng)
    aug_flux = time_shift(aug_flux, time_shift_range, rng)
    
    # Optional: amplitude scaling (use sparingly to preserve transit physics)
    if apply_scale and rng.random() < 0.5:
        aug_flux = scale_amplitude(aug_flux, (0.97, 1.03), rng)
    
    # Optional: add stellar flare (rare, only 10% of samples)
    if apply_flare and rng.random() < 0.1:
        aug_flux = add_stellar_flare(aug_flux, flare_prob=0.3, rng=rng)
    
    return aug_flux


# ─────────────────────────────────────────────────────────────────────────────
# SMOTE FOR TIME-SERIES
# ─────────────────────────────────────────────────────────────────────────────

def smote_time_series(samples: List[np.ndarray], target_count: int, rng) -> List[np.ndarray]:
    """
    SMOTE (Synthetic Minority Over-sampling Technique) adapted for time-series.
    
    For each real sample, create synthetic samples by interpolating with
    k-nearest neighbors in the feature space.
    
    Simplified version: randomly interpolate between pairs of real samples.
    """
    n_real = len(samples)
    n_needed = target_count - n_real
    
    if n_needed <= 0 or n_real == 0:
        return samples
    
    synthetic = []
    for _ in range(n_needed):
        # Pick two random samples
        idx1, idx2 = rng.choice(n_real, size=2, replace=True)
        flux1 = samples[idx1]
        flux2 = samples[idx2]
        
        # Random interpolation weight
        alpha = rng.uniform(0.3, 0.7)
        synthetic_flux = synthetic_interpolation(flux1, flux2, alpha)
        
        # Add small noise to avoid exact duplicates
        synthetic_flux = add_gaussian_noise(synthetic_flux, 0.005, rng)
        
        synthetic.append(synthetic_flux)
    
    return samples + synthetic


# ─────────────────────────────────────────────────────────────────────────────
# UNDERSAMPLING MAJORITY CLASSES
# ─────────────────────────────────────────────────────────────────────────────

def undersample_class(samples: List[Dict], target_count: int, rng) -> List[Dict]:
    """
    Randomly undersample majority class to reduce imbalance.
    """
    if len(samples) <= target_count:
        return samples
    
    indices = rng.choice(len(samples), size=target_count, replace=False)
    return [samples[i] for i in sorted(indices)]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BALANCING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def load_processed_data() -> pd.DataFrame:
    """Load the preprocessed dataset."""
    if not PROCESSED_CSV.exists():
        print(f"ERROR: Processed dataset not found at {PROCESSED_CSV}")
        print("Run preprocessing_pipeline.py first.")
        sys.exit(1)
    
    return pd.read_csv(PROCESSED_CSV)


def balance_dataset(strategy_name: str) -> pd.DataFrame:
    """
    Main balancing function.
    """
    print("=" * 70)
    print(f"  ExoHabitNet — Class Balancing ({strategy_name.upper()} strategy)")
    print("=" * 70)
    
    strategy = STRATEGIES[strategy_name]
    rng = np.random.default_rng(RANDOM_STATE)
    
    # Load data
    df = load_processed_data()
    
    # Extract flux sequences
    flux_cols = [c for c in df.columns if c.startswith("flux_")]
    
    # Split by class
    habitable_df = df[df["label"] == "HABITABLE"].copy()
    non_habitable_df = df[df["label"] == "NON_HABITABLE"].copy()
    false_positive_df = df[df["label"] == "FALSE_POSITIVE"].copy()
    
    print(f"\n  Original class distribution:")
    print(f"    HABITABLE:       {len(habitable_df):4d} samples")
    print(f"    NON_HABITABLE:   {len(non_habitable_df):4d} samples")
    print(f"    FALSE_POSITIVE:  {len(false_positive_df):4d} samples")
    print(f"    TOTAL:           {len(df):4d} samples")
    
    # ── HABITABLE: Aggressive oversampling ────────────────────────────────
    print(f"\n  Augmenting HABITABLE class...")
    hab_samples = habitable_df[flux_cols].values
    n_hab_real = len(hab_samples)
    n_hab_needed = strategy["habitable_target"] - n_hab_real
    
    print(f"    Real:   {n_hab_real}")
    print(f"    Target: {strategy['habitable_target']}")
    print(f"    Needed: {n_hab_needed}")
    
    hab_augmented = []
    
    if strategy["use_smote"]:
        print(f"    Using SMOTE interpolation...")
        hab_flux_list = [row for row in hab_samples]
        hab_smote = smote_time_series(hab_flux_list, strategy["habitable_target"] // 2, rng)
        
        # Additional noise/shift augmentation
        for i in range(n_hab_needed - len(hab_smote) + n_hab_real):
            source_flux = hab_samples[i % n_hab_real]
            aug_flux = augment_sample_advanced(
                source_flux,
                strategy["noise_level"],
                strategy["time_shift_range"],
                rng,
                apply_scale=True,
                apply_flare=False
            )
            hab_augmented.append(aug_flux)
        
        # Combine SMOTE + augmentation
        hab_final = hab_smote + hab_augmented
    else:
        print(f"    Using Gaussian noise + time shift...")
        for i in range(n_hab_needed):
            source_flux = hab_samples[i % n_hab_real]
            aug_flux = augment_sample_advanced(
                source_flux,
                strategy["noise_level"],
                strategy["time_shift_range"],
                rng,
                apply_scale=False,
                apply_flare=False
            )
            hab_augmented.append(aug_flux)
        
        hab_final = list(hab_samples) + hab_augmented
    
    # Create augmented DataFrame
    hab_aug_df = pd.DataFrame([
        {
            "label": "HABITABLE",
            "label_id": 0,
            "is_augmented": i >= n_hab_real,
            "fits_path": f"augmented_HABITABLE_{i}" if i >= n_hab_real else habitable_df.iloc[i]["fits_path"],
            **{f"flux_{j}": hab_final[i][j] for j in range(SEQUENCE_LENGTH)}
        }
        for i in range(len(hab_final))
    ])
    
    # ── NON_HABITABLE: Undersample if needed ──────────────────────────────
    print(f"\n  Processing NON_HABITABLE class...")
    if len(non_habitable_df) > strategy["non_habitable_target"]:
        print(f"    Undersampling from {len(non_habitable_df)} to {strategy['non_habitable_target']}")
        non_hab_indices = rng.choice(
            len(non_habitable_df),
            size=strategy["non_habitable_target"],
            replace=False
        )
        non_hab_final_df = non_habitable_df.iloc[non_hab_indices].copy()
    else:
        print(f"    Keeping all {len(non_habitable_df)} samples")
        non_hab_final_df = non_habitable_df.copy()
    
    non_hab_final_df["is_augmented"] = False
    
    # ── FALSE_POSITIVE: Undersample if needed ─────────────────────────────
    print(f"\n  Processing FALSE_POSITIVE class...")
    if len(false_positive_df) > strategy["false_positive_target"]:
        print(f"    Undersampling from {len(false_positive_df)} to {strategy['false_positive_target']}")
        fp_indices = rng.choice(
            len(false_positive_df),
            size=strategy["false_positive_target"],
            replace=False
        )
        fp_final_df = false_positive_df.iloc[fp_indices].copy()
    else:
        print(f"    Keeping all {len(false_positive_df)} samples")
        fp_final_df = false_positive_df.copy()
    
    fp_final_df["is_augmented"] = False
    
    # ── Combine and shuffle ───────────────────────────────────────────────
    balanced_df = pd.concat([hab_aug_df, non_hab_final_df, fp_final_df], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    print(f"\n  Balanced class distribution:")
    print(f"    HABITABLE:       {len(hab_aug_df):4d} samples ({len(hab_aug_df)/len(balanced_df)*100:.1f}%)")
    print(f"    NON_HABITABLE:   {len(non_hab_final_df):4d} samples ({len(non_hab_final_df)/len(balanced_df)*100:.1f}%)")
    print(f"    FALSE_POSITIVE:  {len(fp_final_df):4d} samples ({len(fp_final_df)/len(balanced_df)*100:.1f}%)")
    print(f"    TOTAL:           {len(balanced_df):4d} samples")
    
    # Compute class weights for PyTorch
    print(f"\n  Class weights for training:")
    counts = balanced_df["label"].value_counts()
    total = len(balanced_df)
    weights = [round(total / (3 * counts["HABITABLE"]), 4),
               round(total / (3 * counts["NON_HABITABLE"]), 4),
               round(total / (3 * counts["FALSE_POSITIVE"]), 4)]
    print(f"    class_weights = torch.tensor({weights})  # [HAB, NON_HAB, FP]")
    
    return balanced_df


def plot_class_distribution(df_original: pd.DataFrame, df_balanced: pd.DataFrame, strategy: str):
    """
    Generate before/after comparison charts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Before
    counts_orig = df_original["label"].value_counts()
    colors = {"HABITABLE": "#66BB6A", "NON_HABITABLE": "#EF5350", "FALSE_POSITIVE": "#FFA726"}
    
    axes[0].bar(counts_orig.index, counts_orig.values,
                color=[colors[c] for c in counts_orig.index],
                edgecolor="white", linewidth=2)
    axes[0].set_title("Original Dataset (Imbalanced)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Number of Samples", fontsize=11)
    axes[0].set_xlabel("Class", fontsize=11)
    
    for i, (cls, val) in enumerate(counts_orig.items()):
        pct = val / len(df_original) * 100
        axes[0].text(i, val + 5, f"{val}\n({pct:.1f}%)", ha="center", fontweight="bold")
    
    # After
    counts_bal = df_balanced["label"].value_counts()
    axes[1].bar(counts_bal.index, counts_bal.values,
                color=[colors[c] for c in counts_bal.index],
                edgecolor="white", linewidth=2)
    axes[1].set_title(f"Balanced Dataset ({strategy.capitalize()} Strategy)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Number of Samples", fontsize=11)
    axes[1].set_xlabel("Class", fontsize=11)
    
    for i, (cls, val) in enumerate(counts_bal.items()):
        pct = val / len(df_balanced) * 100
        axes[1].text(i, val + 5, f"{val}\n({pct:.1f}%)", ha="center", fontweight="bold")
    
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"class_balance_{strategy}.png", dpi=150)
    print(f"\n  Chart saved: {REPORTS_DIR / f'class_balance_{strategy}.png'}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Balance ExoHabitNet dataset")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["conservative", "moderate", "aggressive"],
        default="moderate",
        help="Balancing strategy (default: moderate)"
    )
    
    args = parser.parse_args()
    
    # Load original data
    df_original = load_processed_data()
    
    # Balance dataset
    df_balanced = balance_dataset(args.strategy)
    
    # Save balanced dataset
    df_balanced.to_csv(BALANCED_CSV, index=False)
    print(f"\n  Balanced dataset saved: {BALANCED_CSV}")
    print(f"    Shape: {df_balanced.shape[0]} samples x {df_balanced.shape[1]} columns")
    
    # Generate visualization
    plot_class_distribution(df_original, df_balanced, args.strategy)
    
    print("\n" + "=" * 70)
    print("  Class balancing complete!")
    print(f"  Use {BALANCED_CSV} for model training.")
    print("=" * 70)


if __name__ == "__main__":
    main()
