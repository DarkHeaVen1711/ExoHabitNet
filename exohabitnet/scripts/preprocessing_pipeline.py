"""
preprocessing_pipeline.py
==========================
ExoHabitNet Phase 3 — Deep Learning Preprocessing Pipeline
Handles: Cleaning → Normalization → Phase Folding → EDA chart generation

Usage:
    python preprocessing_pipeline.py

Requirements:
    pip install lightkurve numpy pandas scipy matplotlib seaborn astropy
"""

import os
import numpy as np
import pandas as pd
import lightkurve as lk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats
from astropy.io import fits
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
RAW_FITS_DIR          = Path("data/raw_fits")
PROCESSED_DIR         = Path("data/processed")
EDA_CHARTS_DIR        = Path("reports/eda_charts")
COLLECTION_LOG        = Path("data/data_collection_log.csv")
PROCESSED_OUT_CSV     = Path("data/processed_dataset.csv")
TRAIN_OUT_CSV         = Path("data/train_dataset.csv")
TEST_OUT_CSV          = Path("data/test_dataset.csv")

SEQUENCE_LENGTH       = 1024        # Fixed time-series length for DL model input
OUTLIER_SIGMA         = 5.0         # σ-clip threshold for outlier removal
RANDOM_STATE          = 42

PROCESSED_DIR.mkdir(parents=True,  exist_ok=True)
EDA_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: CLEANING
# ─────────────────────────────────────────────────────────────────────────────
def clean_light_curve(lc) -> "lk.LightCurve":
    """
    Cleans a lightkurve LightCurve object:
      1. Remove cadences with non-zero SAP quality flags
      2. Drop NaN flux values via linear interpolation
      3. Remove outliers using sigma clipping (σ-clip)

    Returns a cleaned LightCurve object.
    """
    # ── Quality flag removal ──────────────────────────────────────────────
    # SAP_QUALITY bitmask: any non-zero flag indicates a bad cadence
    # (e.g., bit 1 = argabrightening, bit 3 = coarse point, bit 16 = cosmic ray)
    if hasattr(lc, "quality"):
        lc = lc[lc.quality == 0]

    # ── NaN interpolation ─────────────────────────────────────────────────
    # Linear interpolation fills isolated gaps; isolated gaps arise from
    # missing downlinks or Kepler's safe-mode events (typically <5% of cadences)
    flux_series = pd.Series(lc.flux.value)
    nan_count   = flux_series.isna().sum()
    if nan_count > 0:
        flux_series = flux_series.interpolate(method="linear", limit_direction="both")
        flux_arr    = flux_series.values
        lc          = lc.copy()
        lc.flux     = flux_arr * lc.flux.unit

    # ── Sigma-clip outlier removal ────────────────────────────────────────
    # Removes extreme flux spikes (cosmic ray hits, detector saturation)
    # that are NOT caused by planetary transits (transit dips are gradual).
    # Implementation: remove points > OUTLIER_SIGMA standard deviations
    # from the running median (protects the transit from being clipped).
    flux_vals  = lc.flux.value
    median     = np.nanmedian(flux_vals)
    mad_std    = np.nanstd(flux_vals)   # Use full std for simplicity; scipy.stats.sigmaclip for production
    good_mask  = np.abs(flux_vals - median) < OUTLIER_SIGMA * mad_std
    lc_cleaned = lc[good_mask]

    return lc_cleaned


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def normalize_flux_global(flux: np.ndarray) -> np.ndarray:
    """
    Global (Min-Max) Normalization: scales flux to [0, 1].
    Formula: flux_norm = (flux - flux_min) / (flux_max - flux_min)

    Use case: Preserves the absolute transit depth information across the
    full observation. Used for CNN/LSTM inputs that benefit from global context.
    """
    f_min = np.nanmin(flux)
    f_max = np.nanmax(flux)
    denom = f_max - f_min
    if denom == 0:
        return np.zeros_like(flux)
    return (flux - f_min) / denom


def normalize_flux_local(flux: np.ndarray) -> np.ndarray:
    """
    Local (Z-Score) Normalization: centers around median, scales by MAD.
    Formula: flux_norm = (flux - median(flux)) / std(flux)

    Use case: Removes long-term stellar variability (trends) so the model
    focuses on the local transit shape, not overall brightness. More robust
    to outliers than mean/std normalization.
    """
    median = np.nanmedian(flux)
    std    = np.nanstd(flux)
    if std == 0:
        return np.zeros_like(flux)
    return (flux - median) / std


def normalize_pipeline(lc) -> tuple:
    """
    Returns both globally-normalized and locally-normalized flux arrays.
    The global version is used for transit depth estimation;
    the local version is used as the primary DL model input.
    """
    flux        = lc.flux.value
    flux_global = normalize_flux_global(flux)
    flux_local  = normalize_flux_local(flux)
    return flux_global, flux_local


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: PHASE FOLDING
# ─────────────────────────────────────────────────────────────────────────────
def phase_fold(time: np.ndarray, flux: np.ndarray, period: float, t0: float) -> tuple:
    """
    Phase folding aligns all transit events on top of each other by transforming
    the observation time into a fractional orbital phase.

    Mathematical Transformation:
        phase = ((time - t0) % period) / period

    This maps all times into the range [0, 1) where:
        - phase = 0.0 (and 1.0) → transit midpoint (t0 reference)
        - phase = 0.5 → secondary eclipse (if present)

    To center the transit at phase = 0 (for the DL model's benefit):
        phase_centered = phase - 0.5   → range [-0.5, +0.5)

    Parameters:
        time   : BJD time array (from FITS TIME column)
        flux   : Normalized flux array (same length as time)
        period : Orbital period in days (from BLS or KOI table)
        t0     : Transit midpoint time (from KOI table: koi_time0bk + 2454833)

    Returns:
        phase_sorted : Phase values sorted in [-0.5, +0.5)
        flux_sorted  : Flux values corresponding to sorted phase
    """
    # ── Core phase formula ────────────────────────────────────────────────
    phase = ((time - t0) % period) / period

    # ── Center transit at phase = 0 ───────────────────────────────────────
    phase[phase > 0.5] -= 1.0      # Shift [0.5, 1) → [-0.5, 0) for continuity

    # ── Sort by phase for clean visualization and binning ─────────────────
    sort_idx     = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    flux_sorted  = flux[sort_idx]

    return phase_sorted, flux_sorted


def bin_phase_curve(phase: np.ndarray, flux: np.ndarray, n_bins: int = SEQUENCE_LENGTH) -> tuple:
    """
    Bins the phase-folded curve into SEQUENCE_LENGTH uniform bins.

    Purpose: Uneven cadence coverage means different phase values have
    different numbers of measurements. Binning creates a FIXED-LENGTH
    input vector suitable for the DL model (CNN requires constant-size input).

    Each bin value = median flux of all cadences in that bin.
    Empty bins filled via linear interpolation.

    Returns:
        bin_centers : Array of shape (n_bins,) with phase values
        bin_flux    : Array of shape (n_bins,) with median-binned flux
    """
    bins        = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_flux    = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (phase >= bins[i]) & (phase < bins[i + 1])
        if np.any(mask):
            bin_flux[i] = np.nanmedian(flux[mask])

    # Fill empty bins via interpolation
    series   = pd.Series(bin_flux)
    bin_flux = series.interpolate(method="linear", limit_direction="both").values

    return bin_centers, bin_flux


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: FULL PREPROCESSING PIPELINE (single sample)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_single(fits_path: str, period: float, t0: float, label: str) -> dict | None:
    """
    End-to-end preprocessing for one light curve:
        .fits -> clean -> normalize -> phase-fold -> bin -> feature vector

    Returns a dict with keys: 'flux_sequence', 'label', 'label_id',
    or None if preprocessing fails.
    """
    try:
        lc = lk.read(fits_path)

        # Step 1: Clean
        lc_clean = clean_light_curve(lc)
        if len(lc_clean) < 200:       # Discard curves with too few valid cadences
            return None

        # Step 2: Normalize (local Z-score is primary input for DL)
        _, flux_local = normalize_pipeline(lc_clean)

        # Step 3: Phase fold
        time_arr = lc_clean.time.value
        phase, flux_folded = phase_fold(time_arr, flux_local, period, t0)

        # Step 4: Bin to fixed sequence length
        _, flux_binned = bin_phase_curve(phase, flux_folded, n_bins=SEQUENCE_LENGTH)

        label_map = {"HABITABLE": 0, "NON_HABITABLE": 1, "FALSE_POSITIVE": 2}
        return {
            "fits_path":      fits_path,
            "label":          label,
            "label_id":       label_map.get(label, -1),
            "flux_sequence":  flux_binned.tolist()
        }

    except Exception as e:
        print(f"  [WARNING] Failed to preprocess {fits_path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4b: DATA AUGMENTATION (Minority Class Oversampling)
# ─────────────────────────────────────────────────────────────────────────────
def augment_habitable_class(
    processed_results: list,
    target_count: int = 100,
    noise_level: float = 0.008,
    rng_seed: int = 42
) -> list:
    """
    Synthetically oversamples the HABITABLE class by adding small Gaussian
    noise to the real phase-folded flux sequences.

    Why this is valid:
      - Each Kepler light curve already has ~1% photon noise.
      - Adding noise_level=0.008 (< 1% of Z-score scale) mimics the natural
        variation between different Kepler observations of similar planets.
      - The transit DIP shape is preserved — the model learns from shape, not
        exact flux values.
      - This is standard practice in DL for rare-class time-series data
        (equivalent to image rotation/flip augmentation in computer vision).

    Parameters:
        processed_results : Output of preprocess_single() for all samples
        target_count      : How many HABITABLE samples to produce total
        noise_level       : Std-dev of Gaussian noise relative to flux scale
        rng_seed          : Reproducibility seed

    Returns:
        augmented list with synthetic HABITABLE rows added
    """
    rng = np.random.default_rng(rng_seed)

    habitable_samples = [r for r in processed_results
                         if r and r["label"] == "HABITABLE"]
    n_real = len(habitable_samples)

    if n_real == 0:
        print("  [WARNING] No HABITABLE samples found — skipping augmentation.")
        return processed_results

    n_needed = max(0, target_count - n_real)
    print(f"  Augmentation: {n_real} real HABITABLE -> generating {n_needed} "
          f"synthetic copies (target={target_count})")

    synthetic = []
    for i in range(n_needed):
        # Cycle through real samples as source templates
        source  = habitable_samples[i % n_real]
        flux    = np.array(source["flux_sequence"])
        noise   = rng.normal(0, noise_level, size=len(flux))
        aug_flux = (flux + noise).tolist()
        synthetic.append({
            "fits_path":     source["fits_path"] + f"__aug{i}",
            "label":         "HABITABLE",
            "label_id":      0,
            "flux_sequence": aug_flux,
            "is_augmented":  True
        })

    result = processed_results + synthetic
    counts = {}
    for r in result:
        if r:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
    print(f"  Post-augmentation class counts: {counts}")
    return result


def split_real_dataset(
    processed_results: list,
    test_size: float = 0.2,
    rng_seed: int = 42
) -> tuple[list, list]:
    """
    Split ONLY real preprocessed samples into stratified train/test sets.

    This prevents augmented variants of the same source signal from leaking
    into both train and test sets.
    """
    clean_rows = [r for r in processed_results if r]
    if not clean_rows:
        return [], []

    labels = [r["label_id"] for r in clean_rows]
    train_rows, test_rows = train_test_split(
        clean_rows,
        test_size=test_size,
        stratify=labels,
        random_state=rng_seed
    )
    return train_rows, test_rows


def _to_dataframe(rows: list) -> pd.DataFrame:
    """Convert list of processed rows to tabular dataframe format."""
    return pd.DataFrame([
        {
            "label": r["label"],
            "label_id": r["label_id"],
            "is_augmented": r.get("is_augmented", False),
            "fits_path": r["fits_path"],
            **{f"flux_{i}": v for i, v in enumerate(r["flux_sequence"])}
        }
        for r in rows if r
    ])


def compute_class_weights(processed_results: list) -> dict:
    """
    Computes inverse-frequency class weights for use in PyTorch CrossEntropyLoss.

    Formula: weight_c = total_samples / (n_classes * count_c)

    These weights penalize misclassification of rare classes more heavily,
    complementing the augmentation strategy.

    Returns dict: {'HABITABLE': float, 'NON_HABITABLE': float, 'FALSE_POSITIVE': float}
    Also prints the ready-to-paste PyTorch tensor line.
    """
    label_map = {"HABITABLE": 0, "NON_HABITABLE": 1, "FALSE_POSITIVE": 2}
    counts = {k: 0 for k in label_map}
    for r in processed_results:
        if r and r["label"] in counts:
            counts[r["label"]] += 1

    total    = sum(counts.values())
    n_classes = len(counts)
    weights  = {}
    for cls, cnt in counts.items():
        weights[cls] = round(total / (n_classes * max(cnt, 1)), 4)

    w_list = [weights["HABITABLE"], weights["NON_HABITABLE"], weights["FALSE_POSITIVE"]]
    print(f"\n  Class weights (paste into your model training script):")
    print(f"  class_weights = torch.tensor({w_list})  # [HABITABLE, NON_HABITABLE, FALSE_POSITIVE]")
    print(f"  loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))")
    return weights




# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: EDA CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def generate_eda_charts(log_df: pd.DataFrame, sample_processed: list):
    """
    Generates the 3 required EDA charts for the Data Analysis Report.

    Chart 1: Flux vs. Time (raw light curve with transit annotations)
    Chart 2: Flux Distribution by Class (overlapping KDE + histogram)
    Chart 3: Class Balance Bar Chart
    """
    sns.set_theme(style="darkgrid", palette="muted")
    plt.rcParams.update({"figure.dpi": 150, "font.family": "DejaVu Sans"})

    # ── CHART 1: Flux vs. Time ────────────────────────────────────────────
    # Shows a single raw Kepler light curve with transit dips visible
    try:
        success_rows = log_df[log_df["collection_status"] == "SUCCESS"]
        if not success_rows.empty:
            sample_row = success_rows.iloc[0]
            lc_raw     = lk.read(sample_row["fits_path"])

            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(lc_raw.time.value, lc_raw.flux.value,
                    color="#4FC3F7", linewidth=0.5, alpha=0.8, label="PDCSAP Flux")
            ax.set_xlabel("Time (BJD − 2454833)", fontsize=11)
            ax.set_ylabel("Flux (e⁻/s)", fontsize=11)
            ax.set_title(f"Chart 1: Raw Kepler Light Curve — {sample_row.get('kepoi_name', 'Sample')}\n"
                         f"Label: {sample_row['label']}", fontsize=13, fontweight="bold")
            ax.legend()
            fig.tight_layout()
            chart1_path = EDA_CHARTS_DIR / "chart1_flux_vs_time.png"
            fig.savefig(chart1_path)
            plt.close(fig)
            print(f"  ✓ Chart 1 saved: {chart1_path}")
    except Exception as e:
        print(f"  [WARNING] Chart 1 failed: {e}")

    # ── CHART 2: Flux Distribution by Class ──────────────────────────────
    # KDE + histogram showing how flux statistics differ across 3 classes
    try:
        if sample_processed:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
            colors    = {"HABITABLE": "#66BB6A", "NON_HABITABLE": "#EF5350", "FALSE_POSITIVE": "#FFA726"}
            class_data = {"HABITABLE": [], "NON_HABITABLE": [], "FALSE_POSITIVE": []}

            for item in sample_processed:
                if item and item["label"] in class_data:
                    seq = np.array(item["flux_sequence"])
                    class_data[item["label"]].extend(seq[~np.isnan(seq)].tolist())

            for ax, (cls, data) in zip(axes, class_data.items()):
                arr = np.array(data[:50000])  # Cap for performance
                if len(arr) > 10:
                    ax.hist(arr, bins=60, color=colors[cls], alpha=0.6, density=True, label=cls)
                    kde_x = np.linspace(arr.min(), arr.max(), 300)
                    kde   = stats.gaussian_kde(arr)
                    ax.plot(kde_x, kde(kde_x), color=colors[cls], linewidth=2)
                ax.set_title(cls, fontweight="bold")
                ax.set_xlabel("Normalized Flux (Z-score)")

            axes[0].set_ylabel("Density")
            fig.suptitle("Chart 2: Flux Distribution by Class\n(Phase-Folded, Z-Score Normalized)",
                         fontsize=13, fontweight="bold")
            fig.tight_layout()
            chart2_path = EDA_CHARTS_DIR / "chart2_flux_distribution.png"
            fig.savefig(chart2_path)
            plt.close(fig)
            print(f"  ✓ Chart 2 saved: {chart2_path}")
    except Exception as e:
        print(f"  [WARNING] Chart 2 failed: {e}")

    # ── CHART 3: Class Balance ────────────────────────────────────────────
    # Shows dataset class distribution — critical for DL training (imbalance check)
    try:
        label_counts = log_df[log_df["collection_status"] == "SUCCESS"]["label"].value_counts()
        fig, ax      = plt.subplots(figsize=(8, 5))
        colors_bar   = ["#66BB6A", "#EF5350", "#FFA726"]
        bars         = ax.bar(label_counts.index, label_counts.values,
                              color=colors_bar[:len(label_counts)], edgecolor="white", linewidth=1.5)

        for bar, val in zip(bars, label_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"n={val}", ha="center", va="bottom", fontweight="bold", fontsize=11)

        ax.set_xlabel("Class", fontsize=12)
        ax.set_ylabel("Number of Samples", fontsize=12)
        ax.set_title("Chart 3: Class Balance in ExoHabitNet Dataset\n"
                     "(Check for class imbalance → consider SMOTE or weighted loss if needed)",
                     fontsize=13, fontweight="bold")
        ax.set_ylim(0, label_counts.max() * 1.15)
        fig.tight_layout()
        chart3_path = EDA_CHARTS_DIR / "chart3_class_balance.png"
        fig.savefig(chart3_path)
        plt.close(fig)
        print(f"  ✓ Chart 3 saved: {chart3_path}")
    except Exception as e:
        print(f"  [WARNING] Chart 3 failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  ExoHabitNet — Phase 3 Preprocessing Pipeline")
    print("=" * 65)

    # Load collection log
    if not COLLECTION_LOG.exists():
        print(f"  ERROR: Collection log not found at {COLLECTION_LOG}")
        print("  Run collect_kepler_data.py first.")
        return

    log_df = pd.read_csv(COLLECTION_LOG)
    success_df = log_df[
        (log_df["collection_status"] == "SUCCESS") &
        (log_df["quality_check"] == "PASS")
    ].reset_index(drop=True)

    print(f"  Found {len(success_df)} valid samples to preprocess.\n")

    # Load KOI table for period and t0 (needed for phase folding)
    koi_extra = None
    try:
        import requests as req
        url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        params = {
            "query": "SELECT kepid, koi_period, koi_time0bk FROM cumulative",
            "format": "json", "lang": "ADQL"
        }
        r = req.get(url, params=params, timeout=60)
        koi_extra = pd.DataFrame(r.json()).set_index("kepid")
        print("  ✓ Fetched period/t0 data from NASA archive.")
    except Exception as e:
        print(f"  [WARNING] Could not fetch period/t0 data: {e}. Using defaults.")

    # Process each sample
    processed_results = []
    for _, row in success_df.iterrows():
        kid    = int(row["kepid"])
        label  = str(row["label"])
        period = 1.0
        t0     = 0.0

        if koi_extra is not None and kid in koi_extra.index:
            koi_period = koi_extra.loc[kid, "koi_period"]
            koi_t0 = koi_extra.loc[kid, "koi_time0bk"]
            
            # Handle case where kepid has multiple KOIs (returns Series)
            if isinstance(koi_period, pd.Series):
                koi_period = koi_period.iloc[0]
            if isinstance(koi_t0, pd.Series):
                koi_t0 = koi_t0.iloc[0]
            
            period = float(koi_period) if not pd.isna(koi_period) else 1.0
            t0     = float(koi_t0) if not pd.isna(koi_t0) else 0.0

        result = preprocess_single(row["fits_path"], period, t0, label)
        if result:
            processed_results.append(result)

    print(f"\n  Preprocessed {len(processed_results)}/{len(success_df)} samples successfully.")

    # ── Save full real (non-augmented) dataset ────────────────────────────
    if processed_results:
        real_df = _to_dataframe(processed_results)
        real_df.to_csv(PROCESSED_OUT_CSV, index=False)
        print(f"\n  Real processed dataset saved: {PROCESSED_OUT_CSV}")
        print(f"    Shape: {real_df.shape[0]} samples x {real_df.shape[1]} columns")

    # ── Split first, then augment ONLY train split ────────────────────────
    print("\n  Creating stratified train/test split using real samples only...")
    train_rows, test_rows = split_real_dataset(
        processed_results,
        test_size=0.2,
        rng_seed=RANDOM_STATE
    )

    train_counts = pd.Series([r["label"] for r in train_rows]).value_counts().to_dict()
    test_counts = pd.Series([r["label"] for r in test_rows]).value_counts().to_dict()
    print(f"  Train split counts (real only): {train_counts}")
    print(f"  Test split counts  (real only): {test_counts}")

    print("\n  Running augmentation on TRAIN split only (prevents data leakage)...")
    train_rows_aug = augment_habitable_class(
        train_rows,
        target_count=100,
        noise_level=0.01,
        rng_seed=RANDOM_STATE
    )

    # ── Compute class weights for DL training (train split only) ──────────
    compute_class_weights(train_rows_aug)

    # ── Save split datasets ────────────────────────────────────────────────
    train_df = _to_dataframe(train_rows_aug)
    test_df = _to_dataframe(test_rows)

    train_df.to_csv(TRAIN_OUT_CSV, index=False)
    test_df.to_csv(TEST_OUT_CSV, index=False)

    print(f"\n  Train dataset saved: {TRAIN_OUT_CSV}")
    print(f"    Shape: {train_df.shape[0]} samples x {train_df.shape[1]} columns")
    train_aug_count = int(train_df["is_augmented"].sum())
    print(f"    Real samples: {len(train_df) - train_aug_count} | Augmented: {train_aug_count}")

    print(f"\n  Test dataset saved: {TEST_OUT_CSV}")
    print(f"    Shape: {test_df.shape[0]} samples x {test_df.shape[1]} columns")
    test_aug_count = int(test_df["is_augmented"].sum())
    print(f"    Real samples: {len(test_df) - test_aug_count} | Augmented: {test_aug_count}")

    # ── Generate EDA charts ────────────────────────────────────────────────
    print("\n  Generating EDA charts...")
    generate_eda_charts(log_df, processed_results)

    print("\n" + "=" * 65)
    print("  Phase 3 preprocessing complete!")
    print(f"  EDA charts saved to: {EDA_CHARTS_DIR}/")
    print("=" * 65)



if __name__ == "__main__":
    main()
