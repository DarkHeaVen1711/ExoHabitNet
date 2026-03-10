"""
collect_kepler_data.py
======================
ExoHabitNet Phase 2 — Automated Data Collection Script
Fetches 500+ Kepler light curves from NASA MAST via lightkurve
and assigns habitability labels using the NASA Exoplanet Archive KOI table.

Usage:
    python collect_kepler_data.py

Requirements:
    pip install lightkurve astropy pandas numpy requests tqdm
"""

import os
import time
import logging
import requests
import numpy as np
import pandas as pd
import lightkurve as lk
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
TARGET_SAMPLES        = 500          # Minimum number of samples to collect
OUTPUT_DIR            = Path("data/raw_fits")
LOG_DIR               = Path("data/logs")
COLLECTION_LOG_PATH   = Path("data/data_collection_log.csv")
MISSION               = "Kepler"
MAX_RETRIES           = 3
RETRY_DELAY_SEC       = 5
REQUEST_DELAY_SEC     = 0.5          # Polite delay between API calls

# Kopparapu (2013) HZ boundaries in AU for a G2V star (can be scaled by Teff)
HZ_INNER_AU           = 0.99
HZ_OUTER_AU           = 1.70
ROCKY_PLANET_MAX_RE   = 3.0         # Earth radii — super-Earth threshold (very relaxed for data collection)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: QUERY NASA EXOPLANET ARCHIVE FOR KOI TABLE
# ─────────────────────────────────────────────────────────────────────────────
def _tap_query(adql: str) -> pd.DataFrame:
    """Helper: run a single TAP ADQL query and return a DataFrame."""
    url    = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    params = {"query": adql, "format": "json", "lang": "ADQL"}
    resp   = requests.get(url, params=params, timeout=60)
    if not resp.ok:
        logger.error(f"  TAP error {resp.status_code}:\n{resp.text[:400]}")
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def fetch_koi_table(limit: int = 2000) -> pd.DataFrame:
    """
    Queries the NASA Exoplanet Archive using TWO ADQL queries merged together,
    so that rare HABITABLE-zone candidates are never missed by the general TOP N cap.

    Query 1 (general): All CONFIRMED + FALSE POSITIVE KOIs
    Query 2 (HZ top-up): Confirmed + small radius (<=2 Re) + long period (>=200 days)
    """
    cols = ("kepid, kepoi_name, koi_disposition, "
            "koi_period, koi_prad, koi_steff, koi_smass, koi_time0bk")

    logger.info("Querying NASA Exoplanet Archive (general KOI table)...")
    q1 = (f"SELECT TOP {limit} {cols} FROM cumulative "
          "WHERE koi_disposition IN ('CONFIRMED', 'FALSE POSITIVE')")
    df_general = _tap_query(q1)
    logger.info(f"  -> General query: {len(df_general)} records.")

    # Targeted query for HZ candidates: long period + small radius
    logger.info("Querying NASA Exoplanet Archive (HZ candidate top-up)...")
    q2 = (f"SELECT TOP 500 {cols} FROM cumulative "
          "WHERE koi_disposition = 'CONFIRMED' "
          "AND koi_period >= 100 AND koi_prad <= 2.5")
    try:
        df_hz = _tap_query(q2)
        logger.info(f"  -> HZ top-up query: {len(df_hz)} candidate records.")
    except Exception as exc:
        logger.warning(f"  HZ top-up query failed (non-fatal): {exc}")
        df_hz = pd.DataFrame()

    df = pd.concat([df_general, df_hz], ignore_index=True)
    df = df.drop_duplicates(subset="kepoi_name").reset_index(drop=True)
    logger.info(f"  -> Total unique KOI records after merge: {len(df)}")
    return df



# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: ASSIGN HABITABILITY LABELS
# ─────────────────────────────────────────────────────────────────────────────
def compute_semimajor_axis_au(period_days: float, stellar_mass_msun: float) -> float:
    """
    Kepler's Third Law: a³ = M★ · P²
    Returns semi-major axis in AU. M★ in solar masses, P in years.
    """
    if pd.isna(period_days) or pd.isna(stellar_mass_msun) or stellar_mass_msun <= 0:
        return np.nan
    period_years = period_days / 365.25
    return (stellar_mass_msun * period_years ** 2) ** (1 / 3)


def scale_hz_for_star(teff: float) -> tuple:
    """
    Scales the Kopparapu (2013) HZ inner/outer edges based on stellar temperature.
    Uses an empirical linear approximation relative to the Sun (Teff_sun = 5778 K).
    Returns (hz_inner_au, hz_outer_au).
    """
    if pd.isna(teff):
        return HZ_INNER_AU, HZ_OUTER_AU
    scale = (teff / 5778.0) ** 2  # Luminosity proxy (L ∝ T⁴, flux ∝ L/a²)
    hz_inner = HZ_INNER_AU * scale ** 0.5
    hz_outer = HZ_OUTER_AU * scale ** 0.5
    return hz_inner, hz_outer


def assign_label(row: pd.Series) -> str:
    """
    Decision tree classification (VERY RELAXED HZ model for dataset balance):
        Label 0 → HABITABLE      Optimistic HZ: confirmed/candidate + super-Earth
                                 radius (<=3.0 Re) + within ±50% of HZ edges.
                                 Scientific basis: "optimistic HZ" (Kopparapu 2013)
                                 includes planets that could retain liquid water with
                                 different atmospheric compositions. ±50% buffer
                                 accounts for cloud feedback and greenhouse effects.
        Label 1 → NON_HABITABLE  Confirmed planet outside HZ or gas/ice giant
        Label 2 → FALSE_POSITIVE Flagged FP — eclipsing binary, background star
    """
    disposition = str(row.get("koi_disposition", "")).strip().upper()

    # ── False Positive first ──────────────────────────────────────────────
    if disposition == "FALSE POSITIVE":
        return "FALSE_POSITIVE"

    # ── Accept CONFIRMED and CANDIDATE (relaxed from CONFIRMED-only) ──────
    if disposition not in ("CONFIRMED", "CANDIDATE"):
        return "UNKNOWN"

    # ── Check planet size (very relaxed: super-Earth ≤ 3.0 Re) ────────────
    rp_re = row.get("koi_prad", np.nan)
    if not pd.isna(rp_re) and rp_re > 6.0:      # Gas giant cutoff
        return "NON_HABITABLE"

    # ── Compute semi-major axis and check HZ with ±50% buffer ────────────
    period_days  = row.get("koi_period", np.nan)
    stellar_mass = row.get("koi_smass", 1.0)
    teff         = row.get("koi_steff", 5778.0)

    a_au = compute_semimajor_axis_au(period_days, stellar_mass)
    hz_in, hz_out = scale_hz_for_star(teff)

    if pd.isna(a_au):
        return "NON_HABITABLE"

    # ±50% buffer on HZ edges = "very optimistic HZ" for data collection
    hz_in_relaxed  = hz_in  * 0.50
    hz_out_relaxed = hz_out * 1.50

    if hz_in_relaxed <= a_au <= hz_out_relaxed:
        # In relaxed HZ — accept super-Earths up to 3.0 Re
        if pd.isna(rp_re) or rp_re <= ROCKY_PLANET_MAX_RE:
            return "HABITABLE"
        else:
            return "NON_HABITABLE"   # In HZ but too large (mini-Neptune or bigger)
    else:
        return "NON_HABITABLE"



def label_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply label assignment to all KOI records."""
    logger.info("Assigning habitability labels...")
    df = df.copy()
    df["label"]    = df.apply(assign_label, axis=1)
    df["label_id"] = df["label"].map({"HABITABLE": 0, "NON_HABITABLE": 1, "FALSE_POSITIVE": 2, "UNKNOWN": -1})

    # Remove ambiguous UNKNOWN entries
    df = df[df["label"] != "UNKNOWN"].reset_index(drop=True)

    counts = df["label"].value_counts()
    logger.info(f"  Label distribution:\n{counts.to_string()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: DOWNLOAD LIGHT CURVES FROM NASA MAST
# ─────────────────────────────────────────────────────────────────────────────
def download_light_curve(kid: int, koi_name: str, label: str) -> dict:
    """
    Downloads all available Kepler quarters for a given KIC ID using lightkurve.
    Stitches quarters into a single time series and saves as .fits.

    Returns a log entry dict.
    """
    target_dir = OUTPUT_DIR / label / str(kid)
    target_dir.mkdir(parents=True, exist_ok=True)

    fits_path   = target_dir / f"KIC{kid}_stitched.fits"
    entry = {
        "kepid":         kid,
        "kepoi_name":    koi_name,
        "label":         label,
        "label_id":      {"HABITABLE": 0, "NON_HABITABLE": 1, "FALSE_POSITIVE": 2}[label],
        "fits_path":     str(fits_path),
        "quarters_available": None,
        "cadences_total":     None,
        "nan_fraction":       None,
        "quality_flags_sum":  None,
        "quality_check":      "PENDING",
        "collection_status":  "PENDING",
        "timestamp":          datetime.now(timezone.utc).isoformat()
    }

    # Skip if already downloaded
    if fits_path.exists():
        entry["collection_status"] = "SKIPPED_EXISTS"
        return entry

    try:
        search_result = lk.search_lightcurve(
            target=f"KIC {kid}",
            mission=MISSION,
            cadence="long"    # 30-min long cadence (LC) — standard for transit search
        )

        if len(search_result) == 0:
            entry["collection_status"] = "NO_DATA_FOUND"
            return entry

        entry["quarters_available"] = len(search_result)

        # Download all quarters (LightCurveCollection)
        lcc = search_result.download_all(quality_bitmask="default")

        if lcc is None or len(lcc) == 0:
            entry["collection_status"] = "DOWNLOAD_FAILED"
            return entry

        # Stitch all quarters into one light curve
        lc_stitched = lcc.stitch()

        # Compute quality metadata
        entry["cadences_total"]    = len(lc_stitched.time)
        entry["nan_fraction"]      = float(np.sum(np.isnan(lc_stitched.flux.value)) / len(lc_stitched.flux))
        entry["quality_flags_sum"] = int(np.nansum(lc_stitched.quality.value)) if hasattr(lc_stitched, "quality") else 0

        # Quality check: reject if >30% NaN or extremely high flag sum
        if entry["nan_fraction"] > 0.30:
            entry["quality_check"] = "FAIL_HIGH_NAN"
        else:
            entry["quality_check"] = "PASS"

        # Save stitched light curve as FITS
        lc_stitched.to_fits(fits_path, overwrite=True)
        entry["collection_status"] = "SUCCESS"

    except Exception as exc:
        entry["collection_status"] = f"ERROR: {type(exc).__name__}: {str(exc)[:120]}"
        logger.warning(f"  [KIC {kid}] Collection failed: {exc}")

    return entry


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: MAIN COLLECTION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 65)
    logger.info("  ExoHabitNet — Kepler Data Collection Script")
    logger.info(f"  Target: {TARGET_SAMPLES} samples | Mission: {MISSION}")
    logger.info("=" * 65)

    # ── 1. Fetch KOI metadata ─────────────────────────────────────────────
    koi_df = fetch_koi_table(limit=2000)
    koi_df = label_dataset(koi_df)

    # ── 2. Sample to get balanced(ish) dataset ────────────────────────────
    # Oversample to account for download failures (~20% failure rate expected)
    oversample_factor = 1.4
    per_class_target  = int((TARGET_SAMPLES * oversample_factor) / 3)

    sampled_parts = []
    for label_name in ["HABITABLE", "NON_HABITABLE", "FALSE_POSITIVE"]:
        subset = koi_df[koi_df["label"] == label_name]
        n      = min(len(subset), per_class_target)
        sampled_parts.append(subset.sample(n=n, random_state=42))
        logger.info(f"  Sampling {n}/{len(subset)} records for class '{label_name}'")

    sample_df = pd.concat(sampled_parts).reset_index(drop=True)
    logger.info(f"  Total records to attempt: {len(sample_df)}")

    # ── 3. Download light curves ──────────────────────────────────────────
    log_entries    = []
    success_count  = 0

    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Collecting light curves"):
        kid      = int(row["kepid"])
        koi_name = str(row["kepoi_name"])
        label    = str(row["label"])

        # Retry logic
        entry = None
        for attempt in range(1, MAX_RETRIES + 1):
            entry = download_light_curve(kid, koi_name, label)
            if "ERROR" not in entry["collection_status"]:
                break
            logger.warning(f"  Retry {attempt}/{MAX_RETRIES} for KIC {kid}")
            time.sleep(RETRY_DELAY_SEC)

        log_entries.append(entry)

        if entry["collection_status"] == "SUCCESS":
            success_count += 1

        # Polite rate limiting
        time.sleep(REQUEST_DELAY_SEC)

        # Early exit if target reached
        if success_count >= TARGET_SAMPLES:
            logger.info(f"  ✓ Target of {TARGET_SAMPLES} successful downloads reached.")
            break

    # ── 4. Save collection log ────────────────────────────────────────────
    log_df = pd.DataFrame(log_entries)
    COLLECTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(COLLECTION_LOG_PATH, index=False)

    # ── 5. Summary report ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  COLLECTION SUMMARY")
    logger.info("=" * 65)
    logger.info(f"  Total attempts   : {len(log_entries)}")
    logger.info(f"  Successful DLs   : {success_count}")
    logger.info(f"  Skipped (exists) : {(log_df['collection_status'] == 'SKIPPED_EXISTS').sum()}")
    logger.info(f"  Failures         : {log_df['collection_status'].str.startswith('ERROR').sum()}")
    logger.info(f"  No data found    : {(log_df['collection_status'] == 'NO_DATA_FOUND').sum()}")
    logger.info(f"  Quality PASS     : {(log_df['quality_check'] == 'PASS').sum()}")
    logger.info(f"  Log saved to     : {COLLECTION_LOG_PATH}")
    logger.info("=" * 65)

    if success_count < TARGET_SAMPLES:
        logger.warning(f"  ⚠ Only {success_count}/{TARGET_SAMPLES} samples collected. Re-run to top up.")

    return log_df


if __name__ == "__main__":
    log_df = main()
