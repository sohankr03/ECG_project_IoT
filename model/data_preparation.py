"""
data_preparation.py
─────────────────────────────────────────────────────────────────────────────
MIT-BIH Arrhythmia Dataset → Feature CSV for Random Forest Training.

Steps:
  1. Load MIT-BIH records via wfdb (auto-downloads from PhysioNet if needed)
  2. For each record: slide 5-second windows at 50% overlap
  3. Extract 6 HRV features per window using feature_extraction.py
  4. Label: Normal (0) or Abnormal (1) based on beat annotation types
  5. Fit StandardScaler on training features → save as scaler_v1.pkl
  6. Save features_dataset.csv

Run:
  cd E:\Project\ECG_Project\model
  python data_preparation.py

Outputs:
  model/features_dataset.csv      — labeled feature dataset
  model/scaler_v1.pkl             — fitted StandardScaler (MUST use at inference)
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import numpy as np
import pandas as pd
import wfdb
import joblib
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# ── Path Setup ────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
MODEL_DIR  = ROOT_DIR / "model"
SRC_DIR    = ROOT_DIR / "src"
DATA_DIR   = ROOT_DIR / "assets" / "mitbih_data"
sys.path.insert(0, str(SRC_DIR))

from feature_extraction import extract_features, FEATURE_COLUMNS

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── MIT-BIH Configuration ─────────────────────────────────────────────────
# Records selected for diversity: normal + various arrhythmia types
# Normal-heavy: 100, 101, 103, 105, 108
# Arrhythmia:   119(PVC), 200(PVC+VE), 201(PVC), 203, 210, 212, 213
RECORDS = [
    "100", "101", "103", "105", "108",   # Normal + mild abnormalities
    "119", "200", "201", "203", "210",   # PVCs and ventricular beats
    "212", "213", "217", "219", "221",   # Mixed arrhythmias
]

# MIT-BIH native sampling rate
MITBIH_FS = 360  # Hz

# Target sampling rate for our system
TARGET_FS = 250  # Hz

# Window configuration
WINDOW_SECONDS = 5.0
WINDOW_SAMPLES_MITBIH  = int(WINDOW_SECONDS * MITBIH_FS)   # 1800 samples
OVERLAP_RATIO          = 0.5
STEP_SAMPLES           = int(WINDOW_SAMPLES_MITBIH * (1 - OVERLAP_RATIO))  # 900

# ── Annotation Mapping ────────────────────────────────────────────────────
# MIT-BIH AAMI annotation categories
# Normal beat types
NORMAL_SYMBOLS = {'N', 'L', 'R', 'e', 'j'}

# Abnormal beat types (arrhythmic)
ABNORMAL_SYMBOLS = {
    'A',  # Atrial premature beat
    'a',  # Aberrated atrial premature beat
    'J',  # Nodal (junctional) premature beat
    'S',  # Supraventricular premature beat
    'V',  # Premature ventricular contraction
    'F',  # Fusion of ventricular and normal beat
    'f',  # Fusion of paced and normal beat
    '[',  # Start of ventricular flutter/fibrillation
    '!',  # Ventricular flutter wave
    ']',  # End of ventricular flutter/fibrillation
    'E',  # Ventricular escape beat
    '/',  # Paced beat
    'Q',  # Unclassifiable beat
    'P',  # Premature ventricular
}


def resample_signal(signal: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    """Resample signal from original_fs to target_fs using linear interpolation."""
    n_original = len(signal)
    n_target   = int(n_original * target_fs / original_fs)
    x_original = np.linspace(0, 1, n_original)
    x_target   = np.linspace(0, 1, n_target)
    return np.interp(x_target, x_original, signal)


def label_window(window_start: int, window_end: int,
                 ann_samples: np.ndarray, ann_symbols: list) -> int:
    """
    Label a window as Normal (0) or Abnormal (1).

    If any beat annotation within the window is in ABNORMAL_SYMBOLS → label 1.
    If all beats are in NORMAL_SYMBOLS → label 0.
    Windows with no annotated beats are skipped (return -1).
    """
    mask = (ann_samples >= window_start) & (ann_samples < window_end)
    beats_in_window = [ann_symbols[i] for i, m in enumerate(mask) if m]

    if not beats_in_window:
        return -1  # no beats → skip

    for beat in beats_in_window:
        if beat in ABNORMAL_SYMBOLS:
            return 1  # at least one abnormal beat → Abnormal

    return 0  # all normal → Normal


def process_record(record_name: str) -> list:
    """
    Download and process one MIT-BIH record.

    Returns list of dicts: {feature_columns..., label}
    """
    rows = []
    try:
        print(f"  → Loading record {record_name}...")
        record = wfdb.rdrecord(record_name, pn_dir='mitdb')
        ann    = wfdb.rdann(record_name, 'atr', pn_dir='mitdb')
    except Exception as e:
        print(f"  ✗ Failed to load {record_name}: {e}")
        return rows

    # Use first channel (MLII lead)
    raw_signal  = record.p_signal[:, 0].astype(float)
    ann_samples = ann.sample
    ann_symbols = ann.symbol
    total_samples = len(raw_signal)

    # Scale to mock ADC units (0–4095) for compatibility with SQI thresholds
    # MIT-BIH is in mV → scale to 12-bit ADC range
    sig_min, sig_max = raw_signal.min(), raw_signal.max()
    if sig_max > sig_min:
        raw_signal = ((raw_signal - sig_min) / (sig_max - sig_min)) * 4095.0
    else:
        raw_signal = np.zeros_like(raw_signal)

    # Slide windows
    start = 0
    while start + WINDOW_SAMPLES_MITBIH <= total_samples:
        end = start + WINDOW_SAMPLES_MITBIH
        window = raw_signal[start:end]

        # Label this window
        label = label_window(start, end, ann_samples, ann_symbols)
        if label == -1:
            start += STEP_SAMPLES
            continue

        # Resample to 250 Hz before feature extraction
        window_250 = resample_signal(window, MITBIH_FS, TARGET_FS)

        # Extract features
        features = extract_features(window_250, fs=TARGET_FS, sqi_threshold=0.4)

        if not features.get("quality_ok", False):
            start += STEP_SAMPLES
            continue

        row = {col: features[col] for col in FEATURE_COLUMNS}
        row["label"] = label
        row["record"] = record_name
        rows.append(row)

        start += STEP_SAMPLES

    print(f"     ✓ {record_name}: {len(rows)} windows extracted")
    return rows


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  MIT-BIH Data Preparation")
    print("=" * 60)

    all_rows = []
    for record in RECORDS:
        rows = process_record(record)
        all_rows.extend(rows)

    if not all_rows:
        print("\n✗ No data extracted. Check network / wfdb installation.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    print(f"\nTotal windows: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    # ── Fit StandardScaler on all features ───────────────────────────
    X = df[FEATURE_COLUMNS].values
    scaler = StandardScaler()
    scaler.fit(X)

    scaler_path = MODEL_DIR / "scaler_v1.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"\n✓ Scaler saved → {scaler_path}")

    # ── Save dataset ─────────────────────────────────────────────────
    dataset_path = MODEL_DIR / "features_dataset.csv"
    df.to_csv(dataset_path, index=False)
    print(f"✓ Dataset saved → {dataset_path}")
    print(f"\nFeature columns: {FEATURE_COLUMNS}")


if __name__ == "__main__":
    main()
