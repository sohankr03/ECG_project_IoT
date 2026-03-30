"""
feature_extraction.py
─────────────────────────────────────────────────────────────────────────────
Feature Extraction Module for the ECG Anomaly Detection System.

Extracts 6 HRV (Heart Rate Variability) features from a 5-second ECG window:
  1. heart_rate    — BPM
  2. rr_mean       — Mean RR interval (ms)
  3. rr_std        — Std of RR intervals (RR variability)
  4. sdnn          — Std of NN intervals
  5. rmssd         — Root mean square of successive differences
  6. beat_variance — Beat-to-beat interval variance

Also computes a Signal Quality Index (SQI) used to gate inference:
  - If SQI < threshold → skip classification, mark as "Poor Signal Quality"

Note:
  - QRS width was removed — single-lead AD8232 gives unreliable QRS width
    estimates due to electrode placement limitations.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from typing import Dict, Optional
from signal_processing import bandpass_filter, pan_tompkins_qrs, compute_rr_intervals


# ── SQI Thresholds ────────────────────────────────────────────────────────
SQI_MIN_VARIANCE   = 1000.0   # Raw ADC units² — signal too flat → bad contact
SQI_MIN_AMPLITUDE  = 50.0     # Raw ADC units  — peak-to-peak too small
SQI_CONSISTENCY    = 0.35     # Coefficient of variation of peak amplitudes


# ══════════════════════════════════════════════════════════════════════════
# Signal Quality Index
# ══════════════════════════════════════════════════════════════════════════

def compute_sqi(raw_signal: np.ndarray, r_peaks: np.ndarray, fs: float = 250.0) -> float:
    """
    Compute a Signal Quality Index (SQI) in [0.0, 1.0].

    Criteria:
      1. Signal variance — too flat means electrode not connected properly
      2. Peak-to-peak amplitude — minimum meaningful swing
      3. Peak amplitude consistency — high CV → noisy/motion artifact

    Returns
    -------
    sqi : float in [0.0, 1.0]
          1.0 = excellent, 0.0 = unusable
    """
    score = 0.0
    total = 3.0

    # 1. Variance test
    variance = np.var(raw_signal)
    if variance >= SQI_MIN_VARIANCE:
        score += 1.0

    # 2. Peak-to-peak amplitude test
    ptp = np.ptp(raw_signal)  # peak-to-peak
    if ptp >= SQI_MIN_AMPLITUDE:
        score += 1.0

    # 3. Peak amplitude consistency (requires at least 3 peaks)
    if len(r_peaks) >= 3:
        amplitudes = raw_signal[r_peaks]
        cv = np.std(amplitudes) / (np.mean(amplitudes) + 1e-9)
        if cv < SQI_CONSISTENCY:
            score += 1.0
    else:
        # Not enough peaks — partial credit
        score += 0.5

    return score / total


# ══════════════════════════════════════════════════════════════════════════
# HRV Feature Extraction
# ══════════════════════════════════════════════════════════════════════════

def extract_features(ecg_window: np.ndarray,
                     fs: float = 250.0,
                     sqi_threshold: float = 0.5) -> Dict:
    """
    Extract 6 HRV features from a 5-second ECG window.

    Parameters
    ----------
    ecg_window    : 1-D numpy array (raw ADC samples, ~1250 points at 250 Hz)
    fs            : Sampling frequency in Hz (default 250)
    sqi_threshold : Minimum SQI to proceed; below this returns quality_ok=False

    Returns
    -------
    results : dict with keys:
        quality_ok    (bool)   — False if poor signal or insufficient beats
        sqi           (float)  — Signal Quality Index [0–1]
        heart_rate    (float)  — BPM
        rr_mean       (float)  — Mean RR interval (ms)
        rr_std        (float)  — Std of RR intervals (ms)
        sdnn          (float)  — Std of NN intervals (ms) [same as rr_std for 5s window]
        rmssd         (float)  — Root mean square of successive differences (ms)
        beat_variance (float)  — Beat-to-beat interval variance (ms²)
        r_peak_count  (int)    — Number of detected R-peaks
    """

    # ── 1. Bandpass filter ────────────────────────────────────────────
    try:
        filtered = bandpass_filter(ecg_window, fs=fs)
    except Exception:
        return _empty_result(quality_ok=False, sqi=0.0, reason="Filter failed")

    # ── 2. QRS detection ──────────────────────────────────────────────
    r_peaks = pan_tompkins_qrs(filtered, fs=fs)

    # ── 3. Signal Quality Index ───────────────────────────────────────
    sqi = compute_sqi(ecg_window, r_peaks, fs=fs)

    if sqi < sqi_threshold:
        return _empty_result(quality_ok=False, sqi=sqi, reason="Poor signal quality")

    # ── 4. Minimum beat check ─────────────────────────────────────────
    if len(r_peaks) < 3:
        return _empty_result(quality_ok=False, sqi=sqi, reason="Not enough beats detected")

    # ── 5. RR intervals ───────────────────────────────────────────────
    rr_ms = compute_rr_intervals(r_peaks, fs=fs)

    if len(rr_ms) < 2:
        return _empty_result(quality_ok=False, sqi=sqi, reason="Insufficient RR intervals")

    # ── 6. Outlier rejection — remove physiologically impossible intervals ──
    # Valid HR range: 20–250 BPM → RR range: 240–3000 ms
    rr_valid = rr_ms[(rr_ms >= 240) & (rr_ms <= 3000)]
    if len(rr_valid) < 2:
        return _empty_result(quality_ok=False, sqi=sqi, reason="RR intervals out of physiological range")

    # ── 7. Feature Computation ────────────────────────────────────────

    # Heart rate (BPM)
    heart_rate = 60000.0 / np.mean(rr_valid)

    # Time-domain HRV features
    rr_mean       = float(np.mean(rr_valid))
    rr_std        = float(np.std(rr_valid, ddof=1))
    sdnn          = rr_std  # SDNN ≡ std of NN (normal-to-normal) intervals

    # RMSSD — root mean square of successive RR differences
    successive_diffs = np.diff(rr_valid)
    rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))

    # Beat-to-beat interval variance
    beat_variance = float(np.var(rr_valid, ddof=1))

    return {
        "quality_ok"    : True,
        "sqi"           : round(sqi, 3),
        "heart_rate"    : round(heart_rate, 2),
        "rr_mean"       : round(rr_mean, 2),
        "rr_std"        : round(rr_std, 2),
        "sdnn"          : round(sdnn, 2),
        "rmssd"         : round(rmssd, 2),
        "beat_variance" : round(beat_variance, 2),
        "r_peak_count"  : int(len(r_peaks)),
    }


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _empty_result(quality_ok: bool, sqi: float, reason: str = "") -> Dict:
    """Return a placeholder result when extraction cannot proceed."""
    return {
        "quality_ok"    : quality_ok,
        "sqi"           : round(sqi, 3),
        "heart_rate"    : 0.0,
        "rr_mean"       : 0.0,
        "rr_std"        : 0.0,
        "sdnn"          : 0.0,
        "rmssd"         : 0.0,
        "beat_variance" : 0.0,
        "r_peak_count"  : 0,
        "reason"        : reason,
    }


# Feature names used for model training/inference (order matters!)
FEATURE_COLUMNS = [
    "heart_rate",
    "rr_mean",
    "rr_std",
    "sdnn",
    "rmssd",
    "beat_variance",
]


def features_to_vector(feature_dict: Dict) -> Optional[np.ndarray]:
    """
    Convert a feature dict to a 1-D numpy array in the correct column order.
    Returns None if quality_ok is False.
    """
    if not feature_dict.get("quality_ok", False):
        return None
    return np.array([feature_dict[col] for col in FEATURE_COLUMNS], dtype=float)
