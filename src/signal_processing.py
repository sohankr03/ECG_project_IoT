"""
signal_processing.py
─────────────────────────────────────────────────────────────────────────────
Signal Processing Module for the ECG Anomaly Detection System.

Functions:
  - bandpass_filter(signal, fs)       : 4th-order Butterworth bandpass 0.5–40 Hz
  - pan_tompkins_qrs(signal, fs)      : Full Pan-Tompkins QRS detection pipeline
  - compute_rr_intervals(r_peaks, fs) : RR intervals in milliseconds

Design Notes:
  - Butterworth chosen for smooth frequency response, no ripple, real-time use.
  - Pan-Tompkins is the clinical gold standard for lightweight QRS detection.
  - 200 ms refractory period prevents double-peak detection on noisy AD8232 data.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from scipy.signal import butter, filtfilt, lfilter


# ══════════════════════════════════════════════════════════════════════════
# 1. Bandpass Filter
# ══════════════════════════════════════════════════════════════════════════

def bandpass_filter(signal: np.ndarray, fs: float = 250.0,
                    lowcut: float = 0.5, highcut: float = 40.0,
                    order: int = 4) -> np.ndarray:
    """
    4th-order Butterworth bandpass filter.

    Removes:
      - Baseline drift (< 0.5 Hz)
      - Muscle noise / power-line interference (> 40 Hz)

    Parameters
    ----------
    signal  : 1-D numpy array of raw ECG samples
    fs      : Sampling frequency in Hz (default 250)
    lowcut  : Low cutoff frequency in Hz  (default 0.5)
    highcut : High cutoff frequency in Hz (default 40)
    order   : Filter order (default 4)

    Returns
    -------
    filtered : 1-D numpy array, same length as input
    """
    nyquist = fs / 2.0
    low  = lowcut  / nyquist
    high = highcut / nyquist

    if high >= 1.0:
        high = 0.99  # safety clamp

    b, a = butter(order, [low, high], btype='bandpass')
    # filtfilt: zero-phase, no group delay (safe for offline windows)
    filtered = filtfilt(b, a, signal)
    return filtered


# ══════════════════════════════════════════════════════════════════════════
# 2. Pan-Tompkins QRS Detection
# ══════════════════════════════════════════════════════════════════════════

def pan_tompkins_qrs(signal: np.ndarray, fs: float = 250.0) -> np.ndarray:
    """
    Pan-Tompkins QRS detection algorithm.

    Pipeline:
      1. Derivative filter  — accentuates QRS slopes
      2. Squaring           — amplifies large values, ensures positivity
      3. Moving window integration (window = 0.15 s)
      4. Adaptive thresholding with signal/noise peak tracking
      5. Refractory period  — ignore peaks within 200 ms of previous peak

    Parameters
    ----------
    signal : 1-D numpy array (should be bandpass filtered first!)
    fs     : Sampling frequency in Hz (default 250)

    Returns
    -------
    r_peaks : 1-D numpy array of R-peak sample indices
    """
    n = len(signal)
    if n < int(0.5 * fs):
        # Too short to detect peaks reliably
        return np.array([], dtype=int)

    # ── Step 1: Derivative filter ─────────────────────────────────────
    # Approximation: y[n] = (1/8)(-2x[n-2] - x[n-1] + x[n+1] + 2x[n+2])
    deriv = np.zeros(n)
    for i in range(2, n - 2):
        deriv[i] = (1.0 / 8.0) * (
            -2 * signal[i - 2] - signal[i - 1]
            + signal[i + 1] + 2 * signal[i + 2]
        )

    # ── Step 2: Squaring ──────────────────────────────────────────────
    squared = deriv ** 2

    # ── Step 3: Moving window integration ────────────────────────────
    # Window ≈ 150 ms
    win_size = int(0.15 * fs)
    if win_size < 1:
        win_size = 1
    integrated = np.convolve(squared, np.ones(win_size) / win_size, mode='same')

    # ── Step 4: Adaptive Thresholding ─────────────────────────────────
    # Initialise thresholds from first 2 seconds
    init_samples = min(int(2 * fs), n)
    signal_peak_i  = np.max(integrated[:init_samples])
    noise_peak_i   = 0.5 * signal_peak_i
    threshold_i1   = noise_peak_i + 0.25 * (signal_peak_i - noise_peak_i)

    # Refractory period in samples (200 ms)
    refractory_samples = int(0.200 * fs)

    r_peaks = []
    last_peak = -refractory_samples  # allow detection from start

    # Scan for local maxima exceeding threshold
    # Use a search window of 36 samples (≈ 144 ms) around each candidate
    search_win = int(0.144 * fs)
    i = 0
    while i < n:
        if integrated[i] > threshold_i1:
            # Find local maximum within search window
            win_start = i
            win_end   = min(i + search_win, n)
            local_max_idx = win_start + np.argmax(integrated[win_start:win_end])

            # Enforce refractory period
            if local_max_idx - last_peak > refractory_samples:
                r_peaks.append(local_max_idx)
                last_peak = local_max_idx

                # Update signal peak and threshold
                signal_peak_i = 0.125 * integrated[local_max_idx] + 0.875 * signal_peak_i
            else:
                # Treat as noise peak
                noise_peak_i = 0.125 * integrated[local_max_idx] + 0.875 * noise_peak_i

            # Update threshold
            threshold_i1 = noise_peak_i + 0.25 * (signal_peak_i - noise_peak_i)
            i = win_end  # skip past search window
        else:
            i += 1

    return np.array(r_peaks, dtype=int)


# ══════════════════════════════════════════════════════════════════════════
# 3. RR Interval Computation
# ══════════════════════════════════════════════════════════════════════════

def compute_rr_intervals(r_peaks: np.ndarray, fs: float = 250.0) -> np.ndarray:
    """
    Compute RR intervals in milliseconds from R-peak sample indices.

    Parameters
    ----------
    r_peaks : 1-D numpy array of R-peak indices
    fs      : Sampling frequency in Hz (default 250)

    Returns
    -------
    rr_ms : 1-D numpy array of RR intervals in milliseconds.
            Length = len(r_peaks) - 1.
            Returns empty array if fewer than 2 peaks.
    """
    if len(r_peaks) < 2:
        return np.array([], dtype=float)

    rr_samples = np.diff(r_peaks).astype(float)
    rr_ms = (rr_samples / fs) * 1000.0  # convert samples → ms
    return rr_ms


# ══════════════════════════════════════════════════════════════════════════
# Utility: Normalise to [0, 1] for display purposes only
# ══════════════════════════════════════════════════════════════════════════

def normalise_for_display(signal: np.ndarray) -> np.ndarray:
    """Min-max normalise signal to [0, 1] for waveform display only."""
    s_min = signal.min()
    s_max = signal.max()
    if s_max == s_min:
        return np.zeros_like(signal)
    return (signal - s_min) / (s_max - s_min)
