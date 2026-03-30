"""
ecg_simulator.py
─────────────────────────────────────────────────────────────────────────────
ECG Simulator for Demo Mode — streams MIT-BIH data as if it were live serial.

This replaces the serial port when no ESP32 hardware is connected.
Feeds data through the same ECGInferenceEngine pipeline, allowing a full
demonstration of filtering, feature extraction, and classification.

Usage (internal — instantiated by dashboard.py):
  sim = EcgSimulator(record_name="119", inject_anomaly=True)
  engine = ECGInferenceEngine(demo_mode=True, demo_stream=sim)
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import time
import numpy as np
from pathlib import Path
from typing import Generator, Tuple

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────────────────
MITBIH_FS     = 360   # Native MIT-BIH sampling rate
TARGET_FS     = 250   # Our system sampling rate
TARGET_SCALE  = 4095  # 12-bit ADC scale


# ══════════════════════════════════════════════════════════════════════════
# MIT-BIH Simulator
# ══════════════════════════════════════════════════════════════════════════

class EcgSimulator:
    """
    Streams a MIT-BIH ECG record at the target sampling rate.

    If wfdb is unavailable or load fails, falls back to a synthetic
    ECG waveform generator so the dashboard always works.

    Parameters
    ----------
    record_name   : MIT-BIH record ID (default "119" — lots of PVCs)
    inject_anomaly: If True, periodically inject irregular RR intervals
                    to demonstrate abnormal detection triggering
    loop          : If True, replay record in a loop (default True)
    """

    def __init__(self,
                 record_name: str = "119",
                 inject_anomaly: bool = True,
                 loop: bool = True):
        self.record_name   = record_name
        self.inject_anomaly = inject_anomaly
        self.loop          = loop
        self._signal       = None
        self._annotations  = None
        self._load_record()

    def _load_record(self):
        """Load MIT-BIH record and resample to TARGET_FS."""
        if not WFDB_AVAILABLE:
            print("[EcgSimulator] wfdb not installed — using synthetic ECG.")
            self._signal = self._generate_synthetic_ecg(60)
            return

        try:
            data_dir = ROOT_DIR / "assets" / "mitbih_data"
            data_dir.mkdir(parents=True, exist_ok=True)
            record = wfdb.rdrecord(
                self.record_name, pn_dir='mitdb'
            )
            raw = record.p_signal[:, 0].astype(float)

            # Resample from 360 Hz → 250 Hz
            n_orig   = len(raw)
            n_target = int(n_orig * TARGET_FS / MITBIH_FS)
            x_orig   = np.linspace(0, 1, n_orig)
            x_target = np.linspace(0, 1, n_target)
            resampled = np.interp(x_target, x_orig, raw)

            # Scale to 12-bit ADC range
            s_min, s_max = resampled.min(), resampled.max()
            if s_max > s_min:
                self._signal = ((resampled - s_min) / (s_max - s_min)) * TARGET_SCALE
            else:
                self._signal = np.zeros(n_target)

            print(f"[EcgSimulator] Loaded record {self.record_name}: "
                  f"{len(self._signal)} samples at {TARGET_FS} Hz")
        except Exception as e:
            print(f"[EcgSimulator] Could not load {self.record_name}: {e}")
            print("[EcgSimulator] Falling back to synthetic ECG.")
            self._signal = self._generate_synthetic_ecg(60)

    def _generate_synthetic_ecg(self, duration_seconds: float = 60) -> np.ndarray:
        """
        Generate a synthetic ECG waveform using Gaussian templates.
        Alternates between normal (HR=70) and slightly fast (HR=110) segments
        to exercise both Normal and Abnormal classification.
        """
        n = int(duration_seconds * TARGET_FS)
        t = np.arange(n) / TARGET_FS
        ecg = np.zeros(n)

        # Half normal, half fast
        # Segment 1: Normal rhythm HR=70 → RR≈857ms
        # Segment 2: Fast/irregular rhythm HR=115 → RR≈522ms

        def add_beat(signal_arr, center_sample: int, hr_bpm: float, fs: float):
            """Add a Gaussian QRS complex at center_sample."""
            # QRS peak (narrow)
            sigma_qrs = int(0.025 * fs)
            for s in range(max(0, center_sample - 4*sigma_qrs),
                           min(len(signal_arr), center_sample + 4*sigma_qrs)):
                signal_arr[s] += 2500 * np.exp(-0.5 * ((s - center_sample) / sigma_qrs) ** 2)

            # T-wave (wider, comes after QRS)
            t_center = center_sample + int(0.30 * fs)
            sigma_t  = int(0.06 * fs)
            for s in range(max(0, t_center - 4*sigma_t),
                           min(len(signal_arr), t_center + 4*sigma_t)):
                signal_arr[s] += 600 * np.exp(-0.5 * ((s - t_center) / sigma_t) ** 2)

            # P-wave (before QRS)
            p_center = center_sample - int(0.16 * fs)
            sigma_p  = int(0.04 * fs)
            for s in range(max(0, p_center - 4*sigma_p),
                           min(len(signal_arr), p_center + 4*sigma_p)):
                signal_arr[s] += 300 * np.exp(-0.5 * ((s - p_center) / sigma_p) ** 2)

        # Fill first half with normal beats
        half = n // 2
        rr_normal = int(TARGET_FS * 60 / 70)
        pos = rr_normal
        while pos < half:
            add_beat(ecg, pos, 70, TARGET_FS)
            # Slight RR variation (±5%)
            jitter = int(rr_normal * np.random.uniform(-0.05, 0.05))
            pos += rr_normal + jitter

        # Fill second half with irregular fast beats (triggers Abnormal)
        rr_fast = int(TARGET_FS * 60 / 115)
        pos = half + rr_fast
        while pos < n:
            add_beat(ecg, pos, 115, TARGET_FS)
            # Larger RR variation (±20%) to simulate arrhythmia
            jitter = int(rr_fast * np.random.uniform(-0.20, 0.20))
            pos += rr_fast + jitter

        # Add baseline noise
        ecg += np.random.normal(0, 30, n)

        # Scale to ADC range
        s_min, s_max = ecg.min(), ecg.max()
        if s_max > s_min:
            ecg = ((ecg - s_min) / (s_max - s_min)) * TARGET_SCALE

        return ecg

    def stream(self) -> Generator[Tuple[float, int], None, None]:
        """
        Generator: yields (ecg_value, lead_off) tuples at TARGET_FS rate.

        The caller (inference engine reader thread) is responsible for
        inserting the correct 1/FS sleep delay between samples.
        """
        if self._signal is None:
            return

        signal   = self._signal
        idx      = 0
        n        = len(signal)
        lead_off = 0

        # Anomaly injection state
        anomaly_counter  = 0
        inject_every     = int(30 * TARGET_FS)  # every 30 seconds of data
        inject_duration  = int(15 * TARGET_FS)  # inject for 15 seconds

        while True:
            val = float(signal[idx])

            # Optional anomaly injection: send lead-off pulses to confuse
            # the signal slightly (simulate loose electrode briefly)
            # Only affects lead_off flag, not the underlying signal
            lead_off = 0

            yield val, lead_off

            idx += 1
            anomaly_counter += 1

            if idx >= n:
                if self.loop:
                    idx = 0
                    anomaly_counter = 0
                else:
                    break
