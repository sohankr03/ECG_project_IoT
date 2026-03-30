"""
tests/test_signal_processing.py
Unit tests for the signal processing module.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from signal_processing import bandpass_filter, pan_tompkins_qrs, compute_rr_intervals


FS = 250  # Hz


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def make_synthetic_ecg(duration_s: float = 10.0, hr_bpm: float = 75.0,
                        noise_std: float = 50.0) -> np.ndarray:
    """Generate a synthetic ECG with Gaussian QRS complexes at given HR."""
    n = int(duration_s * FS)
    ecg = np.zeros(n)

    rr_samples = int(FS * 60 / hr_bpm)
    qrs_sigma  = int(0.025 * FS)  # 25 ms

    pos = rr_samples
    while pos + 4 * qrs_sigma < n:
        for i in range(max(0, pos - 4*qrs_sigma), min(n, pos + 4*qrs_sigma)):
            ecg[i] += 2500 * np.exp(-0.5 * ((i - pos) / qrs_sigma) ** 2)
        pos += rr_samples

    ecg += np.random.default_rng(42).normal(0, noise_std, n)
    return ecg


# ══════════════════════════════════════════════════════════════════════════
# Bandpass Filter Tests
# ══════════════════════════════════════════════════════════════════════════

class TestBandpassFilter:

    def test_output_same_length(self):
        """Filter output must have same length as input."""
        signal = np.random.randn(1000)
        filtered = bandpass_filter(signal, fs=FS)
        assert len(filtered) == len(signal)

    def test_removes_dc_offset(self):
        """High-pass behaviour: large DC offset should be attenuated."""
        n = 1250  # 5 seconds
        signal = np.ones(n) * 2048.0  # pure DC (0 Hz)
        signal += np.random.randn(n) * 10  # small noise
        filtered = bandpass_filter(signal, fs=FS)
        # After filter, mean should be near zero
        assert abs(np.mean(filtered)) < 50.0, f"DC not removed: mean={np.mean(filtered):.2f}"

    def test_output_is_numpy_array(self):
        signal = np.random.randn(500)
        filtered = bandpass_filter(signal, fs=FS)
        assert isinstance(filtered, np.ndarray)

    def test_short_signal_does_not_crash(self):
        """Filter should handle signals shorter than typical window."""
        signal = np.random.randn(50)
        filtered = bandpass_filter(signal, fs=FS)
        assert len(filtered) == 50


# ══════════════════════════════════════════════════════════════════════════
# Pan-Tompkins QRS Detection Tests
# ══════════════════════════════════════════════════════════════════════════

class TestPanTompkins:

    def test_detects_peaks_in_clean_ecg(self):
        """Should detect approximately correct number of beats at 75 BPM."""
        ecg = make_synthetic_ecg(duration_s=10.0, hr_bpm=75.0, noise_std=20.0)
        filtered = bandpass_filter(ecg, fs=FS)
        r_peaks = pan_tompkins_qrs(filtered, fs=FS)

        # 10 seconds × 75 BPM / 60 = 12.5 beats expected
        # Allow ±3 tolerance
        expected_beats = int(10.0 * 75 / 60)
        assert len(r_peaks) >= expected_beats - 3, \
            f"Too few peaks: {len(r_peaks)} (expected ~{expected_beats})"
        assert len(r_peaks) <= expected_beats + 3, \
            f"Too many peaks: {len(r_peaks)} (expected ~{expected_beats})"

    def test_refractory_period_prevents_double_detection(self):
        """No two consecutive peaks should be closer than 200 ms."""
        ecg = make_synthetic_ecg(duration_s=15.0, hr_bpm=60.0, noise_std=80.0)
        filtered = bandpass_filter(ecg, fs=FS)
        r_peaks = pan_tompkins_qrs(filtered, fs=FS)

        if len(r_peaks) >= 2:
            diffs_ms = np.diff(r_peaks) / FS * 1000.0
            assert np.all(diffs_ms >= 200.0), \
                f"Refractory period violated: min diff = {diffs_ms.min():.1f} ms"

    def test_empty_signal_returns_empty_array(self):
        """Too-short signal should return empty array, not crash."""
        tiny = np.random.randn(10)
        r_peaks = pan_tompkins_qrs(tiny, fs=FS)
        assert isinstance(r_peaks, np.ndarray)
        assert len(r_peaks) == 0

    def test_returns_numpy_int_array(self):
        ecg = make_synthetic_ecg(duration_s=5.0)
        filtered = bandpass_filter(ecg, fs=FS)
        r_peaks = pan_tompkins_qrs(filtered, fs=FS)
        assert r_peaks.dtype == int or np.issubdtype(r_peaks.dtype, np.integer)

    def test_peaks_within_signal_bounds(self):
        """All detected peak indices must be within signal range."""
        n = 1250
        ecg = make_synthetic_ecg(duration_s=5.0)
        filtered = bandpass_filter(ecg[:n], fs=FS)
        r_peaks = pan_tompkins_qrs(filtered, fs=FS)
        assert np.all(r_peaks >= 0)
        assert np.all(r_peaks < n)


# ══════════════════════════════════════════════════════════════════════════
# RR Interval Tests
# ══════════════════════════════════════════════════════════════════════════

class TestRRIntervals:

    def test_correct_count(self):
        """RR interval array length should be len(r_peaks) - 1."""
        r_peaks = np.array([100, 350, 600, 850, 1100])
        rr = compute_rr_intervals(r_peaks, fs=FS)
        assert len(rr) == len(r_peaks) - 1

    def test_correct_values(self):
        """Check exact ms values for known peak positions."""
        # Peaks at sample 0, 250, 500 → RR = 250 samples each
        # At 250 Hz: 250/250 * 1000 = 1000 ms each
        r_peaks = np.array([0, 250, 500])
        rr = compute_rr_intervals(r_peaks, fs=FS)
        np.testing.assert_allclose(rr, [1000.0, 1000.0], rtol=1e-6)

    def test_single_peak_returns_empty(self):
        """One peak → no interval → empty array."""
        r_peaks = np.array([100])
        rr = compute_rr_intervals(r_peaks, fs=FS)
        assert len(rr) == 0

    def test_empty_peaks_returns_empty(self):
        rr = compute_rr_intervals(np.array([], dtype=int), fs=FS)
        assert len(rr) == 0

    def test_75bpm_rr_in_range(self):
        """75 BPM → RR ≈ 800 ms."""
        ecg = make_synthetic_ecg(duration_s=10.0, hr_bpm=75.0, noise_std=10.0)
        filtered = bandpass_filter(ecg, fs=FS)
        r_peaks = pan_tompkins_qrs(filtered, fs=FS)
        if len(r_peaks) >= 3:
            rr = compute_rr_intervals(r_peaks, fs=FS)
            mean_rr = np.mean(rr)
            # 75 BPM → 800 ms, allow ±15% tolerance
            assert 680 <= mean_rr <= 920, f"Mean RR {mean_rr:.1f} ms out of expected range"
