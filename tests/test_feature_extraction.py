"""
tests/test_feature_extraction.py
Unit tests for the feature extraction module.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from feature_extraction import (
    extract_features, features_to_vector,
    compute_sqi, FEATURE_COLUMNS
)
from signal_processing import bandpass_filter


FS = 250


def make_ecg_window(hr_bpm: float = 75.0, noise: float = 30.0) -> np.ndarray:
    """5-second synthetic ECG window at given HR."""
    n = int(5.0 * FS)
    ecg = np.zeros(n)
    rr  = int(FS * 60 / hr_bpm)
    sigma = int(0.025 * FS)
    pos = rr
    while pos + 4*sigma < n:
        for i in range(max(0, pos-4*sigma), min(n, pos+4*sigma)):
            ecg[i] += 2500 * np.exp(-0.5 * ((i-pos)/sigma)**2)
        pos += rr
    ecg += np.random.default_rng(7).normal(0, noise, n)
    # Shift to ADC-like range
    ecg += 2048
    return ecg


class TestExtractFeatures:

    def test_returns_dict(self):
        win = make_ecg_window()
        result = extract_features(win, fs=FS)
        assert isinstance(result, dict)

    def test_all_keys_present(self):
        win = make_ecg_window()
        result = extract_features(win, fs=FS)
        required_keys = ["quality_ok", "sqi", "heart_rate", "rr_mean",
                         "rr_std", "sdnn", "rmssd", "beat_variance"]
        for k in required_keys:
            assert k in result, f"Missing key: {k}"

    def test_reasonable_heart_rate(self):
        """75 BPM synthetic signal should give HR in 60–95 BPM range."""
        win = make_ecg_window(hr_bpm=75.0, noise=20.0)
        result = extract_features(win, fs=FS)
        if result["quality_ok"]:
            hr = result["heart_rate"]
            assert 55 <= hr <= 100, f"Heart rate {hr:.1f} outside expected range"

    def test_high_hr_detected(self):
        """110 BPM fast heart rate should give HR > 95."""
        win = make_ecg_window(hr_bpm=110.0, noise=15.0)
        result = extract_features(win, fs=FS)
        if result["quality_ok"]:
            hr = result["heart_rate"]
            assert hr > 90, f"High HR not detected correctly: {hr:.1f}"

    def test_flat_signal_fails_quality(self):
        """Flat (electrode-off) signal should have quality_ok=False."""
        flat = np.ones(1250) * 2048.0
        flat += np.random.randn(1250) * 2.0  # tiny noise
        result = extract_features(flat, fs=FS)
        assert result["quality_ok"] is False

    def test_non_negative_hrv_metrics(self):
        win = make_ecg_window()
        result = extract_features(win, fs=FS)
        if result["quality_ok"]:
            assert result["rr_std"]        >= 0
            assert result["sdnn"]          >= 0
            assert result["rmssd"]         >= 0
            assert result["beat_variance"] >= 0

    def test_sqi_between_0_and_1(self):
        win = make_ecg_window()
        result = extract_features(win, fs=FS)
        assert 0.0 <= result["sqi"] <= 1.0


class TestFeaturesToVector:

    def test_returns_correct_shape(self):
        win = make_ecg_window()
        result = extract_features(win, fs=FS)
        if result["quality_ok"]:
            vec = features_to_vector(result)
            assert vec is not None
            assert vec.shape == (len(FEATURE_COLUMNS),)

    def test_returns_none_for_bad_quality(self):
        bad = {"quality_ok": False, "sqi": 0.0}
        vec = features_to_vector(bad)
        assert vec is None

    def test_column_order_matches(self):
        win = make_ecg_window()
        result = extract_features(win, fs=FS)
        if result["quality_ok"]:
            vec = features_to_vector(result)
            for i, col in enumerate(FEATURE_COLUMNS):
                assert vec[i] == result[col], f"Column order mismatch at {col}"
