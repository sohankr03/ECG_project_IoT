"""
realtime_inference.py
─────────────────────────────────────────────────────────────────────────────
Real-Time ECG Inference Engine for the ECG Anomaly Detection System.

Architecture:
  - Dedicated serial reader THREAD + queue.Queue buffer
    → Streamlit (main thread) never blocks serial reading
  - Sliding window: 1250 samples (5 sec at 250 Hz), 50% overlap
  - StandardScaler applied before prediction (from scaler_v1.pkl)
  - Probability threshold: only Abnormal if predict_proba > 0.70
  - Alert: 3 consecutive abnormal windows → BUZZ_ON sent to ESP32
  - Lead-off guard: pause inference if electrode disconnected
  - SQI check: skip inference on poor signal quality
  - Session CSV log: logs/session_<timestamp>.csv

Public API (ECGInferenceEngine):
  .start()                 — start serial + inference threads
  .stop()                  — graceful shutdown
  .get_ecg_buffer()        — last 5s of ECG (display-downsampled to ~100 Hz)
  .get_latest_prediction() — dict: {label, probability, consecutive_count, timestamp}
  .get_latest_features()   — dict of 6 HRV features + sqi
  .get_status()            — "OK" | "ELECTRODE_DISCONNECTED" | "POOR_SIGNAL"
                              | "CALIBRATING" | "STARTING" | "NO_DATA"
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import time
import threading
import queue
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from collections import deque

import numpy as np
import pandas as pd
import serial
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT_DIR  = Path(__file__).resolve().parent.parent
SRC_DIR   = ROOT_DIR / "src"
MODEL_DIR = ROOT_DIR / "model"
LOGS_DIR  = ROOT_DIR / "logs"
sys.path.insert(0, str(SRC_DIR))
LOGS_DIR.mkdir(parents=True, exist_ok=True)

from feature_extraction import extract_features, features_to_vector, FEATURE_COLUMNS
from signal_processing import bandpass_filter

# ── Logging Setup ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,   # ← DEBUG to see lead-off raw lines; change to INFO when done
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ECGInference")

# ── Constants ─────────────────────────────────────────────────────────────
FS                   = 250          # Sampling rate (Hz)
WINDOW_SECONDS       = 5.0         # Inference window duration
WINDOW_SAMPLES       = int(WINDOW_SECONDS * FS)   # 1250 samples
OVERLAP_RATIO        = 0.50
STEP_SAMPLES         = int(WINDOW_SAMPLES * (1 - OVERLAP_RATIO))  # 625 samples

DISPLAY_DOWNSAMPLE   = 3           # Show every 3rd sample → ~83 Hz display
PROB_THRESHOLD       = 0.70        # Classify Abnormal only if P(abnormal) > 0.70
ALERT_CONSECUTIVE    = 3           # Fire buzzer after N consecutive abnormal windows
CALIBRATION_SECONDS  = 30         # Baseline calibration duration
LEAD_OFF_DEBOUNCE    = 5           # N consecutive lead-off samples before declaring disconnected
SERIAL_BAUD          = 115200
SERIAL_TIMEOUT       = 2.0
QUEUE_MAXSIZE        = 5000        # Max samples buffered before dropping


# ══════════════════════════════════════════════════════════════════════════
# ECG Inference Engine
# ══════════════════════════════════════════════════════════════════════════

class ECGInferenceEngine:
    """
    Thread-safe real-time ECG inference engine.

    Thread model:
      - _serial_reader_thread : reads serial port → raw_queue
      - _inference_thread     : consumes raw_queue → runs inference
    """

    def __init__(self,
                 port: str = "COM3",
                 baud: int = SERIAL_BAUD,
                 demo_mode: bool = False,
                 demo_stream=None):
        """
        Parameters
        ----------
        port        : Serial port (e.g. "COM3", "/dev/ttyUSB0")
        baud        : Baud rate (default 115200)
        demo_mode   : If True, use demo_stream instead of serial port
        demo_stream : EcgSimulator instance (provides same interface as serial)
        """
        self.port        = port
        self.baud        = baud
        self.demo_mode   = demo_mode
        self.demo_stream = demo_stream

        # ── Load ML Model & Scaler ────────────────────────────────────
        model_path  = MODEL_DIR / "ecg_rf_model_v1.pkl"
        scaler_path = MODEL_DIR / "scaler_v1.pkl"

        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(
                f"Model or scaler not found in {MODEL_DIR}.\n"
                "Run model/data_preparation.py then model/model_training.py first."
            )

        self._clf    = joblib.load(model_path)
        self._scaler = joblib.load(scaler_path)
        log.info("Model and scaler loaded.")

        # ── Shared State (protected by _lock) ─────────────────────────
        self._lock = threading.Lock()

        self._status           : str  = "STARTING"
        self._lead_off         : bool = False
        self._consecutive      : int  = 0
        self._lead_off_count   : int  = 0   # debounce counter for lead-off

        # ECG ring buffer for display (holds ~10 sec of display-rate data)
        self._ecg_display_buf : deque = deque(maxlen=int(FS * 10 / DISPLAY_DOWNSAMPLE))
        # Full-rate buffer for inference windowing
        self._ecg_full_buf    : deque = deque(maxlen=WINDOW_SAMPLES * 3)
        self._full_buf_count  : int   = 0   # total samples seen

        self._latest_prediction : Dict = {
            "label"            : "—",
            "probability"      : 0.0,
            "consecutive_count": 0,
            "timestamp"        : "—",
        }
        self._latest_features : Dict = {}

        # ── Raw sample queue (between reader ↔ inference thread) ──────
        self._raw_queue : queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

        # ── Calibration ───────────────────────────────────────────────
        self._calibrating        : bool  = False
        self._calibration_buf    : list  = []
        self._calibration_rr_mean: float = 0.0

        # ── Session CSV Log ───────────────────────────────────────────
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOGS_DIR / f"session_{ts}.csv"
        self._log_file   = open(log_path, "w", buffering=1)  # line-buffered
        header = "timestamp," + ",".join(FEATURE_COLUMNS) + ",prediction,probability,consecutive\n"
        self._log_file.write(header)
        log.info(f"Session log: {log_path}")

        # ── Control events ────────────────────────────────────────────
        self._stop_event = threading.Event()
        self._serial_obj : Optional[serial.Serial] = None

        # ── Threads ───────────────────────────────────────────────────
        self._reader_thread   = threading.Thread(target=self._serial_reader_loop,
                                                  name="SerialReader", daemon=True)
        self._inference_thread = threading.Thread(target=self._inference_loop,
                                                   name="InferenceWorker", daemon=True)

    # ── Public API ─────────────────────────────────────────────────────

    def start(self):
        """Start reader and inference threads."""
        log.info("Starting ECG inference engine...")
        self._reader_thread.start()
        self._inference_thread.start()

    def stop(self):
        """Gracefully stop all threads and close resources."""
        log.info("Stopping ECG inference engine...")
        self._stop_event.set()
        self._reader_thread.join(timeout=5)
        self._inference_thread.join(timeout=5)
        if self._serial_obj and self._serial_obj.is_open:
            self._serial_obj.close()
        self._log_file.close()
        log.info("Engine stopped.")

    def start_calibration(self):
        """Begin 30-second baseline calibration."""
        with self._lock:
            self._calibrating     = True
            self._calibration_buf = []
            self._status          = "CALIBRATING"
        log.info(f"Calibration started ({CALIBRATION_SECONDS}s)...")

    def get_ecg_buffer(self) -> List[float]:
        """Return last ~5s of ECG samples at display rate (~83 Hz)."""
        with self._lock:
            return list(self._ecg_display_buf)

    def get_latest_prediction(self) -> Dict:
        with self._lock:
            return dict(self._latest_prediction)

    def get_latest_features(self) -> Dict:
        with self._lock:
            return dict(self._latest_features)

    def get_status(self) -> str:
        with self._lock:
            return self._status

    # ── Serial Reader Thread ─────────────────────────────────────────────

    def _serial_reader_loop(self):
        """
        Continuously read lines from serial port.
        Puts parsed (ecg_value, lead_off) tuples into raw_queue.
        Never blocks the inference or main thread.
        """
        if self.demo_mode:
            self._demo_reader_loop()
            return

        while not self._stop_event.is_set():
            try:
                self._serial_obj = serial.Serial(
                    self.port, self.baud, timeout=SERIAL_TIMEOUT
                )
                log.info(f"Serial port {self.port} opened.")
                with self._lock:
                    self._status = "OK"

                while not self._stop_event.is_set():
                    try:
                        line = self._serial_obj.readline().decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue
                        parts = line.split(",")
                        if len(parts) != 3:
                            continue
                        _ts, ecg_val_str, lead_off_str = parts
                        ecg_val  = float(ecg_val_str.strip())
                        # Clamp lead_off to 0 or 1 — guards against stray chars
                        lead_off = 1 if lead_off_str.strip() not in ("0", "0.0") else 0
                        if lead_off:
                            log.debug(f"Lead-off flag received. Raw line: {repr(line)}")
                        if not self._raw_queue.full():
                            self._raw_queue.put((ecg_val, lead_off))
                    except (ValueError, UnicodeDecodeError):
                        continue
                    except serial.SerialException:
                        log.warning("Serial read error. Reconnecting...")
                        break

            except serial.SerialException as e:
                log.warning(f"Cannot open {self.port}: {e}. Retrying in 3s...")
                with self._lock:
                    self._status = "NO_DATA"
                time.sleep(3)

    def _demo_reader_loop(self):
        """Feed data from EcgSimulator into raw_queue at 250 Hz."""
        with self._lock:
            self._status = "OK"
        try:
            for ecg_val, lead_off in self.demo_stream.stream():
                if self._stop_event.is_set():
                    break
                if not self._raw_queue.full():
                    self._raw_queue.put((float(ecg_val), int(lead_off)))
                time.sleep(1.0 / FS)
        except Exception as e:
            log.error(f"Demo stream error: {e}")

    # ── Inference Thread ───────────────────────────────────────────────────

    def _inference_loop(self):
        """
        Consumes samples from raw_queue.
        Runs inference every STEP_SAMPLES new samples.
        """
        pending = 0  # samples since last inference run

        while not self._stop_event.is_set() or not self._raw_queue.empty():
            try:
                ecg_val, is_lead_off = self._raw_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # ── Lead-off guard (with debounce) ─────────────────────────
            with self._lock:
                self._lead_off = bool(is_lead_off)
                if is_lead_off:
                    self._lead_off_count += 1
                    if self._lead_off_count >= LEAD_OFF_DEBOUNCE:
                        self._status = "ELECTRODE_DISCONNECTED"
                        self._send_serial_command("BUZZ_OFF")
                    continue  # skip this sample regardless
                else:
                    if self._lead_off_count > 0:
                        log.debug(f"Lead-off cleared after {self._lead_off_count} samples.")
                    self._lead_off_count = 0
                    if self._status == "ELECTRODE_DISCONNECTED":
                        self._status = "OK"

            # ── Calibration accumulation ───────────────────────────────
            with self._lock:
                if self._calibrating:
                    self._calibration_buf.append(ecg_val)
                    if len(self._calibration_buf) >= CALIBRATION_SECONDS * FS:
                        self._finish_calibration()
                    continue

            # ── Update ECG buffers ─────────────────────────────────────
            with self._lock:
                self._ecg_full_buf.append(ecg_val)
                self._full_buf_count += 1
                # Downsample for display
                if self._full_buf_count % DISPLAY_DOWNSAMPLE == 0:
                    self._ecg_display_buf.append(ecg_val)

            pending += 1

            # ── Run inference every STEP_SAMPLES ──────────────────────
            if pending >= STEP_SAMPLES:
                pending = 0
                with self._lock:
                    buf_snapshot = np.array(self._ecg_full_buf)

                if len(buf_snapshot) < WINDOW_SAMPLES:
                    continue  # not enough data yet

                window = buf_snapshot[-WINDOW_SAMPLES:]
                self._run_inference(window)

    def _run_inference(self, window: np.ndarray):
        """Extract features + classify one 5-second ECG window."""

        # ── Feature extraction ─────────────────────────────────────────
        features = extract_features(window, fs=FS)

        with self._lock:
            self._latest_features = features

        if not features.get("quality_ok", False):
            with self._lock:
                self._status = "POOR_SIGNAL"
                self._latest_prediction = {
                    "label"            : "Poor Signal",
                    "probability"      : 0.0,
                    "consecutive_count": 0,
                    "timestamp"        : datetime.now().strftime("%H:%M:%S"),
                }
            return

        with self._lock:
            self._status = "OK"

        # ── Scale features ─────────────────────────────────────────────
        feat_vec = features_to_vector(features)
        if feat_vec is None:
            return

        feat_scaled = self._scaler.transform(feat_vec.reshape(1, -1))

        # ── Classify with probability threshold ───────────────────────
        prob_abnormal = float(self._clf.predict_proba(feat_scaled)[0][1])
        is_abnormal   = prob_abnormal > PROB_THRESHOLD
        label         = "ABNORMAL" if is_abnormal else "Normal"

        # ── Consecutive window counter ─────────────────────────────────
        with self._lock:
            if is_abnormal:
                self._consecutive += 1
            else:
                self._consecutive = 0
            consecutive = self._consecutive

        # ── Buzzer alert: 3 consecutive abnormal windows ───────────────
        if consecutive >= ALERT_CONSECUTIVE:
            self._send_serial_command("BUZZ_ON")
            log.warning(f"ALERT: {consecutive} consecutive ABNORMAL windows!")
        else:
            self._send_serial_command("BUZZ_OFF")

        ts_str = datetime.now().strftime("%H:%M:%S")

        # ── Update latest prediction ───────────────────────────────────
        with self._lock:
            self._latest_prediction = {
                "label"            : label,
                "probability"      : round(prob_abnormal, 3),
                "consecutive_count": consecutive,
                "timestamp"        : ts_str,
            }

        # ── Log to CSV ────────────────────────────────────────────────
        feature_vals = ",".join(str(features.get(c, 0)) for c in FEATURE_COLUMNS)
        self._log_file.write(
            f"{ts_str},{feature_vals},{label},{prob_abnormal:.3f},{consecutive}\n"
        )

    # ── Calibration ───────────────────────────────────────────────────────

    def _finish_calibration(self):
        """Compute baseline RR mean from calibration buffer."""
        buf = np.array(self._calibration_buf)
        features = extract_features(buf, fs=FS, sqi_threshold=0.3)
        if features.get("quality_ok", False):
            self._calibration_rr_mean = features["rr_mean"]
            log.info(f"Calibration done. Baseline RR mean: {self._calibration_rr_mean:.1f} ms")
        else:
            log.warning("Calibration signal quality too poor. Using defaults.")
        self._calibrating = False
        self._status = "OK"

    # ── Serial Command ────────────────────────────────────────────────────

    def _send_serial_command(self, cmd: str):
        """Send a command string to ESP32 (best-effort)."""
        if self.demo_mode:
            return  # no hardware in demo mode
        try:
            if self._serial_obj and self._serial_obj.is_open:
                self._serial_obj.write((cmd + "\n").encode("utf-8"))
        except Exception:
            pass
