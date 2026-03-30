"""
server.py
─────────────────────────────────────────────────────────────────────────────
Flask + Socket.IO backend for the ECG Anomaly Detection Dashboard.

Replaces Streamlit with a proper real-time web server.
- Serves the HTML dashboard from dashboard/index.html
- Pushes ECG data to browser every 200ms via WebSocket (Socket.IO)
- REST API endpoints for start / stop / calibrate

Run:
    cd E:\\Project\\ECG_Project
    python server.py

Then open: http://localhost:5000
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import threading
import time
import logging
from pathlib import Path

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

# ── Path Setup ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR  = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from realtime_inference import ECGInferenceEngine

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("ECGServer")

# ── Flask + Socket.IO Setup ───────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=str(ROOT_DIR / "dashboard"),
    static_folder=str(ROOT_DIR / "dashboard"),
)
app.config["SECRET_KEY"] = "ecg_secret_2026"

# Use threading mode (works on Windows without gevent install issues)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                    logger=False, engineio_logger=False)

# ── Global State ──────────────────────────────────────────────────────────
engine        = None
push_thread   = None
push_running  = False
engine_lock   = threading.Lock()

PUSH_INTERVAL = 0.20   # seconds — push to browser every 200ms
ECG_DISPLAY_POINTS = 500  # last N samples sent to browser (~5s at ~100Hz display)


# ══════════════════════════════════════════════════════════════════════════
# Background push loop — runs in its own thread
# ══════════════════════════════════════════════════════════════════════════

def push_data_loop():
    """Push ECG state to all connected browsers every 200ms."""
    global push_running
    while push_running:
        try:
            with engine_lock:
                eng = engine

            if eng is not None:
                ecg_buf  = eng.get_ecg_buffer()
                pred     = eng.get_latest_prediction()
                features = eng.get_latest_features()
                status   = eng.get_status()

                # Send last ECG_DISPLAY_POINTS samples (prevents huge payloads)
                ecg_slice = list(ecg_buf)[-ECG_DISPLAY_POINTS:]

                socketio.emit("update", {
                    "ecg"       : ecg_slice,
                    "prediction": pred,
                    "features"  : {k: round(float(v), 2) if isinstance(v, (int, float)) else v
                                   for k, v in features.items()},
                    "status"    : status,
                })

        except Exception as e:
            log.warning(f"Push error: {e}")

        time.sleep(PUSH_INTERVAL)


# ══════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


# ── API: Start Engine ─────────────────────────────────────────────────────
@app.route("/api/start", methods=["POST"])
def api_start():
    global engine, push_thread, push_running

    data      = request.get_json() or {}
    demo_mode = data.get("demo_mode", True)
    port      = data.get("port", "COM3")

    # Stop existing engine first
    with engine_lock:
        eng = engine
    if eng is not None:
        eng.stop()

    try:
        if demo_mode:
            from ecg_simulator import EcgSimulator
            sim = EcgSimulator(record_name="119", inject_anomaly=True, loop=True)
            new_engine = ECGInferenceEngine(demo_mode=True, demo_stream=sim)
            log.info("Starting in DEMO mode.")
        else:
            new_engine = ECGInferenceEngine(port=port, demo_mode=False)
            log.info(f"Starting with hardware on {port}.")

        new_engine.start()

        with engine_lock:
            engine = new_engine

        # Start push thread if not already running
        if not push_running:
            push_running = True
            push_thread  = threading.Thread(target=push_data_loop,
                                             name="PushThread", daemon=True)
            push_thread.start()

        return jsonify({"ok": True, "mode": "demo" if demo_mode else "hardware", "port": port})

    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ── API: Stop Engine ──────────────────────────────────────────────────────
@app.route("/api/stop", methods=["POST"])
def api_stop():
    global engine, push_running

    push_running = False

    with engine_lock:
        eng    = engine
        engine = None

    if eng is not None:
        eng.stop()
        log.info("Engine stopped.")

    return jsonify({"ok": True})


# ── API: Calibrate ────────────────────────────────────────────────────────
@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    with engine_lock:
        eng = engine
    if eng is None:
        return jsonify({"ok": False, "error": "Engine not running"}), 400
    eng.start_calibration()
    log.info("Calibration started.")
    return jsonify({"ok": True})


# ── API: Status ───────────────────────────────────────────────────────────
@app.route("/api/status")
def api_status():
    with engine_lock:
        eng = engine
    running = eng is not None
    status  = eng.get_status() if eng else "STOPPED"
    return jsonify({"running": running, "status": status})


# ══════════════════════════════════════════════════════════════════════════
# Socket.IO Events
# ══════════════════════════════════════════════════════════════════════════

@socketio.on("connect")
def on_connect():
    log.info(f"Browser connected.")
    emit("connected", {"msg": "ECG Server connected"})


@socketio.on("disconnect")
def on_disconnect():
    log.info("Browser disconnected.")


# ══════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  ECG Anomaly Detection — Web Server")
    print("  Open: http://localhost:5000")
    print("=" * 55)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)
