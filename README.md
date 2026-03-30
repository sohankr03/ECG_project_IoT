# AI-Based Real-Time ECG Anomaly Detection System

> **Final Year Project** — ESP32 + AD8232 + Random Forest + Real-Time Web Dashboard
Slide :https://docs.google.com/presentation/d/1TVrM1xiIEPbAa-GOkwLAtFbYfK3NQyFA0Z4PHhXpLKc/edit?usp=sharing
---

## 🔬 System Overview

```
ECG Electrodes
  → AD8232 (Signal Conditioning)
    → ESP32 GPIO34 (12-bit ADC at 250 Hz)
      → Serial USB (115200 baud)
        → Python Pipeline (Filter → Features → Random Forest)
          → Flask + Socket.IO Web Dashboard + Buzzer Alert
```

---

## 📁 Project Structure

```
ECG_Project/
├── firmware/ECG_Firmware/ECG_Firmware.ino   # ESP32 Arduino sketch (Core v3.x ready)
├── src/
│   ├── signal_processing.py                 # Butterworth filter + Pan-Tompkins
│   ├── feature_extraction.py                # 6 HRV features + SQI
│   ├── realtime_inference.py                # Threaded inference engine
│   └── ecg_simulator.py                     # MIT-BIH Demo streamer
├── model/
│   ├── data_preparation.py                  # MIT-BIH → feature CSV
│   ├── model_training.py                    # Train Random Forest
│   └── model_evaluation.py                  # Evaluation + plots
├── dashboard/
│   └── index.html                           # Custom HTML/CSS/JS frontend
├── server.py                                # Flask + Socket.IO backend
├── logs/                                    # Session CSV logs
├── assets/                                  # Plots, confusion matrix
├── tests/                                   # Unit tests
├── requirements.txt
└── conftest.py
```

---

## ⚡ Hardware Wiring

| AD8232 Pin | ESP32 Pin | Notes |
|---|---|---|
| **3.3V** | **3V3** | Power |
| **GND** | **GND** | Ground |
| **OUTPUT** | **GPIO34** | ADC input-only, 12-bit |
| **LO+** | **GPIO32** | Lead-off detection |
| **LO-** | **GPIO33** | Lead-off detection |
| **SDN** | — | **Not connected** (Optional) |

**Buzzer (5V Active Type):**
Requires a transistor (BC547/2N2222) switching circuit driven by **GPIO25**.

---

## 🛠 Setup & Installation

### 1. Install Python dependencies
```bash
cd E:\Project\ECG_Project
pip install -r requirements.txt
```

### 2. Flash ESP32 firmware
- Open `firmware/ECG_Firmware/ECG_Firmware.ino` in Arduino IDE
- Install **ESP32 board package** (v3.2.0 or later)
- Select board: `ESP32 Dev Module`
- Set baud: `115200`
- Upload

### 3. Train the AI model (one-time setup)
Downloads the MIT-BIH dataset from PhysioNet and trains the Random Forest classifier.
```bash
cd E:\Project\ECG_Project\model
python data_preparation.py
python model_training.py
python model_evaluation.py
```

---

## 🖥 Running the Dashboard

We use a custom, high-performance web dashboard over WebSockets (no page reloads).

### 1. Start the Server
```bash
cd E:\Project\ECG_Project
python server.py
```

### 2. Open the Interface
Open your web browser and navigate to:
**http://localhost:5000**

### 3. Controls
| Feature | Location |
|---|---|
| Demo Mode (no hardware) | Sidebar toggle — **enabled by default** |
| Live hardware mode | Disable Demo Mode → enter COM port (e.g., `COM3`) |
| Start/Stop engine | Sidebar buttons |
| Calibration (30s baseline) | Sidebar → Calibrate button (Personalizes alerts) |

---

## 🧠 AI Model Specifications

| Parameter | Value |
|---|---|
| Algorithm | Random Forest (scikit-learn) |
| Trees | 100 |
| Features | 6 HRV features |
| Training data | MIT-BIH Arrhythmia Dataset (15 records) |
| Labels | Normal (0) / Abnormal (1) |
| Alert threshold | P(abnormal) > **0.70** |
| Alert trigger | **3 consecutive** abnormal windows |

---

## 🔬 Signal Processing Pipeline

| Stage | Method | Parameters |
|---|---|---|
| Bandpass filter | Butterworth (4th order) | 0.5–40 Hz |
| QRS detection | Pan-Tompkins | Refractory: 200 ms |
| Feature extraction | HRV time-domain | 5-second windows, 50% overlap |
| Normalisation | StandardScaler | Fitted on MIT-BIH training data |

---

## 📊 Features Extracted

1. `heart_rate` — BPM
2. `rr_mean` — Mean RR interval (ms)
3. `rr_std` — RR variability (ms)
4. `sdnn` — Std deviation of NN intervals
5. `rmssd` — Root mean square of successive differences
6. `beat_variance` — Beat-to-beat interval variance

---

## 🧪 Running Unit Tests
```bash
cd E:\Project\ECG_Project
pytest tests/ -v
```

---

## ⚠️ Important Notes

- **This system detects abnormal ECG patterns only — it does NOT diagnose diseases.**
- AD8232 single-lead setup is NOT reliable for P-wave or ST-segment analysis.
- The model must always be used with its paired `scaler_v1.pkl` for normalisation.
- Buzzer activates after **3 consecutive abnormal windows** (~15 seconds) to prevent false alarms.
- If electrodes are disconnected, inference is automatically paused.

---
*Built with Python 3.11 · scikit-learn · Flask · Socket.IO · wfdb*
