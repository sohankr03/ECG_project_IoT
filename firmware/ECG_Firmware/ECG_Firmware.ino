/*
 * ECG_Firmware.ino
 * ─────────────────────────────────────────────────────────────────────────
 * AI-Based Real-Time ECG Anomaly Detection System
 * ESP32 + AD8232
 *
 * Pin Mapping:
 *   AD8232 OUTPUT → GPIO34  (ADC input-only, 12-bit)
 *   AD8232 LO+    → GPIO32  (Lead-off detection)
 *   AD8232 LO-    → GPIO33  (Lead-off detection)
 *   Buzzer        → GPIO25  (Alert output)
 *   AD8232 VCC    → 3.3V
 *   AD8232 GND    → GND
 *
 * Serial Output Format (115200 baud):
 *   <millis>,<ecg_value>,<lead_off>\n
 *   e.g.  12345,2048,0
 *
 * Serial Commands from Laptop:
 *   BUZZ_ON   → activate buzzer
 *   BUZZ_OFF  → deactivate buzzer
 *
 * Sampling Rate: 250 Hz (4 ms interval)
 * ─────────────────────────────────────────────────────────────────────────
 */

// ── Pin Definitions ──────────────────────────────────────────────────────
#define ECG_PIN   34   // ADC input from AD8232 OUTPUT
#define LO_PLUS   32   // Lead-off detection positive
#define LO_MINUS  33   // Lead-off detection negative
#define BUZZER    25   // Buzzer output

// ── Sampling Configuration ───────────────────────────────────────────────
#define SAMPLE_RATE_HZ    250
#define SAMPLE_INTERVAL_US (1000000 / SAMPLE_RATE_HZ)  // 4000 µs

// ── Moving Average Window ────────────────────────────────────────────────
#define MA_WINDOW 3  // 3-sample moving average to reduce quantization noise

// ── Timer Handle ─────────────────────────────────────────────────────────
hw_timer_t *timer = NULL;
portMUX_TYPE timerMux = portMUX_INITIALIZER_UNLOCKED;

// ── Shared State (set in ISR, read in loop) ──────────────────────────────
volatile bool sampleReady = false;

// ── Moving Average Buffer ─────────────────────────────────────────────────
int maBuffer[MA_WINDOW] = {0};
int maIndex = 0;

// ── Serial Input Buffer ───────────────────────────────────────────────────
String serialCmd = "";

// ══════════════════════════════════════════════════════════════════════════
// Timer ISR — fires every 4 ms (250 Hz)
// Only set flag; do NOT do Serial.print inside ISR
// ══════════════════════════════════════════════════════════════════════════
void IRAM_ATTR onTimer() {
  portENTER_CRITICAL_ISR(&timerMux);
  sampleReady = true;
  portEXIT_CRITICAL_ISR(&timerMux);
}

// ══════════════════════════════════════════════════════════════════════════
// Moving Average — returns smoothed ADC value
// ══════════════════════════════════════════════════════════════════════════
int movingAverage(int newSample) {
  maBuffer[maIndex] = newSample;
  maIndex = (maIndex + 1) % MA_WINDOW;
  long sum = 0;
  for (int i = 0; i < MA_WINDOW; i++) sum += maBuffer[i];
  return (int)(sum / MA_WINDOW);
}

// ══════════════════════════════════════════════════════════════════════════
// Setup
// ══════════════════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);

  // ── ADC Configuration ─────────────────────────────────────────────────
  // 12-bit resolution → values 0–4095
  analogReadResolution(12);
  // ADC_ATTENDB_MAX: maximum attenuation (11 dB) = input range 0–3.3V
  // This is the correct constant name in ESP32 Arduino core v3.2.0
  analogSetAttenuation(ADC_ATTENDB_MAX);

  // ── Pin Modes ─────────────────────────────────────────────────────────
  pinMode(LO_PLUS,  INPUT);
  pinMode(LO_MINUS, INPUT);
  pinMode(BUZZER,   OUTPUT);
  digitalWrite(BUZZER, LOW);

  // ── Hardware Timer — ESP32 Arduino Core v3.x API ──────────────────────
  // timerBegin(freq): freq = base clock in Hz. 1000000 = 1 MHz → 1 tick = 1 µs
  timer = timerBegin(1000000);
  // timerAttachInterrupt: no 3rd edge argument in v3.x
  timerAttachInterrupt(timer, &onTimer);
  // timerAlarm: replaces timerAlarmWrite + timerAlarmEnable
  //   arg2 = alarm value in ticks (4000 ticks @ 1MHz = 4000 µs = 250 Hz)
  //   arg3 = auto-reload (true = repeat)
  //   arg4 = 0 (reload count, 0 = infinite)
  timerAlarm(timer, SAMPLE_INTERVAL_US, true, 0);

  // Warm-up delay to let AD8232 stabilise
  delay(500);
}

// ══════════════════════════════════════════════════════════════════════════
// Main Loop
// ══════════════════════════════════════════════════════════════════════════
void loop() {

  // ── 1. Detect timer flag ──────────────────────────────────────────────
  bool doSample = false;
  portENTER_CRITICAL(&timerMux);
  if (sampleReady) {
    sampleReady = false;
    doSample = true;
  }
  portEXIT_CRITICAL(&timerMux);

  // ── 2. Read and transmit ECG sample ──────────────────────────────────
  if (doSample) {
    // Lead-off detection: if either lead is disconnected → flag = 1
    int leadOff = (digitalRead(LO_PLUS) == HIGH || digitalRead(LO_MINUS) == HIGH) ? 1 : 0;

    // Read raw ADC (0–4095)
    int rawAdc = analogRead(ECG_PIN);

    // Apply 3-sample moving average
    int smoothed = movingAverage(rawAdc);

    // Send CSV line: timestamp(ms), ecg_value, lead_off
    Serial.print(millis());
    Serial.print(',');
    Serial.print(smoothed);
    Serial.print(',');
    Serial.println(leadOff);
  }

  // ── 3. Listen for commands from laptop ───────────────────────────────
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      serialCmd.trim();
      if (serialCmd == "BUZZ_ON") {
        digitalWrite(BUZZER, HIGH);
      } else if (serialCmd == "BUZZ_OFF") {
        digitalWrite(BUZZER, LOW);
      }
      serialCmd = "";
    } else {
      serialCmd += c;
    }
  }
}
