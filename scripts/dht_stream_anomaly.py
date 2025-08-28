# scripts/dht_stream_anomaly.py
# Reads data from a DHT11 sensor on Raspberry Pi, detects anomalies
# using a rolling z-score method, logs readings to CSV, and controls
# optional LEDs (red = anomaly, green = normal).
#pushes readings to a Power BI Streaming Dataset (set POWER_BI_URL below)
import os
import time
import csv
from datetime import datetime
import requests
import json

import adafruit_dht
import board

# Optional LEDs
USE_LEDS = True
try:
    import RPi.GPIO as GPIO
except Exception:
    USE_LEDS = False
    GPIO = None  # avoid NameError if referenced

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "sensor_anomalies.csv")

# Sensor pin (BCM)
# Change to DHT22 if your sensor is DHT22: adafruit_dht.DHT22(board.D4)
DHT = adafruit_dht.DHT11(board.D4)

# LED pins (BCM)
RED_PIN = 17
GREEN_PIN = 27

# Initialize LEDs if available
if USE_LEDS:
    try:
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(RED_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(GREEN_PIN, GPIO.OUT, initial=GPIO.LOW)
    except Exception:
        USE_LEDS = False

# LED helper functions
def set_leds(anomaly: bool):
    if not USE_LEDS:
        return
    try:
        GPIO.output(RED_PIN, GPIO.HIGH if anomaly else GPIO.LOW)
        GPIO.output(GREEN_PIN, GPIO.LOW if anomaly else GPIO.HIGH)
    except Exception:
        pass  # keep streaming even if LED toggle fails

def safe_close():
    if USE_LEDS:
        try:
            GPIO.output(RED_PIN, GPIO.LOW)
            GPIO.output(GREEN_PIN, GPIO.LOW)
            GPIO.cleanup()
        except Exception:
            pass

# Rolling Z-score detector
class RollingZ:
    def __init__(self, window=90, z_thresh=3.5, min_std_t=0.1, min_std_h=0.2, debounce=2):
        self.window = window
        self.z_thresh = z_thresh
        self.min_std_t = min_std_t
        self.min_std_h = min_std_h
        self.debounce = debounce
        self.t_vals, self.h_vals = [], []
        self._anom_streak = 0  # require consecutive anomalies to latch

    def update_and_score(self, temp, hum):
        import statistics as stats

        # Build baseline
        if len(self.t_vals) < self.window:
            self.t_vals.append(temp)
            self.h_vals.append(hum)
            return {"status": "warming_up", "anomaly": False, "z_temp": None, "z_hum": None}

        # Means & clamped stds
        t_mean = stats.mean(self.t_vals)
        h_mean = stats.mean(self.h_vals)
        t_std = max(stats.pstdev(self.t_vals), self.min_std_t)
        h_std = max(stats.pstdev(self.h_vals), self.min_std_h)

        z_t = abs((temp - t_mean) / t_std)
        z_h = abs((hum - h_mean) / h_std)

        # Slide window
        self.t_vals.pop(0); self.t_vals.append(temp)
        self.h_vals.pop(0); self.h_vals.append(hum)

        instant_anom = (z_t >= self.z_thresh) or (z_h >= self.z_thresh)

        # Debounce logic: require N consecutive anomalies
        if instant_anom:
            self._anom_streak += 1
        else:
            self._anom_streak = 0

        anomaly = self._anom_streak >= self.debounce

        return {"status": "ready", "anomaly": anomaly, "z_temp": z_t, "z_hum": z_h}

# CSV logging
def append_log(ts, temp, hum, status, anomaly, zt, zh):
    # Write header if file is missing OR empty
    new_file = (not os.path.exists(LOG_PATH)) or os.path.getsize(LOG_PATH) == 0
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "temp_c", "humidity", "status", "anomaly", "z_temp", "z_hum"])
        w.writerow([ts, temp, hum, status, int(anomaly), zt, zh])

# Power BI push function
POWER_BI_URL = "https://api.powerbi.com/beta/63ff6df4-9ea2-4e23-b6a9-b4c24dce2bbc/datasets/93d7e14f-0cca-42b0-8b14-e34cfa52659e/rows?experience=power-bi&key=Wsb5%2BdN4GibxZ1EIrbuRoX2Tj7EDXDb5M6RmeexxELOLf%2BNCnrw5FdfVBv2OwvL3CYxV7XRFPdrX3uU8eqRs1w%3D%3D"

def push_to_powerbi(temp, hum, status_text):
    """Send one row to Power BI and print HTTP result for debugging."""
    payload = [{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "temperature": float(temp),
        "humidity": float(hum),
        "status": status_text  # must be "Normal" or "Anomaly"
    }]
    try:
        r = requests.post(
            POWER_BI_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=8
        )
        # 200 or 202 are OK. Print first 120 chars of body for clues if not.
        print(f"Power BI POST {r.status_code}: {r.text[:120]}")
    except Exception as e:
        print("❌ Power BI POST failed:", e)

# Main loop
def main():
    print("🟢 Starting DHT stream with rolling anomaly detection...")
    if not USE_LEDS:
        print("(LEDs disabled or not available)")

    detector = RollingZ(window=90, z_thresh=3.5, min_std_t=0.1, min_std_h=0.2, debounce=2)
    set_leds(False)

    try:
        while True:
            try:
                temp = DHT.temperature
                hum  = DHT.humidity

                if temp is None or hum is None:
                    print("⚠️  Read error (None). Retrying...")
                    time.sleep(2)
                    continue

                score = detector.update_and_score(temp, hum)
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if score["status"] == "warming_up":
                    print(f"{ts} | TEMP {temp:.1f}°C HUM {hum:.0f}% | Baseline warming up ({len(detector.t_vals)}/{detector.window})")
                    set_leds(False)
                else:
                    label = "🔴 Anomaly" if score["anomaly"] else "🟢 Normal"
                    print(f"{ts} | TEMP {temp:.1f}°C HUM {hum:.0f}% | {label} | zT={score['z_temp']:.2f} zH={score['z_hum']:.2f}")
                    set_leds(score["anomaly"])

                append_log(ts, temp, hum, score["status"], score["anomaly"], score["z_temp"], score["z_hum"])

                # Push to Power BI  -> send "Normal"/"Anomaly" 
                status_text = "Anomaly" if score["anomaly"] else "Normal"
                push_to_powerbi(temp, hum, status_text)

                time.sleep(2)

            except RuntimeError as e:
                # DHTs often throw intermittent CRC/timeouts – retry
                print("Read error:", e)
                time.sleep(2)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        safe_close()

if __name__ == "__main__":
    main()
