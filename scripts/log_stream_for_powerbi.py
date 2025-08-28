# scripts/log_stream_for_powerbi.py
# Stream-like logger for Power BI: loads the trained model + encoders,
# feeds rows one-by-one (balanced stream if available, else shuffled full set),
# writes timestamped predictions to ../logs/powerbi_logs.csv, and prints totals.

import os
import time
import joblib
import pandas as pd
from datetime import datetime
from sklearn.exceptions import NotFittedError
from pathlib import Path

# Paths
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

MODEL_PATH   = PROJECT / "scripts" / "models" / "kan_model.joblib"
ENCODER_PATH = PROJECT / "scripts" / "models" / "kan_encoders.joblib"

BALANCED_PATH = PROJECT / "data" / "Train_Test_Network_dataset" / "balanced_stream.csv"
FULL_PATH     = PROJECT / "data" / "Train_Test_Network_dataset" / "train_test_network.csv"

LOGS_DIR  = PROJECT / "logs"
LOG_PATH  = LOGS_DIR / "powerbi_logs.csv"

# Load model + encoders 
print("Loading model and encoders...")
try:
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
except FileNotFoundError as e:
    print(f"❌ {e}\nMake sure you’ve trained and saved the model/encoders.")
    raise SystemExit(1)

# ---------- Pick data source ----------
if BALANCED_PATH.exists():
    data_path = BALANCED_PATH
    print(f"Using balanced stream: {data_path}")
else:
    data_path = FULL_PATH
    print(f"Balanced stream not found; using full dataset (shuffled): {data_path}")

df = pd.read_csv(data_path)

# If using full dataset, shuffle so the stream isn’t all one class at the top
if data_path == FULL_PATH:
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

#Encode categoricals the same way as during training
obj_cols = df.select_dtypes(include="object").columns.tolist()
for col in obj_cols:
    if col in encoders:
        df[col] = encoders[col].transform(df[col].astype(str))
    else:
        # If an unexpected categorical column appears, drop it to keep features aligned
        df.drop(columns=[col], inplace=True)
        print(f"⚠️ Dropped unseen categorical column: {col}")

# Remove label for “live” prediction
if "label" in df.columns:
    df.drop(columns=["label"], inplace=True)

# Prepare logging 
LOGS_DIR.mkdir(parents=True, exist_ok=True)
with LOG_PATH.open("w", encoding="utf-8") as f:
    f.write("timestamp,prediction\n")

print("Logging stream started... Press CTRL+C to stop.\n")

total_normal = 0
total_anom   = 0

try:
    for i, row in df.iterrows():
        # Predict with a DataFrame to keep feature names (no sklearn warnings)
        sample = pd.DataFrame([row], columns=df.columns)

        try:
            pred = model.predict(sample)[0]
        except NotFittedError:
            print("❌ The model isn't trained yet.")
            break

        label = "Normal" if pred == 0 else "Anomaly"
        if label == "Normal":
            total_normal += 1
        else:
            total_anom += 1

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f"{ts},{label}\n")

        print(f"[{ts}] → Prediction: {label} | Totals → Normal: {total_normal}, Anomaly: {total_anom}")

        time.sleep(0.5)  # simulate streaming delay

except KeyboardInterrupt:
    print("\n✅ Stream logging ended by user.")

print(f"\n📝 Log written to: {LOG_PATH}")
