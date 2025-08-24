# scripts/log_stream_for_powerbi.py
# Simulates a streaming pipeline: loads trained model & encoders, 
# applies them to dataset rows one by one, and logs predictions 
# with timestamps to a CSV for Power BI dashboards.

import os
import time
import joblib
import pandas as pd
from datetime import datetime
from sklearn.exceptions import NotFittedError

# Paths
MODEL_PATH = "../scripts/models/kan_model.joblib"
ENCODER_PATH = "../scripts/models/kan_encoders.joblib"
DATA_PATH = "../data/Train_Test_Network_dataset/train_test_network.csv"
LOG_PATH = "../logs/powerbi_logs.csv"

print("Loading model and encoders...")
try:
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
except FileNotFoundError:
    print("❌ Model or encoder file not found. Please train the model first.")
    exit()

# Load dataset
df = pd.read_csv(DATA_PATH)

# Apply label encoders
for col in df.select_dtypes(include="object").columns:
    if col in encoders:
        df[col] = encoders[col].transform(df[col].astype(str))

# Drop target (simulate live stream input only)
if "label" in df.columns:
    df.drop(columns=["label"], inplace=True)

# Ensure logs directory exists
os.makedirs("../logs", exist_ok=True)

# Initialize CSV log
with open(LOG_PATH, "w") as f:
    f.write("timestamp,prediction\n")

print("Logging stream started... Press CTRL+C to stop.\n")

try:
    for i, row in df.iterrows():
        input_data = row.values.reshape(1, -1)
        try:
            pred = model.predict(input_data)[0]
        except NotFittedError:
            print("❌ The model isn't trained yet.")
            break

        label = "Normal" if pred == 0 else "Anomaly"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(LOG_PATH, "a") as f:
            f.write(f"{timestamp},{label}\n")

        print(f"[{timestamp}] → Prediction: {label}")
        time.sleep(0.5)  # simulate streaming delay

except KeyboardInterrupt:
    print("\n✅ Stream logging ended by user.")
