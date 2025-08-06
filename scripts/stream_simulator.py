# scripts/stream_simulator.py
import os
import time
import pandas as pd
import joblib
from datetime import datetime
from termcolor import colored

# Load KAN model and encoders
model = joblib.load("models/kan_model.joblib")
encoders = joblib.load("models/kan_encoders.joblib")

# Load full dataset to simulate stream
df = pd.read_csv("../data/Train_Test_Network_dataset/balanced_stream.csv")

# Encode categorical columns using pre-fitted encoders
for col in df.select_dtypes(include='object').columns:
    if col in encoders:
        df[col] = encoders[col].transform(df[col].astype(str))

# Drop the label column for prediction
if 'label' in df.columns:
    df.drop(columns=['label'], inplace=True)

# Ensure logs folder exists
os.makedirs("../logs", exist_ok=True)

# Create log file
log_path = "../logs/anomaly_predictions.csv"
with open(log_path, "w") as f:
    f.write("timestamp,record_number,prediction\n")

print("🚀 Starting simulated stream...\n")

# Stream records one by one
for idx, row in df.iterrows():
    sample_df = pd.DataFrame([row])  # preserve column names
    pred = model.predict(sample_df)[0]
    label = "🟢 Normal" if pred == 0 else "🔴 Anomaly"
    color = "green" if pred == 0 else "red"
    print(f"Record #{idx + 1}: Prediction → {colored(label, color)}")

    # Log prediction with timestamp
    with open(log_path, "a") as f:
        timestamp = datetime.now().isoformat()
        f.write(f"{timestamp},{idx+1},{'Normal' if pred == 0 else 'Anomaly'}\n")

    time.sleep(0.3)  # Simulate delay

print("\n✅ Stream simulation complete. Logs saved at:", log_path)
