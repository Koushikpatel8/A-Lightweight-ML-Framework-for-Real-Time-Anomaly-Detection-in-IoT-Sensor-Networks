# scripts/log_stream_for_powerbi.py
import pandas as pd
import joblib
import os
import time
from datetime import datetime
from sklearn.exceptions import NotFittedError

# Load model and encoders
MODEL_PATH = "../scripts/models/kan_model.joblib"
ENCODER_PATH = "../scripts/models/kan_encoders.joblib"
DATA_PATH = "../data/Train_Test_Network_dataset/train_test_network.csv"
LOG_PATH = "../logs/powerbi_logs.csv"

print("🔁 Loading model and encoders...")
try:
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
except FileNotFoundError:
    print("❌ Model or encoder file not found. Please train the model first.")
    exit()

# Load dataset
df = pd.read_csv(DATA_PATH)

# Apply label encoding
for col in df.select_dtypes(include='object').columns:
    if col in encoders:
        df[col] = encoders[col].transform(df[col].astype(str))

# Remove label column (simulate real-time input)
if 'label' in df.columns:
    df.drop(columns=['label'], inplace=True)

# Ensure output folder exists
os.makedirs("../data", exist_ok=True)

# Initialize or clear CSV log
with open(LOG_PATH, 'w') as f:
    f.write("timestamp,prediction\n")

print("🚀 Logging stream started... Press CTRL+C to stop.\n")

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

        with open(LOG_PATH, 'a') as f:
            f.write(f"{timestamp},{label}\n")

        print(f"[{timestamp}] → Prediction: {label}")
        time.sleep(0.5)  # Simulate delay

except KeyboardInterrupt:
    print("\n✅ Stream logging ended by user.")
