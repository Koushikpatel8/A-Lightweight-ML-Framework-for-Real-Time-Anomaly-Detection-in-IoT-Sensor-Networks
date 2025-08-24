# scripts/stream_simulator.py
# This script simulates a live IoT anomaly detection stream.
# It loads a pre-trained model and encoders, processes a balanced dataset,
# and prints predictions record by record while saving them to a log file.
# The simulation helps demonstrate how the system would behave in real-time.

from pathlib import Path
import time
import pandas as pd
import joblib
from datetime import datetime
from termcolor import colored
import os

# Path handling:
# The script checks for environment overrides, Windows absolute paths (laptop),
# or falls back to repo-relative paths (Raspberry Pi / Linux).
HERE = Path(__file__).resolve().parent           # .../scripts
PROJECT = HERE.parent                            # project root

REL_MODEL_PATH = HERE / "models" / "kan_model.joblib"
REL_ENC_PATH   = HERE / "models" / "kan_encoders.joblib"
REL_DATA_PATH  = PROJECT / "data" / "Train_Test_Network_dataset" / "balanced_stream.csv"
REL_LOGS_DIR   = PROJECT / "logs"
REL_LOG_PATH   = REL_LOGS_DIR / "anomaly_predictions.csv"

WIN_MODEL_PATH = Path(r"C:\Users\saite\Desktop\IoT_Anomaly_Detection_Project\scripts\models\kan_model.joblib")
WIN_ENC_PATH   = Path(r"C:\Users\saite\Desktop\IoT_Anomaly_Detection_Project\scripts\models\kan_encoders.joblib")
WIN_DATA_PATH  = Path(r"C:\Users\saite\Desktop\IoT_Anomaly_Detection_Project\data\Train_Test_Network_dataset\balanced_stream.csv")
WIN_LOGS_DIR   = Path(r"C:\Users\saite\Desktop\IoT_Anomaly_Detection_Project\logs")
WIN_LOG_PATH   = Path(r"C:\Users\saite\Desktop\IoT_Anomaly_Detection_Project\logs\anomaly_predictions.csv")

ENV_MODEL_PATH = os.environ.get("KAN_MODEL_PATH")
ENV_ENC_PATH   = os.environ.get("KAN_ENCODERS_PATH")
ENV_DATA_PATH  = os.environ.get("KAN_DATA_PATH")
ENV_LOG_DIR    = os.environ.get("KAN_LOG_DIR")
ENV_LOG_PATH   = os.environ.get("KAN_LOG_PATH")

def choose_path(win_path: Path, rel_path: Path, env_override: str | None) -> Path:
    """Decides which path to use: environment override, Windows absolute, or repo-relative."""
    if env_override:
        p = Path(env_override)
        if p.exists() or p.parent.exists():
            return p
    if win_path.exists():
        return win_path
    return rel_path

MODEL_PATH = choose_path(WIN_MODEL_PATH, REL_MODEL_PATH, ENV_MODEL_PATH)
ENC_PATH   = choose_path(WIN_ENC_PATH,   REL_ENC_PATH,   ENV_ENC_PATH)
DATA_PATH  = choose_path(WIN_DATA_PATH,  REL_DATA_PATH,  ENV_DATA_PATH)
LOGS_DIR   = choose_path(WIN_LOGS_DIR,   REL_LOGS_DIR,   ENV_LOG_DIR)
LOG_PATH   = choose_path(WIN_LOG_PATH,   REL_LOG_PATH,   ENV_LOG_PATH)

def main():
    # Check if the model, encoders, and dataset exist
    if not MODEL_PATH.exists() or not ENC_PATH.exists():
        raise FileNotFoundError(
            f"❌ Model/encoders missing.\n"
            f"- Model:   {MODEL_PATH}\n"
            f"- Encoders:{ENC_PATH}\n"
            f"Tip: train/save them first (e.g., run `python scripts/kan_model.py`)."
        )
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"❌ Data not found at:\n{DATA_PATH}\n"
            f"Tip: create it with your balancing script or point DATA_PATH to an existing CSV."
        )

    # Load model and encoders
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENC_PATH)

    # Load stream dataset
    df = pd.read_csv(DATA_PATH)

    # Encode categorical columns (drop unseen columns if necessary)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        if col in encoders:
            df[col] = encoders[col].transform(df[col].astype(str))
        else:
            df.drop(columns=[col], inplace=True)
            print(f"⚠️ Dropped unseen categorical column: {col}")

    # Drop label column, since we are simulating predictions
    if "label" in df.columns:
        df = df.drop(columns=["label"])

    # Ensure logs directory exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Start a new log file for this simulation
    with LOG_PATH.open("w", encoding="utf-8") as f:
        f.write("timestamp,record_number,prediction\n")

    print(" Starting simulated stream...\n")
    print(f"Model: {MODEL_PATH}")
    print(f"Encoders: {ENC_PATH}")
    print(f"Data: {DATA_PATH}")
    print(f"Log: {LOG_PATH}\n")

    # Process data one record at a time to simulate streaming
    for idx, row in df.iterrows():
        sample_df = pd.DataFrame([row], columns=df.columns)
        pred = model.predict(sample_df)[0]

        # Display result in console
        label_text = "🟢 Normal" if pred == 0 else "🔴 Anomaly"
        color = "green" if pred == 0 else "red"
        print(f"Record #{idx + 1}: Prediction → {colored(label_text, color)}")

        # Log result with timestamp
        with LOG_PATH.open("a", encoding="utf-8") as f:
            ts = datetime.now().isoformat()
            f.write(f"{ts},{idx+1},{'Normal' if pred == 0 else 'Anomaly'}\n")

        time.sleep(0.3)  # small delay to mimic real-time streaming

    print(f"\n✅ Stream simulation complete.\n Logs saved at: {LOG_PATH}")

if __name__ == "__main__":
    main()
