import pandas as pd
import numpy as np
import joblib
import os

# === Load trained model and encoders ===
model_path = "../models/model.pkl"
encoder_path = "../models/label_encoders.pkl"
sample_path = "../data/sample.csv"

if not os.path.exists(model_path) or not os.path.exists(encoder_path):
    raise FileNotFoundError("❌ Trained model or encoders not found in 'models' folder.")

model = joblib.load(model_path)
encoders = joblib.load(encoder_path)

# === Load input sample CSV ===
df = pd.read_csv(sample_path)
print(f"✅ Sample loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# === Encode categorical columns ===
for col in df.select_dtypes(include='object').columns:
    if col in encoders:
        le = encoders[col]
        # Add 'UNKNOWN' class if missing
        df[col] = df[col].apply(lambda x: x if x in le.classes_ else "UNKNOWN")
        if "UNKNOWN" not in le.classes_:
            le.classes_ = np.append(le.classes_, "UNKNOWN")
        df[col] = le.transform(df[col])
    else:
        print(f"⚠️ Encoder not found for column: {col}")

# === Make predictions ===
predictions = model.predict(df)

# === Print results ===
print("\n📊 Prediction Results:")
for i, pred in enumerate(predictions):
    label = "🟢 Normal" if pred == 0 else "🔴 Anomaly"
    print(f"📶 Record #{i+1} → Prediction: {label}")
