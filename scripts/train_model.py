# scripts/train_model.py
# Trains a Random Forest for IoT anomaly detection with class imbalance handling,
# saves the model + encoders to scripts/models, and prints a quick evaluation.

import os
import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Paths
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA_PATH = PROJECT / "data" / "Train_Test_Network_dataset" / "train_test_network.csv"
MODELS_DIR = HERE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUT = MODELS_DIR / "kan_model.joblib"        # keep names your other scripts expect
ENC_OUT   = MODELS_DIR / "kan_encoders.joblib"

# 1) Load data
df = pd.read_csv(DATA_PATH)

# 2) Encode categoricals
encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# 3) Split
X = df.drop(columns=["label"])
y = df["label"]

# Show class balance
print("Class counts:", y.value_counts().to_dict())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y  # stratify helps class balance
)

# 4) Train RandomForest with imbalance awareness
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"  # combats imbalance
)
rf.fit(X_train, y_train)

# 5) Quick evaluation
y_pred = rf.predict(X_test)
print("\nEvaluation (RandomForest):")
print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

# 6) Save
joblib.dump(rf, MODEL_OUT)
joblib.dump(encoders, ENC_OUT)
print(f"\n✅ Saved model → {MODEL_OUT}")
print(f"✅ Saved encoders → {ENC_OUT}")
