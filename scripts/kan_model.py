# scripts/kan_model.py
# Trains a compact MLP (“simulated KAN”) for IoT anomaly detection.
# Encodes categorical features, splits train/test, reports metrics, and
# saves the model plus label encoders under models/.

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from memory_profiler import profile

@profile
def train_kan_model(csv_path):
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Encode categorical columns and keep encoders
    label_encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    if "label" not in df.columns:
        raise ValueError("Dataset must have a 'label' column.")

    # Split features/target and train/test
    X = df.drop(columns=["label"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Compact MLP used as a stand-in for KAN
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        max_iter=300,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("✅ KAN Model Evaluation:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/kan_model.joblib")
    joblib.dump(label_encoders, "models/kan_encoders.joblib")
    print("KAN model and encoders saved to models/.")

if __name__ == "__main__":
    train_kan_model("../data/Train_Test_Network_dataset/train_test_network.csv")
