
"""
kan_model.py - Simulated Kolmogorov–Arnold Network (KAN) for IoT Anomaly Detection
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from memory_profiler import profile

@profile
def train_kan_model(csv_path):
    print(f"📥 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    if 'label' not in df.columns:
        raise ValueError("Dataset must have a 'label' column.")

    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Simulated KAN using a compact MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', max_iter=300, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("✅ KAN Model Evaluation:")
    print(classification_report(y_test, y_pred))
    print("📊 Accuracy:", accuracy_score(y_test, y_pred))

    # Save model and encoders
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/kan_model.joblib")
    joblib.dump(label_encoders, "models/kan_encoders.joblib")
    print("💾 KAN model and encoders saved.")

if __name__ == "__main__":
    train_kan_model("../data/Train_Test_Network_dataset/train_test_network.csv")
