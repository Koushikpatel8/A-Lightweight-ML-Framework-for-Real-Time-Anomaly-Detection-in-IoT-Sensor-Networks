# scripts/evaluate_stream.py
import pandas as pd
import joblib
from sklearn.metrics import classification_report

# Load model and encoders
model = joblib.load("models/kan_model.joblib")
encoders = joblib.load("models/kan_encoders.joblib")

# Load balanced stream with true labels
df = pd.read_csv("../data/Train_Test_Network_dataset/balanced_stream.csv")

# Encode categorical columns
for col in df.select_dtypes(include='object').columns:
    if col in encoders:
        df[col] = encoders[col].transform(df[col].astype(str))

# Split features and target
X = df.drop(columns=["label"])
y_true = df["label"]

# Predict
y_pred = model.predict(X)

# Show classification metrics
print("📊 Evaluation on Balanced Stream:")
print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))

# Save predictions to a CSV file
df["prediction"] = y_pred
df.to_csv("../data/stream_predictions.csv", index=False)
print("📁 Predictions saved to data/stream_predictions.csv")
