# scripts/evaluate_stream.py
# Evaluates the trained model on a balanced test stream.
# Encodes categorical features, generates predictions, prints
# classification metrics, and saves results to a CSV file.

import pandas as pd
import joblib
from sklearn.metrics import classification_report

# Load model and encoders
model = joblib.load("models/kan_model.joblib")
encoders = joblib.load("models/kan_encoders.joblib")

# Load balanced stream with true labels
df = pd.read_csv("../data/Train_Test_Network_dataset/balanced_stream.csv")

# Encode categorical columns using saved encoders
for col in df.select_dtypes(include='object').columns:
    if col in encoders:
        df[col] = encoders[col].transform(df[col].astype(str))

# Separate features and target
X = df.drop(columns=["label"])
y_true = df["label"]

# Predict using trained model
y_pred = model.predict(X)

# Print evaluation metrics
print("Evaluation on Balanced Stream:")
print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))

# Save predictions alongside original data
df["prediction"] = y_pred
df.to_csv("../data/stream_predictions.csv", index=False)
print("Predictions saved to data/stream_predictions.csv")
