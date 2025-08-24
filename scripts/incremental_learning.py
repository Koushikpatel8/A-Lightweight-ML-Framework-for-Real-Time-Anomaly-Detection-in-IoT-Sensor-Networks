# scripts/incremental_learning.py
# Trains and evaluates an online (incremental) Hoeffding Tree on a data stream.
# Encodes categorical features, streams 500 records, tracks accuracy over time,
# and saves an accuracy log CSV for later visualization.

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from river import tree, metrics, stream
from memory_profiler import profile

@profile
def run_incremental_learning():
    # Load dataset
    df = pd.read_csv("../data/Train_Test_Network_dataset/train_test_network.csv")

    # Encode categorical columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Split features and target
    X = df.drop(columns=["label"])
    y = df["label"]

    # Create a streaming iterator from pandas
    data_stream = stream.iter_pandas(X, y)

    # Initialize online model and metric
    model = tree.HoeffdingTreeClassifier()
    accuracy = metrics.Accuracy()
    accuracy_log = []

    print("Starting incremental learning on 500 records...\n")

    # Train + evaluate incrementally
    for i, (x, y_true) in enumerate(data_stream):
        y_pred = model.predict_one(x)
        model.learn_one(x, y_true)

        # Only update accuracy if a prediction was made
        if y_pred is not None:
            accuracy.update(y_true, y_pred)

        acc = accuracy.get()
        accuracy_log.append({"Record": i + 1, "Accuracy": acc})

        if i % 100 == 0:
            print(f"Record #{i+1} → Predicted: {y_pred}, Actual: {y_true}, Accuracy: {acc:.4f}")

        if i >= 499:  # stop after 500 samples
            break

    print(f"\n✅ Final accuracy after 500 samples: {accuracy.get():.4f}")

    # Save accuracy log
    os.makedirs("../outputs", exist_ok=True)
    out_path = "../outputs/incremental_accuracy_log.csv"
    pd.DataFrame(accuracy_log).to_csv(out_path, index=False)
    print(f"Accuracy log saved to: {out_path}")

if __name__ == "__main__":
    run_incremental_learning()
