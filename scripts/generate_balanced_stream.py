# scripts/generate_balanced_stream.py
# Creates a balanced sample dataset for streaming tests.
# Randomly selects 50 normal and 50 anomaly records, shuffles them,
# and saves the result as balanced_stream.csv.

import pandas as pd

# Load the full dataset
df = pd.read_csv("../data/Train_Test_Network_dataset/train_test_network.csv")

# Randomly sample 50 records of each class
normal_df = df[df['label'] == 0].sample(50, random_state=42)
anomaly_df = df[df['label'] == 1].sample(50, random_state=42)

# Combine both samples and shuffle
balanced_df = pd.concat([normal_df, anomaly_df]).sample(frac=1, random_state=42)

# Save balanced dataset for stream testing
balanced_df.to_csv("../data/balanced_stream.csv", index=False)
print("✅ Created 'balanced_stream.csv' with 50 normal and 50 anomaly records.")
