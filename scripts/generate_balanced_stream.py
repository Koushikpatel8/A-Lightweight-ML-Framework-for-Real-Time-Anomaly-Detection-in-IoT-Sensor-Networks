# scripts/generate_balanced_stream.py
import pandas as pd

# Load your full dataset
df = pd.read_csv("../data/Train_Test_Network_dataset/train_test_network.csv")

# Sample 50 normal and 50 anomaly records
normal_df = df[df['label'] == 0].sample(50, random_state=42)
anomaly_df = df[df['label'] == 1].sample(50, random_state=42)

# Combine and shuffle
balanced_df = pd.concat([normal_df, anomaly_df]).sample(frac=1, random_state=42)

# Save the new CSV for stream testing
balanced_df.to_csv("../data/balanced_stream.csv", index=False)
print("✅ Created 'balanced_stream.csv' with 50 normal and 50 anomaly records.")
