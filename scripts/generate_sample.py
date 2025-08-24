# scripts/generate_sample.py
# Creates a one-row sample from the main dataset.
# Useful for quick testing of model predictions.

import pandas as pd
import os

# Path to the original dataset
data_path = os.path.join("..", "data", "Train_Test_Network_dataset", "train_test_network.csv")

# Path to save the sample file
output_path = os.path.join("..", "data", "sample.csv")

# Load dataset
df = pd.read_csv(data_path)

# Randomly select 1 row
df_sample = df.sample(1, random_state=42)

# Save as sample.csv
df_sample.to_csv(output_path, index=False)
print("✅ sample.csv generated successfully!")
