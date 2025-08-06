import pandas as pd
import os

# Absolute path to the original dataset
data_path = os.path.join("..", "data", "Train_Test_Network_dataset","train_test_network.csv")

# Output path
output_path = os.path.join("..","data", "sample.csv")

# Load the dataset
df = pd.read_csv(data_path)

# Take 1 random row and save it
df_sample = df.sample(1, random_state=42)
df_sample.to_csv(output_path, index=False)

print("✅ sample.csv generated successfully!")
