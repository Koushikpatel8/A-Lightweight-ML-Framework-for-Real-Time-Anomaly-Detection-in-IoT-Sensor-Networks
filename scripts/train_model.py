# scripts/train_model.py
# This script trains a Random Forest model for IoT anomaly detection.
# It loads the dataset, encodes categorical features, splits data into train/test,
# trains the model, and finally saves the trained model and encoders for later use.

import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Step 1: Load dataset
# The dataset is stored in the project data folder
data_path = "../data/Train_Test_Network_dataset/train_test_network.csv"
df = pd.read_csv(data_path)

# Step 2: Encode categorical features
# Categorical columns are transformed into numeric codes
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 3: Split into features and labels
# Features (X) exclude the label column, target (y) is the label
X = df.drop('label', axis=1)
y = df['label']

# Step 4: Train/test split
# The dataset is split 80/20 for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
# Random Forest is chosen for its robustness and ensemble learning approach
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Save model and encoders
# The trained model and encoders are saved to the models directory
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/model.pkl")
joblib.dump(label_encoders, "../models/label_encoders.pkl")

print("✅ Model and encoders saved successfully!")
