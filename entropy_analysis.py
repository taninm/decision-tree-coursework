import os
import numpy as np
import pandas as pd

# Print the current working directory
print("Current working directory:", os.getcwd())

# Absolute path to the 'car.data' file
file_path = "/Users/nikki/Documents/MyProjects/DecisionTreeProject/src/car.data"

# Function to calculate entropy
def calculate_entropy(y):
    counts = np.bincount(y)  # Count occurrences of each class
    probabilities = counts / len(y)  # Convert counts to probabilities
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

# Function to encode categorical variables manually
def encode_labels(df):
    label_encoders = {}
    for col in df.columns:
        unique_values = df[col].unique()
        label_map = {value: index for index, value in enumerate(unique_values)}
        df[col] = df[col].map(label_map)
        label_encoders[col] = label_map
    return df, label_encoders

# Load dataset using absolute path
df = pd.read_csv(file_path, names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"])

# Encode categorical variables manually
df, label_encoders = encode_labels(df)

# Extract target variable
y = df["class"]

# Compute entropy of the target variable
class_entropy = calculate_entropy(y)
print(f"Entropy of class labels: {class_entropy:.4f}")

