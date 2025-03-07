import pandas as pd
import numpy as np
from entropy_analysis import calculate_entropy  # Import entropy function

# Load dataset using the absolute path
file_path = "/Users/nikki/Documents/MyProjects/DecisionTreeProject/src/car.data"
df = pd.read_csv(file_path, names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"])

# Function to manually encode categorical variables
def encode_labels(df):
    label_encoders = {}
    for col in df.columns:
        unique_values = df[col].unique()
        label_map = {value: index for index, value in enumerate(unique_values)}
        df[col] = df[col].map(label_map)
        label_encoders[col] = label_map
    return df, label_encoders

# Encode categorical variables manually
df, label_encoders = encode_labels(df)

# Compute entropy of the class labels (before splitting)
class_entropy = calculate_entropy(df["class"])
print(f"Entropy of class labels: {class_entropy:.4f}")

# Function to compute information gain
def compute_information_gain(data, feature, target="class"):
    """Calculates information gain of a feature"""
    total_entropy = calculate_entropy(data[target])
    values, counts = np.unique(data[feature], return_counts=True)

    # Calculate weighted entropy after split
    weighted_entropy = sum(
        (counts[i] / sum(counts)) * calculate_entropy(data[data[feature] == values[i]][target])
        for i in range(len(values))
    )

    # Compute information gain
    info_gain = total_entropy - weighted_entropy
    return info_gain

# Compute information gain for each feature
info_gains = {feature: compute_information_gain(df, feature) for feature in df.columns if feature != "class"}

# Print results
print("\nInformation Gain for each feature:")
for feature, gain in info_gains.items():
    print(f"{feature}: {gain:.4f}")

# Find the best feature for the first split
best_feature = max(info_gains, key=info_gains.get)
print(f"\nBest feature to split on: {best_feature}")
