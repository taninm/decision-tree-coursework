import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

file_path = "/Users/nikki/Documents/MyProjects/DecisionTreeProject/src/car.data"

if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' was not found. Please check the file path.")
else:
    df = pd.read_csv(file_path, names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"])

    unique_classes = df["class"].unique()
    class_counts = df["class"].value_counts()

    print(df.head())
    print(f"Dataset Shape: {df.shape}")
    print("Class distribution:\n", class_counts)

    easiest_class = class_counts.idxmax()
    hardest_class = class_counts.idxmin()

    print(f"Easiest to recognize: {easiest_class} (Most samples: {class_counts.max()})")
    print(f"Hardest to recognize: {hardest_class} (Fewest samples: {class_counts.min()})")

    plt.figure(figsize=(8, 5))
    plt.bar(class_counts.index, class_counts.values, color='skyblue')
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution in Dataset")
    plt.show()
