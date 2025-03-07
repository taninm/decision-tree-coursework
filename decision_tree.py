import os
from graphviz import Digraph
import pandas as pd
import numpy as np
from collections import Counter

# Manually set the path to the Graphviz executable
os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ["PATH"]

# Your other functions and code
def calculate_entropy(data):
    label_counts = Counter(data['class'])
    total_count = len(data)
    entropy = 0.0
    for count in label_counts.values():
        prob = count / total_count
        entropy -= prob * np.log2(prob)
    return entropy

def best_split(data, features):
    best_feature = None
    best_value = None
    best_entropy = float('inf')

    current_entropy = calculate_entropy(data)

    for feature in features:
        values = data[feature].unique()

        for value in values:
            left_split = data[data[feature] == value]
            right_split = data[data[feature] != value]

            left_entropy = calculate_entropy(left_split)
            right_entropy = calculate_entropy(right_split)

            weighted_entropy = (len(left_split) / len(data)) * left_entropy + (
                        len(right_split) / len(data)) * right_entropy

            if weighted_entropy < best_entropy:
                best_entropy = weighted_entropy
                best_feature = feature
                best_value = value

    return best_feature, best_value

def build_tree(data, features):
    if len(data['class'].unique()) == 1:
        return data['class'].iloc[0]

    if features.empty:
        return data['class'].mode()[0]

    best_feature, best_value = best_split(data, features)

    if best_feature is None:
        return data['class'].mode()[0]

    left_split = data[data[best_feature] == best_value]
    right_split = data[data[best_feature] != best_value]

    remaining_features = features[features != best_feature]

    left_tree = build_tree(left_split, remaining_features)
    right_tree = build_tree(right_split, remaining_features)

    return {
        'feature': best_feature,
        'value': best_value,
        'left': left_tree,
        'right': right_tree
    }

def plot_tree(tree, parent_name, graph):
    if isinstance(tree, dict):  # If it's a decision node
        feature = tree.get('feature', '')
        value = tree.get('value', '')

        # Convert to strings to avoid TypeError
        feature_str = str(feature)
        value_str = str(value)

        node_name = f"{parent_name}_{feature_str}_{value_str}"

        print(f"Feature: {feature_str}, Value: {value_str}, Node Name: {node_name}")

        graph.node(node_name, label=f"{feature_str}={value_str}", shape="box", width="0.2", height="0.2")
        graph.edge(parent_name, node_name)

        plot_tree(tree.get('left', {}), node_name, graph)
        plot_tree(tree.get('right', {}), node_name, graph)

    else:  # If it's a leaf node
        leaf_label = str(tree)  # Force conversion to string
        leaf_node = f"{parent_name}_leaf"

        print(f"Leaf Node: {leaf_node}, Label: {leaf_label}")

        graph.node(leaf_node, label=leaf_label, shape="ellipse", width="0.2", height="0.2")
        graph.edge(parent_name, leaf_node)

# Ensure file exists
data_path = "/Users/nikki/Documents/MyProjects/DecisionTreeProject/src/car.data"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"The data file was not found at {data_path}")

# Read the data
df = pd.read_csv(data_path, names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"])

# Prepare the features and build the tree
features = df.columns[df.columns != 'class']
tree = build_tree(df, features)

# Plot and save the tree
graph = Digraph(format='png')
graph.attr(size='10,10')

plot_tree(tree, "Root", graph)

# Render the tree to an output file
output_path = "/Users/nikki/Documents/MyProjects/DecisionTreeProject/output_tree"
graph.render(output_path)

