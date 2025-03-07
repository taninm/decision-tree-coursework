import pandas as pd
import numpy as np

# Define the entropy function
def entropy(y):
    value_counts = y.value_counts(normalize=True)
    return -sum(value_counts * np.log2(value_counts))

# Define the function to find the best split
def best_split(X, y):
    best_entropy = float("inf")
    best_column = None
    best_value = None

    # Iterate through each column to find the best split
    for column in X.columns:
        values = X[column].unique()
        for value in values:
            left_mask = X[column] <= value
            right_mask = ~left_mask

            if left_mask.any() and right_mask.any():
                left_entropy = entropy(y[left_mask])
                right_entropy = entropy(y[right_mask])
                total_entropy = (len(y[left_mask]) / len(y)) * left_entropy + (len(y[right_mask]) / len(y)) * right_entropy

                if total_entropy < best_entropy:
                    best_entropy = total_entropy
                    best_column = column
                    best_value = value

    if best_column is None:
        print("No valid split found.")
    return best_column, best_value

# Define the function to build the decision tree
def build_tree(X, y, depth=0, max_depth=5):
    if len(y.unique()) == 1 or depth == max_depth:
        return {'class': y.iloc[0]}

    if X.empty or len(y) == 0:
        return {'class': y.mode()[0]}

    best_split_column, best_split_value = best_split(X, y)

    # Debugging output
    print(f"Best split column: {best_split_column}")
    print(f"Best split value: {best_split_value}")

    # If no valid split is found, return the majority class
    if best_split_column is None or best_split_value is None:
        return {'class': y.mode()[0]}

    left_mask = X[best_split_column] <= best_split_value
    right_mask = ~left_mask

    if not left_mask.any() or not right_mask.any():
        return {'class': y.mode()[0]}

    left_tree = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth)
    right_tree = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth)

    return {'column': best_split_column, 'value': best_split_value, 'left': left_tree, 'right': right_tree}

# Example usage with a sample dataset
if __name__ == "__main__":
    # Example data
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
        'feature2': [2, 3, 4, 5, 6, 7, 8, 9],
        'class': ['acc', 'good', 'unacc', 'vgood', 'acc', 'good', 'unacc', 'vgood']
    }

    df = pd.DataFrame(data)
    X = df.drop(columns=['class'])
    y = df['class']

    # Show class distribution before encoding
    print("Class distribution before encoding:")
    print(y.value_counts())

    # Encode the target variable
    y_encoded = y.map({'acc': 0, 'good': 1, 'unacc': 2, 'vgood': 3})
    print("\nClass distribution after encoding:")
    print(y_encoded.value_counts())

    # Build the decision tree
    tree_model = build_tree(X, y_encoded, max_depth=3)
    print("\nDecision Tree Model:")
    print(tree_model)

