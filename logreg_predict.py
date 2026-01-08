import json
import math
import sys
from typing import Dict, List, Tuple

from utils import is_float, read_dataset


def sigmoid(z: float) -> float:
    """Sigmoid function."""
    if z < -709:
        return 0.0
    if z > 709:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def load_model_json(path: str) -> Dict:
    """Load the JSON model file."""
    try: 
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def prepare_features(
    rows: List[Dict[str, str]],
    feature_names: List[str],
    means: List[float],
    stds: List[float],
) -> Tuple[List[List[float]], List[int]]:
    """
    Build a standardized feature matrix for the test set and track row indices.

    For each input row we:
    - read the `Index` column (used later in the output file),
    - for each course feature:
        * replace missing values by the training mean,
        * standardize the value with training mean and std.
    """
    X: List[List[float]] = []
    indices: List[int] = []
    for row in rows:
        index_str = row.get("Index", "").strip()
        if index_str == "" or not index_str.isdigit():
            continue
        idx = int(index_str)
        indices.append(idx)

        features: List[float] = []
        for j, col in enumerate(feature_names):
            val = row.get(col, "").strip()
            if val == "" or not is_float(val):
                v = means[j]
            else:
                v = float(val)
            std = stds[j] if stds[j] != 0 else 1.0
            # Apply the same standardization as in training
            features.append((v - means[j]) / std)
        X.append(features)
    return X, indices


def predict(
    X: List[List[float]],
    weights: List[List[float]],
    houses: List[str],
) -> List[str]:
    """
    Apply the logistic regression model to each feature vector.

    For each example:
        - compute the probability for each class (house),
        - return the house with the highest probability.
    """
    predictions: List[str] = []
    m = len(X)
    num_classes = len(houses)
    for i in range(m):
        xi = X[i]
        best_class = 0
        best_score = -1.0
        for k in range(num_classes):
            w = weights[k]
            # Compute z_k = w0 + w1*x1 + ... + wd*xd
            z = w[0]
            for j in range(len(xi)):
                z += w[j + 1] * xi[j]
            probability = sigmoid(z)
            if probability > best_score:
                best_score = probability
                best_class = k
        predictions.append(houses[best_class])
    return predictions


def write_predictions(indices: List[int], houses: List[str], path: str) -> None:
    """Write predictions to a CSV file. """
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("Index,Hogwarts House\n")
            for idx, house in zip(indices, houses):
                f.write(f"{idx},{house}\n")
    except Exception as e:
        print(f"Error writing predictions: {e}")
        sys.exit(1)

def main(argv: List[str]) -> None:
    if len(argv) != 3:
        print("Usage: python logreg_predict.py <test dataset> <weights file>")
        sys.exit(1)

    try:
        test_path = argv[1]
        weights_path = argv[2]

        model = load_model_json(weights_path)
        feature_names: List[str] = model["features"]
        means: List[float] = model["means"]
        stds: List[float] = model["stds"]
        houses: List[str] = model["houses"]
        weights: List[List[float]] = model["weights"]

        header, rows = read_dataset(test_path)
        if not header:
            print("Empty test dataset.")
            sys.exit(1)

        # Transform rows into standardized feature vectors
        X, indices = prepare_features(rows, feature_names, means, stds)
        preds = predict(X, weights, houses)
        write_predictions(indices, preds, path="houses.csv")
        print("Predictions written to houses.csv")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv)
