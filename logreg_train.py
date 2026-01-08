import json
import math
import sys
from typing import List, Tuple

from utils import compute_stats, is_float, read_dataset

COURSE_COLUMNS: List[str] = [
    "Arithmancy",
    "Astronomy",
    "Herbology",
    "Defense Against the Dark Arts",
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Potions",
    "Care of Magical Creatures",
    "Charms",
    "Flying",
]

HOUSES: List[str] = [
    "Gryffindor",
    "Hufflepuff",
    "Ravenclaw",
    "Slytherin",
]


def prepare_dataset(
    path: str,
) -> Tuple[List[List[float]], List[int], List[float], List[float]]:
    """
    Load the training dataset and build the learning matrices:

    - X: list of standardized feature vectors (imputed and scaled)
    - y: list of integer house indices (0..3)
    - means, stds: per-feature statistics used for standardization
    """
    header, rows = read_dataset(path)
    if not header:
        raise ValueError("Empty training dataset.")

    # Raw feature values per feature index (for computing means/stds)
    raw_features: List[List[float]] = [[] for _ in COURSE_COLUMNS]  # Initialize empty list per feature to collect numeric values
    labels: List[int] = []  # List to store integer house labels (0, 1, 2, or 3)

    for row in rows:
        house = row.get("Hogwarts House", "")
        if house not in HOUSES:
            continue
        labels.append(HOUSES.index(house))

        for j, col in enumerate(COURSE_COLUMNS):
            val = row.get(col, "").strip()
            if val == "" or not is_float(val):
                continue
            raw_features[j].append(float(val))

    if not labels:
        raise ValueError("No labeled rows found in training dataset.")

    means: List[float] = []
    stds: List[float] = []
    for j in range(len(COURSE_COLUMNS)):
        stats = compute_stats(raw_features[j])
        mean = stats.get("mean", 0.0)
        std = stats.get("std", 1.0)
        if std == 0.0:
            std = 1.0
        means.append(mean)
        stds.append(std)
    
    X: List[List[float]] = [] # List to store standardized feature vectors (one per student)
    for row in rows:
        house = row.get("Hogwarts House", "")
        if house not in HOUSES:
            continue
        features: List[float] = [] # Initialize empty list for this student's feature vector
        for j, col in enumerate(COURSE_COLUMNS):
            val = row.get(col, "").strip()
            if val == "" or not is_float(val):
                v = means[j]
            else:
                v = float(val)
            # Standardize: (x - mean) / std
            features.append((v - means[j]) / stds[j])  # Apply z-score normalization and add to feature vector
        X.append(features)  # Add this student's complete feature vector to the dataset

    return X, labels, means, stds  # Return standardized features, labels, and statistics for later use


def sigmoid(z: float) -> float:
    """
    Numerically stable implementation of the logistic sigmoid function:

        sigmoid(z) = 1 / (1 + exp(-z))

    We clamp extreme values of `z` to avoid overflow in exp().
    """
    if z < -709:
        return 0.0
    if z > 709:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def train_one_vs_all(
    X: List[List[float]],
    y: List[int],
    num_classes: int,
    learning_rate: float = 0.1,
    num_iterations: int = 1000,
) -> List[List[float]]:
    """
    Train a one-vs-all logistic regression classifier using batch gradient descent.

    For each class k (each house), we fit a binary classifier:
        y_k = 1 if y == k else 0
    and learn a weight vector [bias, w1, ..., wd].

    Returns:
        weights: list of weight vectors, one per class.
    """
    m = len(X)
    if m == 0:
        raise ValueError("Empty feature matrix.")
    n_features = len(X[0])  # dimensionality of each feature vector (number of course features)

    # Initialize weights for each class: one bias + one weight per feature
    weights: List[List[float]] = [
        [0.0] * (n_features + 1) for _ in range(num_classes)  # Create weight vector [bias, w1, w2, ...] for each class, initialized to zeros
    ]

    for k in range(num_classes):  # Train a binary classifier for each house (one-vs-all)
        w = weights[k]
        for iteration in range(num_iterations):  # Gradient descent
            print(f"Iteration {iteration} of {num_iterations} of class {k}")
            grad_bias = 0.0
            grad: List[float] = [0.0] * n_features
            for i in range(m): # Loop through each training example (student)
                xi = X[i] # Get the feature vector for student i
                yi = 1.0 if y[i] == k else 0.0 # one-vs-all target for class k (1 if student belongs to house k, else 0)
                z = w[0] # Initialize z to bias
                for j in range(n_features):
                    z += w[j + 1] * xi[j] # Multiply feature value by its weight and add to z
                probability = sigmoid(z) # Apply sigmoid to get probability in [0, 1]
                error = probability - yi
                grad_bias += error # Accumulate gradient for bias term
                for j in range(n_features):  # Accumulate gradient for each feature weight
                    grad[j] += error * xi[j]  # Gradient for weight j is error times feature j value

            # Average the gradients and apply gradient descent step
            grad_bias /= m  # Average the bias gradient
            for j in range(n_features):  # Average each feature gradient
                grad[j] /= m

            w[0] -= learning_rate * grad_bias
            for j in range(n_features):
                w[j + 1] -= learning_rate * grad[j]

    return weights


def save_model(
    weights: List[List[float]],
    means: List[float],
    stds: List[float],
    path: str = "weights.json",
) -> None:
    """ Serialize the trained model to disk as a JSON file."""
    model = {
        "features": COURSE_COLUMNS,
        "means": means,
        "stds": stds,
        "houses": HOUSES,
        "weights": weights,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(model, f, indent=4)
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
        sys.exit(1)


def main(argv: List[str]) -> None:
    if len(argv) != 2:
        print("Usage: python logreg_train.py <training_dataset.csv>")
        sys.exit(1)

    try:
        train_path = argv[1]
        X, y, means, stds = prepare_dataset(train_path)
        weights = train_one_vs_all(X, y, len(HOUSES))
        save_model(weights, means, stds, "weights.json")
        print("Training completed. Weights saved to weights.json")

    except Exception as e:
        print(f"{type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv)
