import json
import math
import random
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


def _predict_single(w: List[float], xi: List[float]) -> float:
    """Compute probability for a single example with current weights."""
    z = w[0]
    for j in range(len(xi)):
        z += w[j + 1] * xi[j]
    return sigmoid(z)


def _train_batch(
    X: List[List[float]],
    y_binary: List[float],
    learning_rate: float,
    num_iterations: int,
) -> List[float]:
    """Batch gradient descent (full dataset per step)."""
    m = len(X)
    n_features = len(X[0])
    w: List[float] = [0.0] * (n_features + 1)
    for it in range(num_iterations):
        print(f"Iteration {it} of {num_iterations}")
        grad_bias = 0.0
        grad: List[float] = [0.0] * n_features
        for i in range(m):
            xi = X[i]
            error = _predict_single(w, xi) - y_binary[i]
            grad_bias += error
            for j in range(n_features):
                grad[j] += error * xi[j]
        grad_bias /= m
        for j in range(n_features):
            grad[j] /= m
        w[0] -= learning_rate * grad_bias
        for j in range(n_features):
            w[j + 1] -= learning_rate * grad[j]
    return w


def _train_sgd(
    X: List[List[float]],
    y_binary: List[float],
    learning_rate: float,
    num_iterations: int,
) -> List[float]:
    """Stochastic gradient descent (one example per step)."""
    m = len(X)
    n_features = len(X[0])
    w: List[float] = [0.0] * (n_features + 1)
    for it in range(num_iterations):
        print(f"Iteration {it} of {num_iterations}")
        indices = list(range(m))
        random.shuffle(indices)
        for idx in indices:
            xi = X[idx]
            error = _predict_single(w, xi) - y_binary[idx]
            w[0] -= learning_rate * error
            for j in range(n_features):
                w[j + 1] -= learning_rate * error * xi[j]
    return w


def _train_minibatch(
    X: List[List[float]],
    y_binary: List[float],
    learning_rate: float,
    num_iterations: int,
    batch_size: int,
) -> List[float]:
    """Mini-batch gradient descent (subset per step)."""
    m = len(X)
    n_features = len(X[0])
    w: List[float] = [0.0] * (n_features + 1)
    batch_size = max(1, min(batch_size, m))
    for it in range(num_iterations):
        print(f"Iteration {it} of {num_iterations}")
        indices = list(range(m))
        random.shuffle(indices)
        for start in range(0, m, batch_size):
            batch_idx = indices[start : start + batch_size]
            b_len = len(batch_idx)
            grad_bias = 0.0
            grad: List[float] = [0.0] * n_features
            for idx in batch_idx:
                xi = X[idx]
                error = _predict_single(w, xi) - y_binary[idx]
                grad_bias += error
                for j in range(n_features):
                    grad[j] += error * xi[j]
            grad_bias /= b_len
            for j in range(n_features):
                grad[j] /= b_len
            w[0] -= learning_rate * grad_bias
            for j in range(n_features):
                w[j + 1] -= learning_rate * grad[j]
    return w


def train_one_vs_all(
    X: List[List[float]],
    y: List[int],
    num_classes: int,
    learning_rate: float = 0.1,
    num_iterations: int = 1000,
    algorithm: str = "batch",
    batch_size: int = 32,
) -> List[List[float]]:
    """
    Train a one-vs-all logistic regression classifier using the selected algorithm.

    Supported algorithms:
        - "batch": full batch gradient descent
        - "sgd": stochastic gradient descent
        - "minibatch": mini-batch gradient descent
    """
    m = len(X)
    if m == 0:
        raise ValueError("Empty feature matrix.")
    n_features = len(X[0])

    weights: List[List[float]] = [[0.0] * (n_features + 1) for _ in range(num_classes)]

    algorithm = algorithm.lower()
    for k in range(num_classes):
        y_binary = [1.0 if label == k else 0.0 for label in y]
        if algorithm == "batch":
            weights[k] = _train_batch(X, y_binary, learning_rate, num_iterations)
        elif algorithm == "sgd":
            weights[k] = _train_sgd(X, y_binary, learning_rate, num_iterations)
        elif algorithm == "minibatch" or algorithm == "mini-batch":
            weights[k] = _train_minibatch(
                X, y_binary, learning_rate, num_iterations, batch_size
            )
        else:
            raise ValueError(f"Unknown training algorithm: {algorithm}")

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
        algorithm = input(
            "Select training algorithm [batch | sgd | minibatch]: "
        ).strip()
        if algorithm == "":
            algorithm = "batch"

        batch_size = 32
        if algorithm.lower() in {"minibatch", "mini-batch"}:
            user_batch = input("Mini-batch size (default 32): ").strip()
            if user_batch.isdigit() and int(user_batch) > 0:
                batch_size = int(user_batch)

        X, y, means, stds = prepare_dataset(train_path)
        weights = train_one_vs_all(
            X,
            y,
            len(HOUSES),
            algorithm=algorithm,
            batch_size=batch_size,
        )
        save_model(weights, means, stds, "weights.json")
        print("Training completed. Weights saved to weights.json")

    except Exception as e:
        print(f"{type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv)
