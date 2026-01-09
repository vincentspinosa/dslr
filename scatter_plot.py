import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from utils import compute_stats_as_dict


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


HOUSES_COLORS: Dict[str, str] = {
    "Gryffindor": "red",
    "Hufflepuff": "gold",
    "Ravenclaw": "blue",
    "Slytherin": "green",
}


def compute_pearson(x: List[float], y: List[float]) -> float:
    """
    Compute the **Pearson correlation coefficient** between two lists of numbers.

    Pearson correlation, usually denoted by r, measures the **strength and
    direction of a linear relationship** between two variables X and Y.

    r is always in [-1, 1]:
        - r ≈  1  → strong positive linear relationship,
        - r ≈ -1  → strong negative linear relationship,
        - r ≈  0  → little or no linear relationship.
    """
    if len(x) == 0 or len(x) != len(y):
        # If there are no values, or the lists are misaligned, return 0 by convention.
        return 0.0
    # Use our manual stats function for means and standard deviations
    stats_x = compute_stats_as_dict(x)  # contains mean and std for X values
    stats_y = compute_stats_as_dict(y)  # contains mean and std for Y values
    mean_x = stats_x["mean"]  # mean of X
    mean_y = stats_y["mean"]  # mean of Y
    std_x = stats_x["std"]  # standard deviation of X
    std_y = stats_y["std"]  # standard deviation of Y
    if std_x == 0 or std_y == 0:
        # If one std is 0 -> correlation is undefined,
        # and we treat it as 0 because there is no variation to correlate.
        return 0.0

    # Compute covariance:
    n = len(x)
    cov_sum = 0.0
    for i in range(n):
        var_x = x[i] - mean_x
        var_y = y[i] - mean_y
        cov_sum += var_x * var_y
    cov = cov_sum / n
    # Finally, Pearson r = covariance divided by the product of standard deviations.
    return cov / (std_x * std_y)


def plot_best_pair(path: str) -> None:
    """
    Find the pair of features with the largest Pearson correlation using pandas,
    print the pair and its correlation, and display a scatter plot colored
    by house.
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        print("Could not read dataset.")
        return

    if df.empty:
        print("Empty dataset.")
        return

    # Keep only numeric course columns and drop rows where both values are NaN
    numeric_df = df[COURSE_COLUMNS].apply(pd.to_numeric, errors="coerce")

    # Best feature pair and correlation score found so far
    best_c1 = ""
    best_c2 = ""
    best_r = 0.0

    # Try all pairs of course features
    for i in range(len(COURSE_COLUMNS)):
        for j in range(i + 1, len(COURSE_COLUMNS)):
            c1 = COURSE_COLUMNS[i]
            c2 = COURSE_COLUMNS[j]
            pair_df = numeric_df[[c1, c2]].dropna()
            if len(pair_df) < 2:
                continue
            x_values = pair_df[c1].tolist()
            y_values = pair_df[c2].tolist()
            r = compute_pearson(x_values, y_values)
            # Keep the pair with the strongest linear relationship
            if r > best_r:
                best_c1, best_c2, best_r = c1, c2, r

    if best_c1 == "" or best_c2 == "":
        print("Could not find a suitable pair of features.")
        return

    print(
        f"Most correlated pair: {best_c1} vs {best_c2} "
        f"(Pearson r = {best_r:.4f})"
    )

    # Build per-house coordinates for plotting using pandas filtering
    house_points: Dict[str, Tuple[List[float], List[float]]] = {
        house: ([], []) for house in HOUSES_COLORS.keys()
    }

    # Ensure House column exists
    if "Hogwarts House" not in df.columns:
        print("Column 'Hogwarts House' not found in dataset.")
        return

    # Use numeric data for the chosen best pair
    pair_df = df[["Hogwarts House", best_c1, best_c2]].copy()
    pair_df[best_c1] = pd.to_numeric(pair_df[best_c1], errors="coerce")
    pair_df[best_c2] = pd.to_numeric(pair_df[best_c2], errors="coerce")
    pair_df = pair_df.dropna(subset=[best_c1, best_c2])

    for house in HOUSES_COLORS.keys():
        subset = pair_df[pair_df["Hogwarts House"] == house]
        if subset.empty:
            continue
        x_list = subset[best_c1].tolist()
        y_list = subset[best_c2].tolist()
        house_points[house] = (x_list, y_list)

    # Draw the scatter plot
    plt.figure(figsize=(8, 6))
    for house, color in HOUSES_COLORS.items():
        x_values, y_values = house_points[house]
        if not x_values:
            continue
        plt.scatter(x_values, y_values, s=10, alpha=0.7, label=house, color=color)
    plt.xlabel(best_c1)
    plt.ylabel(best_c2)
    plt.title(
        f"Scatter plot of {best_c1} vs {best_c2}\n"
        f"(most correlated features, r = {best_r:.3f})"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


def main(argv: List[str]) -> None:
    if len(argv) != 2:
        print("Usage: python scatter_plot.py <training dataset>")
        sys.exit(1)
    try:
        plot_best_pair(argv[1])
    except Exception as e:
        print(f"Error plotting best pair: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main(sys.argv)
