import sys
import math
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

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
    stats_x = compute_stats(x)  # contains mean and std for X values
    stats_y = compute_stats(y)  # contains mean and std for Y values
    mean_x = stats_x["mean"]  # mean of X
    mean_y = stats_y["mean"]  # mean of Y
    std_x = stats_x["std"]  # standard deviation of X
    std_y = stats_y["std"]  # standard deviation of Y
    if std_x == 0 or std_y == 0:
        # If one std is 0 → correlation is undefined,
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
    Find the pair of features with the largest absolute Pearson correlation,
    print the pair and its correlation, and display a scatter plot colored
    by house.
    """
    header, rows = read_dataset(path)
    if not header:
        print("Empty dataset.")
        return

    # Best feature pair and correlation score found so far
    best_c1 = ""
    best_c2 = ""
    best_r = 0.0

    # Prepare per-feature lists with NaNs for missing values, aligned per row index
    values_per_feature: Dict[str, List[float]] = {c: [] for c in COURSE_COLUMNS}
    for row in rows:
        for c in COURSE_COLUMNS:
            val = row.get(c, "").strip()
            if val == "" or not is_float(val):
                # Missing values are stored as NaN to keep alignment across features
                values_per_feature[c].append(float("nan"))
            else:
                values_per_feature[c].append(float(val))

    m = len(rows)  # number of observations
    # Try all pairs of course features
    for i in range(len(COURSE_COLUMNS)):
        for j in range(i + 1, len(COURSE_COLUMNS)):
            x_values: List[float] = []
            y_values: List[float] = []
            # Collect only rows where both features are present
            for k in range(m):
                if not math.isnan(values_per_feature[COURSE_COLUMNS[i]][k]) \
                    and not math.isnan(values_per_feature[COURSE_COLUMNS[j]][k]):
                    x_values.append(values_per_feature[COURSE_COLUMNS[i]][k])
                    y_values.append(values_per_feature[COURSE_COLUMNS[j]][k])
            if len(x_values) < 2:
                # Need at least two points to calculate a correlation
                continue
            r = compute_pearson(x_values, y_values)
            # Keep the pair with the strongest linear relationship
            if r > best_r:
                best_c1, best_c2, best_r = COURSE_COLUMNS[i], COURSE_COLUMNS[j], r

    if best_c1 == "" or best_c2 == "":
        print("Could not find a suitable pair of features.")
        return

    print(
        f"Most correlated pair: {best_c1} vs {best_c2} "
        f"(Pearson r = {best_r:.4f})"
    )

    # Build per-house coordinates for plotting
    house_points: Dict[str, Tuple[List[float], List[float]]] = {
        house: ([], []) for house in HOUSES_COLORS.keys()
    }
    for row in rows:
        house = row.get("Hogwarts House", "")
        if house not in HOUSES_COLORS:
            continue
        x_val = row.get(best_c1, "").strip()
        y_val = row.get(best_c2, "").strip()
        if (
            x_val == ""
            or y_val == ""
            or not is_float(x_val)
            or not is_float(y_val)
        ):
            continue
        x_value, y_value = float(x_val), float(y_val)
        x_values, y_values = house_points[house]
        x_values.append(x_value)
        y_values.append(y_value)

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
    plot_best_pair(argv[1])


if __name__ == "__main__":
    main(sys.argv)
