import sys
from typing import Dict, List

import matplotlib.pyplot as plt

from utils import is_float, read_dataset

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


def build_clean_rows(path: str) -> List[Dict[str, float | None]]:
    """Read the CSV and normalize each row into a structure convenient for plotting."""
    header, rows = read_dataset(path)
    if not header:
        return []

    clean_rows = []
    for row in rows:
        house = row.get("Hogwarts House", "")
        if house not in HOUSES_COLORS:
            continue
        parsed = {"Hogwarts House": house}
        for col in COURSE_COLUMNS:
            val = row.get(col, "").strip()
            if val == "" or not is_float(val):
                parsed[col] = None
            else:
                parsed[col] = float(val)
        clean_rows.append(parsed)
    return clean_rows


def plot_pair_matrix(path: str) -> None:
    """
    Draw a scatter plot matrix of all course features:

    - Diagonal cells show the feature name.
    - Off-diagonal cells show scatter plots colored by house.
    """
    rows = build_clean_rows(path)
    if not rows:
        print("No data to plot.")
        return

    n_features = len(COURSE_COLUMNS)
    fig, axes = plt.subplots(
        n_features, n_features, figsize=(2.5 * n_features, 2.5 * n_features)
    )

    for i, col_y in enumerate(COURSE_COLUMNS):
        for j, col_x in enumerate(COURSE_COLUMNS):
            ax = axes[i, j]
            if i == j:
                # Diagonal: show the feature name in the cell center
                ax.text(
                    0.5,
                    0.5,
                    col_x,
                    ha="center",
                    va="center",
                    fontsize=8,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Off-diagonal: scatter colored by house
            for house, color in HOUSES_COLORS.items():
                x_values: List[float] = []
                y_values: List[float] = []
                for row in rows:
                    if row["Hogwarts House"] != house:
                        continue
                    x_val = row[col_x]
                    y_val = row[col_y]
                    if x_val is None or y_val is None:
                        continue
                    x_values.append(x_val)
                    y_values.append(y_val)
                if x_values:
                    ax.scatter(x_values, y_values, s=5, alpha=0.6, color=color)

            # Only label bottom row and leftmost column to avoid clutter
            if i == n_features - 1:
                ax.set_xlabel(col_x, fontsize=6)
            else:
                ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(col_y, fontsize=6)
            else:
                ax.set_yticks([])

    # Add a single legend outside the grid to explain color coding
    handles = []
    labels = []
    for house, color in HOUSES_COLORS.items():
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=5,
            )
        )
        labels.append(house)
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(1, 1),
        fontsize=8,
    )

    fig.suptitle("Pair plot of Hogwarts course scores", fontsize=14)
    # Adjust layout to fit everything (titles, labels, legend) nicely
    fig.tight_layout(rect=[0, 0, 0.98, 0.97])
    plt.show()


def main(argv: List[str]) -> None:
    if len(argv) != 2:
        print("Usage: python pair_plot.py <dataset_train.csv>")
        sys.exit(1)
    try:
        plot_pair_matrix(argv[1])
    except Exception as e:
        print(f"Error plotting pair matrix: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv)


