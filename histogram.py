import sys
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional

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


HOUSES: List[str] = list(HOUSES_COLORS.keys())


def load_course_scores(path: str) -> Optional[Dict[str, Dict[str, List[float]]]]:
    """
    Read the CSV at `path` using pandas and build a nested dictionary:

        data[course][house] -> list of float scores

    Only numeric scores are kept; missing / non-numeric values are ignored.
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    
    if df.empty:
        return None

    df = df[df["Hogwarts House"].isin(HOUSES)]
    
    if df.empty:
        return None

    data: Dict[str, Dict[str, List[float]]] = {
        course: {house: [] for house in HOUSES}
        for course in COURSE_COLUMNS
    }

    for course in COURSE_COLUMNS:
        if course not in df.columns:
            continue
        for house in HOUSES:
            house_data = df[df["Hogwarts House"] == house][course]
            numeric_values = house_data.dropna().tolist()
            data[course][house] = [v for v in numeric_values if isinstance(v, (int, float))]
    
    return data


def plot_histograms(path: str) -> None:
    """Plot histograms of course scores by house."""
    data = load_course_scores(path)
    if data is None:
        print("No data to plot.")
        return

    num_courses = len(COURSE_COLUMNS)
    ncols = 4
    nrows = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))
    
    # Flatten axes matrix into a simple list for easier indexing
    axes_list = axes.flatten()

    for idx, course in enumerate(COURSE_COLUMNS):
        ax = axes_list[idx]
        # Plot histogram for each house in this course
        for house, color in HOUSES_COLORS.items():
            values = data[course][house]
            if not values:
                continue
            ax.hist(
                values,
                bins=30,  # number of histogram bins
                alpha=0.5,  # transparency so distributions overlap visibly
                label=house,
                color=color,
            )
        ax.set_title(course, fontsize=9)
        ax.tick_params(labelsize=7)
        if idx == 0:
            # Only put a legend on the first subplot
            ax.legend(fontsize=7)

    # Hide any remaining subplots that are not used
    for j in range(num_courses, len(axes_list)):
        fig.delaxes(axes_list[j])

    fig.suptitle(
        "Score distributions per house for each Hogwarts course", fontsize=14
    )

    # Adjust layout to prevent titles/labels from overlapping
    # rect : [left, bottom, right, top] as margins
    # more margin at top for suptitle
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()


def main(argv: List[str]) -> None:
    if len(argv) != 2:
        print("Usage: python histogram.py <training dataset>")
        sys.exit(1)
    try:
        plot_histograms(argv[1])
    except Exception as e:
        print(f"Error plotting histograms: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv)
