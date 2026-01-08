import csv
import describe_helpers as dh
from typing import Dict, List, Tuple


def is_float(value: str) -> bool:
    """True if value can be converted to a float, False otherwise."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute basic descriptive statistics for a list of numbers."""
    n = len(values)
    stats: Dict[str, float] = {}

    if n == 0:
        return stats

    sorted_vals = sorted(values)

    stats["count"] = float(n)
    stats["mean"] = sum(sorted_vals) / n
    stats["std"] = (sum((x - stats["mean"]) ** 2 for x in sorted_vals) / n) ** 0.5
    stats["min"] = sorted_vals[0]
    q1, q3 = dh.quartile_(values)
    stats["25%"] = q1
    stats["50%"] = dh.median_(values)
    stats["75%"] = q3
    stats["max"] = sorted_vals[-1]

    return stats


def read_dataset(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read a CSV file and return the header and rows."""
    with open(path, newline="", encoding="utf-8") as f:
        try:
            reader = csv.reader(f)
            header = next(reader)
        except Exception:
            return [], []

        rows: List[Dict[str, str]] = []
        for row in reader:
            if not row or len(row) != len(header):
                continue
            # append a dictionary containing the header as keys and the rows values as values
            rows.append({header[i]: row[i] for i in range(len(header))})
    return header, rows
