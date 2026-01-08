import sys
from typing import Dict, List, Tuple
from utils import read_dataset


def load_predictions(path: str) -> Dict[int, str]:
    """
    Load predictions from a CSV file and return a dictionary mapping Index -> House.
    
    Args:
        path: Path to the CSV file (e.g., houses.csv or dataset_truth.csv)
    
    Returns:
        Dictionary mapping index (int) to house name (str)
    """
    header, rows = read_dataset(path)
    if not header:
        raise ValueError(f"Empty file: {path}")
    
    predictions: Dict[int, str] = {}
    for row in rows:
        index_str = row.get("Index", "").strip()
        house = row.get("Hogwarts House", "").strip()
        
        if index_str == "" or house == "":
            continue
        
        try:
            idx = int(index_str)
            predictions[idx] = house
        except ValueError:
            continue
    
    return predictions


def calculate_accuracy(predicted: Dict[int, str], truth: Dict[int, str]) -> float:
    """
    Calculate accuracy by comparing predicted houses to ground truth.
    
    Args:
        predicted: Dictionary mapping index -> predicted house
        truth: Dictionary mapping index -> true house
    
    Returns:
        Accuracy score between 0.0 and 1.0
    """
    if not predicted:
        return 0.0
    
    # Find common indices (predictions that exist in both)
    common_indices = set(predicted.keys()) & set(truth.keys())
    
    if not common_indices:
        return 0.0
    
    correct = 0
    total = len(common_indices)
    
    for idx in common_indices:
        if predicted[idx] == truth[idx]:
            correct += 1
    
    return correct / total if total > 0 else 0.0


def main(argv: List[str]) -> None:
    if len(argv) != 3:
        print("Usage: python evaluate.py <predictions.csv> <truth.csv>")
        print("Example: python evaluate.py houses.csv dataset_truth.csv")
        sys.exit(1)
    
    predictions_path = argv[1]
    truth_path = argv[2]
    
    try:
        predicted = load_predictions(predictions_path)
        truth = load_predictions(truth_path)
        
        if not predicted:
            print(f"Error: No predictions found in {predictions_path}")
            sys.exit(1)
        
        if not truth:
            print(f"Error: No ground truth found in {truth_path}")
            sys.exit(1)
        
        accuracy = calculate_accuracy(predicted, truth)
        
        # Count matches for detailed output
        common_indices = set(predicted.keys()) & set(truth.keys())
        correct = sum(1 for idx in common_indices if predicted[idx] == truth[idx])
        total = len(common_indices)
        
        print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv)

