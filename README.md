# Hogwarts House Prediction - Logistic Regression

A machine learning project that predicts which Hogwarts house a student belongs to based on their course grades using logistic regression with multiple gradient descent optimization algorithms.

## Project Overview

This project implements a one-vs-all logistic regression classifier to predict Hogwarts houses (Gryffindor, Hufflepuff, Ravenclaw, Slytherin) from student academic performance data. The model uses 13 course features including Arithmancy, Astronomy, Herbology, Defense Against the Dark Arts, and more.

## Features

- **Comprehensive Data Analysis**: Statistical description, histogram, scatter plot, and pair plot
- **Multiple Gradient Descent Algorithms**: Choose between batch, stochastic (SGD), or mini-batch gradient descent
- **Model Training & Prediction**: Train logistic regression models and make predictions on test data
- **Model Evaluation**: Calculate accuracy scores against ground truth labels

## Requirements

- Python 3.x
- pandas
- matplotlib (for visualization scripts)

## Installation

```bash
# Clone or download the project
cd dslr

# Install required packages (if not already installed)
pip install pandas matplotlib
```

## Project Structure

```
dslr/
├── logreg_train.py          # Train logistic regression model
├── logreg_predict.py        # Make predictions on test data
├── describe.py              # Generate statistical summary of dataset
├── evaluate.py              # Evaluate prediction accuracy
├── histogram.py             # Generate histograms for features
├── scatter_plot.py          # Generate scatter plots
├── pair_plot.py             # Generate pair plots
├── utils.py                 # Utility functions (CSV reading, stats)
├── describe_helpers.py       # Statistical helper functions
├── dataset_train.csv         # Training dataset
├── dataset_test.csv         # Test dataset
├── dataset_truth.csv         # Ground truth labels for evaluation
└── weights.json             # Saved model weights (generated after training)
```

## Usage

### 1. Data Analysis

#### Generate Statistical Summary
```bash
python describe.py dataset_train.csv
```

This displays comprehensive statistics for all numerical columns including:
- Count, missing values
- Mean, standard deviation, variance
- Min, 25th percentile, median, 75th percentile, max
- Range

#### Generate Visualizations
```bash
# Histogram
python histogram.py dataset_train.csv

# Scatter plot
python scatter_plot.py dataset_train.csv

# Pair plot
python pair_plot.py dataset_train.csv
```

### 2. Model Training

Train a logistic regression model with your choice of gradient descent algorithm:

```bash
python logreg_train.py dataset_train.csv
```

The program will prompt you to select a training algorithm:
- **batch** (default): Full batch gradient descent - uses entire dataset per iteration
- **sgd**: Stochastic gradient descent - uses one example per iteration
- **minibatch**: Mini-batch gradient descent - uses a subset of examples per iteration

For mini-batch, you'll also be prompted to specify the batch size (default: 32).

**Example:**
```
Select training algorithm [batch | sgd | minibatch]: minibatch
Mini-batch size (default 32): 64
```

The trained model will be saved to `weights.json`.

### 3. Making Predictions

Generate predictions on test data:

```bash
python logreg_predict.py dataset_test.csv weights.json
```

Predictions will be saved to `houses.csv` in the format:
```csv
Index,Hogwarts House
0,Gryffindor
1,Slytherin
...
```

### 4. Evaluating Model Performance

Compare predictions against ground truth:

```bash
python evaluate.py houses.csv dataset_truth.csv
```

This will output the accuracy score and the number of correct predictions.

## Algorithm Details

### Batch Gradient Descent
- Uses the entire training dataset to compute gradients
- More stable convergence but slower per iteration
- Best for smaller datasets

### Stochastic Gradient Descent (SGD)
- Uses one randomly selected example per gradient step
- Faster per iteration, more noisy updates
- Good for large datasets, may require more iterations

### Mini-Batch Gradient Descent
- Uses a small random subset (batch) of examples per step
- Balances stability and speed
- Most commonly used in practice
- Batch size is configurable (typically 16-128)

## Model Architecture

The model uses **one-vs-all** classification:
- Trains 4 separate binary classifiers (one per house)
- Each classifier predicts: "this house" vs "not this house"
- Final prediction selects the house with highest probability

### Features Used
1. Arithmancy
2. Astronomy
3. Herbology
4. Defense Against the Dark Arts
5. Divination
6. Muggle Studies
7. Ancient Runes
8. History of Magic
9. Transfiguration
10. Potions
11. Care of Magical Creatures
12. Charms
13. Flying

### Data Preprocessing
- Missing values are imputed with feature means
- Features are standardized (z-score normalization)
- Standardization parameters are saved with the model for consistent preprocessing during prediction

## File Formats

### Training/Test CSV Format
```csv
Index,Hogwarts House,Arithmancy,Astronomy,...
0,Gryffindor,85.5,92.0,...
```

### Weights JSON Format
```json
{
    "features": ["Arithmancy", "Astronomy", ...],
    "means": [mean1, mean2, ...],
    "stds": [std1, std2, ...],
    "houses": ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"],
    "weights": [[bias, w1, w2, ...], ...]
}
```
