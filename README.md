# Breast Cancer Decision Tree Classifier

A from-scratch implementation of a decision tree classifier to diagnose breast cancer from patient medical data. Built without ML libraries to demonstrate understanding of tree construction, entropy calculation, information gain, and pruning.

## Features

- Custom decision tree implementation in pure Python/NumPy
- Entropy and information gain calculations
- Optimal split selection across features and thresholds
- Tree pruning to control overfitting
- Predictions on held-out test samples from both the full and pruned trees
- Tree visualization through readable text output

## Tech Stack

- Python
- NumPy
- Wisconsin Breast Cancer dataset

## Data

The script expects two files in the project folder:

- `breast-cancer-wisconsin.data`: the Wisconsin Breast Cancer (Original) dataset from the UCI Machine Learning Repository. Each row is a sample ID, nine features scored from 1 to 10, and a class label (2 = benign, 4 = malignant).
- `test.txt`: samples to classify, in the same comma-separated format.

Rows with a missing value are dropped. The script writes the full tree to `tree.txt` and the pruned tree to `pruned_tree.txt` as nested if/else rules.

By default the tree splits on six of the nine features and is pruned to a depth of 6. Both are set at the top of `decision_tree.py`.

## Implementation Details

- **Entropy calculation:** Measures dataset uncertainty based on class distribution
- **Information gain:** Identifies optimal splits by comparing entropy before and after splitting
- **Recursive tree construction:** Builds tree by greedily selecting best feature and threshold at each node
- **Pruning:** Limits tree depth to prevent overfitting
- **Binary classification:** Predicts benign (class 2) vs malignant (class 4)

## Key Concepts Demonstrated

- Decision tree construction from scratch
- Entropy and information theory
- Recursive tree algorithms
- Tree pruning and overfitting prevention
- Binary classification on real medical data
- Object-oriented design with Node class
