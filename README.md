# Breast Cancer Decision Tree Classifier

A from-scratch implementation of decision stumps and a decision tree classifier to diagnose breast cancer from patient medical data. Built without ML libraries to demonstrate understanding of tree construction, entropy calculation, information gain, and pruning.

## Features

- Custom decision tree implementation in pure Python/NumPy
- Entropy and information gain calculations
- Optimal split selection across features and thresholds
- Tree pruning to control overfitting
- Predictions on test data with accuracy tracking
- Tree visualization through readable text output

## Tech Stack

- Python
- NumPy
- Wisconsin Breast Cancer dataset

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
