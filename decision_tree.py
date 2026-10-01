"""Decision tree classifier for breast cancer diagnosis, built from scratch.

Trains a binary decision tree on the Wisconsin Breast Cancer dataset from the
UCI Machine Learning Repository. Each split is chosen by information gain, the
tree is grown until no split improves it, and it is then pruned to a fixed
depth. Both trees are written out as readable if/else rules and used to
classify a held-out set of samples.

Columns of breast-cancer-wisconsin.data, numbered from 1:
     1  sample code number          7  bare nuclei
     2  clump thickness             8  bland chromatin
     3  uniformity of cell size     9  normal nucleoli
     4  uniformity of cell shape   10  mitoses
     5  marginal adhesion          11  class (2 = benign, 4 = malignant)
     6  single epithelial cell size

Columns 2 to 10 are scored from 1 to 10. Rows with a missing value ('?') are
dropped.
"""

import numpy as np

TRAIN_FILE = 'breast-cancer-wisconsin.data'
TEST_FILE = 'test.txt'

# Columns the tree is allowed to split on, using the numbering above.
feature_list = [9, 5, 4, 6, 2, 10]

# Every feature is scored 1 to 10, so these are all the possible cut points.
threshold_list = range(1, 11)

# Depth the full tree is pruned back to.
target_depth = 6


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

with open(TRAIN_FILE, 'r') as f:
    data_raw = [l.strip('\n').split(',') for l in f if '?' not in l]
data = np.array(data_raw).astype(int)

with open(TEST_FILE, 'r') as f:
    test_data = [l.strip('\n').split(',') for l in f if '?' not in l]
test_data = np.array(test_data).astype(int)

total_n2 = np.sum(data[:, -1] == 2)
total_n4 = np.sum(data[:, -1] == 4)


# ---------------------------------------------------------------------------
# Choosing a split
# ---------------------------------------------------------------------------

def entropy(data):
    """Entropy of the class labels, in bits. 0 when every sample agrees."""
    entropy = 0
    count = len(data)
    n2 = np.sum(data[:, -1] == 2)
    n4 = np.sum(data[:, -1] == 4)
    if n2 == 0 or n4 == 0:
        return 0
    else:
        for n in [n2, n4]:
            p = n / count
            entropy += - (p * np.log2(p))
        return entropy


def infogain(data, feature, threshold):
    """Reduction in entropy from splitting on feature <= threshold."""
    count = len(data)
    d1 = data[data[:, feature - 1] <= threshold]
    d2 = data[data[:, feature - 1] > threshold]
    proportion_d1 = len(d1) / count
    proportion_d2 = len(d2) / count
    return entropy(data) - proportion_d1 * entropy(d1) - proportion_d2 * entropy(d2)


def majority_class(data):
    """The more common class label. Ties go to 2."""
    n2 = np.sum(data[:, -1] == 2)
    n4 = np.sum(data[:, -1] == 4)
    return 2 if n2 >= n4 else 4


def get_best_split(data, feature_list, threshold_list):
    """Find the feature and threshold with the highest information gain.

    Return:
        (feature, threshold, left prediction, right prediction) for a split,
        where each prediction is the majority class on that side.
        (class label, None, None, None) when the data should not be split,
        either because it is all one class or because no split helps.
    """
    c = len(data)
    c0 = sum(b[-1] == 2 for b in data)
    if c0 == c: return 2, None, None, None
    if c0 == 0: return 4, None, None, None

    ig = [[infogain(
        data, feature, threshold) for threshold in threshold_list] for feature in feature_list]
    ig = np.array(ig)
    max_ig = max(max(i) for i in ig)

    if max_ig == 0:
        if c0 >= c - c0:
            return 2, None, None, None
        else:
            return 4, None, None, None

    idx = np.unravel_index(np.argmax(ig, axis=None), ig.shape)
    feature, threshold = feature_list[idx[0]], threshold_list[idx[1]]

    dl = data[data[:, feature - 1] <= threshold]
    dr = data[data[:, feature - 1] > threshold]

    return feature, threshold, majority_class(dl), majority_class(dr)


# ---------------------------------------------------------------------------
# Building the tree
# ---------------------------------------------------------------------------

class Node:
    """One split in the tree.

    Samples with feature <= threshold go left, the rest go right. Each side
    has a prediction, used when that side has no child node, and optionally a
    child node (l or r) that splits it further.
    """

    def __init__(self, feature=None, threshold=None, l_prediction=None, r_prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.l_prediction = l_prediction
        self.r_prediction = r_prediction
        self.l = None
        self.r = None


def split(data, node):
    """Split the data into the two sides of a node."""
    feature, threshold = node.feature, node.threshold
    d1 = data[data[:, feature - 1] <= threshold]
    d2 = data[data[:, feature - 1] > threshold]
    return (d1, d2)


def create_tree(data, node, feature_list):
    """Grow the tree below a node, recursively, until no split helps."""
    d1, d2 = split(data, node)
    f1, t1, l1_prediction, r1_prediction = get_best_split(d1, feature_list, threshold_list)
    f2, t2, l2_prediction, r2_prediction = get_best_split(d2, feature_list, threshold_list)

    # When a side should not be split, get_best_split returns its class label.
    if t1 is None:
        node.l_prediction = f1
    else:
        node.l = Node(f1, t1, l1_prediction, r1_prediction)
        create_tree(d1, node.l, feature_list)

    if t2 is None:
        node.r_prediction = f2
    else:
        node.r = Node(f2, t2, l2_prediction, r2_prediction)
        create_tree(d2, node.r, feature_list)


def expand_root(data, feature_list, threshold_list):
    """Build the full tree and return its root."""
    feature, threshold, l_prediction, r_prediction = get_best_split(
        data, feature_list, threshold_list)
    root = Node(feature, threshold, l_prediction, r_prediction)
    create_tree(data, root, feature_list)
    return root


def max_depth(node):
    """Number of nodes on the longest path from this node down."""
    if node is None:
        return 0
    return max(max_depth(node.l), max_depth(node.r)) + 1


def prune(node, depth):
    """Cut the tree to the given depth.

    Nodes at the cutoff lose their children and fall back on their majority
    class predictions.
    """
    if depth == 1:
        node.l = None
        node.r = None
    else:
        if node.l is not None:
            prune(node.l, depth - 1)
        if node.r is not None:
            prune(node.r, depth - 1)


# ---------------------------------------------------------------------------
# Using the tree
# ---------------------------------------------------------------------------

def print_tree(node, f, prefix=''):
    """Write the tree to file f as nested if/else rules."""
    feature = node.feature
    threshold = node.threshold
    l_prediction = node.l_prediction
    r_prediction = node.r_prediction
    l = node.l
    r = node.r

    if l is None:
        f.write(prefix + 'if (x' + str(feature) + ' <= ' + str(threshold) + ') return ' + str(l_prediction) + '\n')
    else:
        f.write(prefix + 'if (x' + str(feature) + ' <= ' + str(threshold) + ')\n')
        print_tree(l, f, prefix + ' ')

    if r is None:
        f.write(prefix + 'else return ' + str(r_prediction) + '\n')
    else:
        f.write(prefix + 'else\n')
        print_tree(r, f, prefix + ' ')


def tree_prediction(node, x):
    """Predict the class of sample x by following the tree down from node."""
    if x[node.feature - 1] <= node.threshold:
        if node.l is None:
            return node.l_prediction
        else:
            return tree_prediction(node.l, x)
    else:
        if node.r is None:
            return node.r_prediction
        else:
            return tree_prediction(node.r, x)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

print(f"Training samples: {total_n2} benign, {total_n4} malignant")

# Full tree.
root = expand_root(data, feature_list, threshold_list)
print(f"Full tree depth: {max_depth(root)}")

with open('tree.txt', 'w') as f:
    print_tree(root, f)

predictions = [str(tree_prediction(root, x)) for x in test_data]
print("Predictions from the full tree:")
print(', '.join(predictions))

# Pruned tree.
prune(root, depth=target_depth)
print(f"Pruned tree depth: {max_depth(root)}")

with open('pruned_tree.txt', 'w') as f:
    print_tree(root, f)

predictions = [str(tree_prediction(root, x)) for x in test_data]
print("Predictions from the pruned tree:")
print(', '.join(predictions))
