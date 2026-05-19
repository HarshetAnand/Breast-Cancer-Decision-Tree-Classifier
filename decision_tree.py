import numpy as np

threshold_list = range(1, 11)

# Adjust the following parameters by yourself
# The parameters I had were as follows:
part_one_feature = [4]
feature_list = [9, 5, 4, 6, 2, 10]
target_depth = 6

# Now, we need to open the file named "breast-cancer-wisconsin.data" in read mode. We then assign it to variable f
# We add commands in order to remove any trailing newline characters from the line and to split the line into a list of
# values based on the CSV format.
with open('breast-cancer-wisconsin.data', 'r') as f:
    data_raw = [l.strip('\n').split(',') for l in f if '?' not in l]
data = np.array(data_raw).astype(int)  # This is the training data used in the code

# The following function calculates the entropy of a dataset based on the labels present in the last column of the
# data list. Entropy is a measure of uncertainty or randomness in the dataset.
def entropy(data):
    entropy = 0  # Initialize the entropy variable to store the calculated entropy value.
    count = len(data)  # Get the total number of instances in the 'data' list.
    n2 = np.sum(data[:, -1] == 2)  # number of k1
    n4 = np.sum(data[:, -1] == 4)  # number of k2
    if n2 == 0 or n4 == 0:
        # If either the count of k1 (label '2') or k2 (label '4') is zero,
        # it means there is only one class of instances, so entropy is zero.
        return 0
    else:
        # If both k1 and k2 instances are present, calculate entropy for each class.
        for n in [n2, n4]:
            p = n / count # Calculate the probability of the class.
            entropy += - (p * np.log2(p)) # Update the entropy by adding the contribution of each class.
        return entropy # Return the calculated entropy value.

# We need to print the total number of 2s and 4s from the dataset.
total_n2 = np.sum(data[:, -1] == 2)
total_n4 = np.sum(data[:, -1] == 4)

# print the total number of 2s and 4s
print(total_n2)
print(total_n4)

# The function infogain takes three parameters: data (the input dataset),
# feature (the index of the feature to split on), and threshold (the threshold value for the split).
# It calculates the total number of data points in the dataset, which will be used later for calculating proportions.
def infogain(data, feature, threshold):
    count = len(data)
    d1 = data[data[:, feature - 1] <= threshold]
    d2 = data[data[:, feature - 1] > threshold]
    # The proportions of data points in each subset are calculated by dividing the
    # number of data points in each subset by the total count of data points.
    proportion_d1 = len(d1) / count
    proportion_d2 = len(d2) / count
    return entropy(data) - proportion_d1 * entropy(d1) - proportion_d2 * entropy(d2)


# The following helps us understand the "best split". We start by calculating the total number of data points.
# Then we count the number of data points with class level 2.
def get_best_split(data, feature_list, threshold_list):
    c = len(data)
    c0 = sum(b[-1] == 2 for b in data)
    # If all data points have class label 2, return class label 2 and no split information.
    if c0 == c: return 2, None, None, None
    if c0 == 0: return 4, None, None, None
    # Calculate the information gain for all possible combinations of features and thresholds.
    ig = [[infogain(
        data, feature, threshold) for threshold in threshold_list] for feature in feature_list]
    # Convert the information gain list into a numpy array for easier manipulation.
    ig = np.array(ig)
    # Find the maximum information gain value.
    max_ig = max(max(i) for i in ig)

    # If the maximum information gain is 0, it means no further splitting will improve the classification.
    # In such cases, return the majority class label as the prediction for both branches.
    if max_ig == 0:
        if c0 >= c - c0:
            return 2, None, None, None
        else:
            return 4, None, None, None
    # Find the index of the maximum information gain in the 2D array.
    idx = np.unravel_index(np.argmax(ig, axis=None), ig.shape)
    feature, threshold = feature_list[idx[0]], threshold_list[idx[1]]

    # Split the dataset into two subsets based on the selected feature and threshold.
    dl = data[data[:, feature - 1] <= threshold]  # Subset with values less than or equal to the threshold.

    # Calculate the number of data points in each subset with class label 2 and 4.
    dl_n2 = np.sum(dl[:, -1] == 2)
    dl_n2 = np.sum(dl[:, -1] == 2)
    dl_n4 = np.sum(dl[:, -1] == 4)

    # Check if the number of data points with class label 2 in the left subset
    # is greater than or equal to the number of data points with class label 4.
    if dl_n2 >= dl_n4:
        dl_prediction = 2
    else:
        dl_prediction = 4
    dr = data[data[:, feature - 1] > threshold] # Subset with values greater than the threshold.
    dr_n2 = np.sum(dr[:, -1] == 2)
    dr_n4 = np.sum(dr[:, -1] == 4)

    # Check if the number of data points with class label 2 in the right subset is greater than or equal
    # to the number of data points with class label 4.
    if dr_n2 >= dr_n4:
        if dr_n2 >= dr_n4:
            dr_prediction = 2
    else:
        dr_prediction = 4

    # Return the selected feature, threshold, and predictions for the left and right branches.
    return feature, threshold, dl_prediction, dr_prediction

# def get_best_split(data, feature_list, threshold_list):
#     c = len(data)
#     c0 = sum(b[-1] == 2 for b in data)
#     if c0 == c: return 2, None, None, None
#     if c0 == 0: return 4, None, None, None
#     ig = [[infogain(
#         data, feature, threshold) for threshold in threshold_list] for feature in feature_list]
#     ig = np.array(ig)
#     max_ig = max(max(i) for i in ig)
#     if max_ig == 0:
#         if c0 >= c - c0:
#             return 2, None, None, None
#         else:
#             return 4, None, None, None

#     idx = np.unravel_index(np.argmax(ig, axis=None), ig.shape)
#     feature, threshold = feature_list[idx[0]], threshold_list[idx[1]]

#     # data below threshold
#     dl = data[data[:, feature - 1] <= threshold]
#     dl_n2 = np.sum(dl[:, -1] == 2)  # positive instances below threshold
#     dl_n4 = np.sum(dl[:, -1] == 4)  # negative instances below threshold

#     # data above threshold
#     dr = data[data[:, feature - 1] > threshold]
#     dr_n2 = np.sum(dr[:, -1] == 2)  # positive instances above threshold
#     dr_n4 = np.sum(dr[:, -1] == 4)  # negative instances above threshold

#     # print the results
#     print(f"For feature {feature} and threshold {threshold}:")
#     print(f"Below threshold: {dl_n2} positive instances, {dl_n4} negative instances")
#     print(f"Above threshold: {dr_n2} positive instances, {dr_n4} negative instances")


# The following class is the node class where we initialize all parameters.
class Node:
    # Initialize a Node object representing a node in a decision tree.
    def __init__(self, feature=None, threshold=None, l_prediction=None, r_prediction=None):

        # Feature: The index of the feature used for splitting at this node.
        # If this node is a leaf node (has no children), it will be set to None.
        # Threshold: The threshold value used to split the data based on the feature.
        # If this node is a leaf node, it will be set to None.

        self.feature = feature
        self.threshold = threshold

        # Left Prediction: The class label prediction for the left branch of the tree.
        # If this node represents a leaf node, this will be the predicted class label for the left branch.
        # Right Prediction: The class label prediction for the right branch of the tree.
        # If this node represents a leaf node, this will be the predicted class label for the right branch.
        self.l_prediction = l_prediction
        self.r_prediction = r_prediction

        # Left Child: A reference to the left child node (subtree) in the decision tree.
        # Initially set to None, as the node might be a leaf node with no children.
        # Right Child: A reference to the right child node (subtree) in the decision tree.
        # Initially set to None, as the node might be a leaf node with no children.

        self.l = None
        self.r = None

        # Correct: A variable used to store the number of correctly classified instances in the node's subtree.
        # This is typically used during the construction of the decision tree and can be useful for pruning or analysis.
        self.correct = 0


# The following function splits the input data into two subsets based on a given node's features.
# Extract the feature and threshold from the node to be used for splitting.
# We create two subsets - one containing data points with values less than or equal to the threshold,
# and one containing data points with values greater than the threshold for the selected feature.
def split(data, node):
    # split the data into two parts
    feature, threshold = node.feature, node.threshold
    d1 = data[data[:, feature - 1] <= threshold]
    d2 = data[data[:, feature - 1] > threshold]
    return (d1, d2)


# With the following function, we define a recursive function that constructs a decision tree starting from the given
# node and using the given input data.
def create_tree(data, node, feature_list):

    # The data is split into two subsets d1 and d2 based on the feature and threshold of the current node.
    # This split divides the data into two branches, left and right.
    d1, d2 = split(data, node)
    f1, t1, l1_prediction, r1_prediction = get_best_split(d1, feature_list, threshold_list)
    f2, t2, l2_prediction, r2_prediction = get_best_split(d2, feature_list, threshold_list)

    # If t1 is None, it means no further splitting improves the classification for the left subset.
    # In this case, set the left prediction directly to the majority class label in d1.
    if t1 == None:
        node.l_pre = f1
    else:
        node.l = Node(f1, t1, l1_prediction, r1_prediction)
        create_tree(d1, node.l, feature_list)

    # If t2 is None, it means no further splitting improves the classification for the right subset.
    # In this case, set the right prediction directly to the majority class label in d2.
    if t2 == None:
        node.r_pre = f2
    else:
        node.r = Node(f2, t2, l2_prediction, r2_prediction)
        # Recursively call create_tree with d2 and the new right node to build the right branch of the tree.
        create_tree(d2, node.r, feature_list)

# The following function is a recursive function that calculates the maximum depth of a decision tree. The depth of a
# tree represents the maximum number of edges between the tree's root and any of its leaf nodes.
def maxDepth(node):
    # If the node is None, it means it's a leaf node or an empty tree with no children.
    # In such cases, the depth of the current subtree is 0.
    if node is None:
        return 0;

    else:
        # Recursively calculate the maximum depth of the left subtree.
        left_depth = maxDepth(node.l)
        # Recursively calculate the maximum depth of the right subtree.
        right_depth = maxDepth(node.r)

        # Return the maximum depth of the current subtree by taking the maximum of the left and right subtrees,
        # and adding 1 to account for the current node.
        return max(left_depth, right_depth) + 1


# The expand_root function is designed to expand the root node of a decision tree
# by finding the best split for it based on the provided data (data), feature list (feature_list),
# and threshold list (threshold_list).
def expand_root(data, feature_list, threshold_list):
    # Expands the root node of a decision tree by finding the best split based on the provided data.
    # Get the best split for the root node using the 'get_best_split' function,
    # which returns the optimal feature, threshold, and the left and right branches.
    feature, threshold, dl, dr = get_best_split(
        data, feature_list, threshold_list)
    root = Node(feature, threshold)
    # first split
    data1, data2 = split(data, root)
    create_tree(data, root, feature_list)
    return root

# Get the best split for the root node of the decision tree using the provided data,
# feature list, and threshold list.
feature, threshold, dl, dr = get_best_split(
    data, feature_list, threshold_list)

# Expand the root node of the decision tree using the 'expand_root' function,
# which constructs the decision tree based on the best split obtained from the data.
root = expand_root(data, feature_list, threshold_list)

# Calculate the maximum depth (height) of the decision tree rooted at 'root'
# using the 'maxDepth' function. The result represents the longest path from the root node to a leaf node.
maxDepth(root)


# The following lines of code are specific to Q5 and Q6
# The print_tree function is designed to recursively print the decision tree rooted at the given node to the provided
# file f.
# It prints the decision rules and predictions for each node in the tree.

def print_tree(node, f, prefix=''):
    # Recursively prints the decision tree rooted at 'node' to a file 'f' with the provided prefix.
    feature = node.feature
    threshold = node.threshold
    # Extract information from the current node for printing purposes.
    l_prediction = node.l_prediction
    r_prediction = node.r_prediction
    l = node.l
    r = node.r

    # Check if the left child node is None. If so, it means the current node is a leaf node.
    # In this case, print the decision rule for the leaf node and its prediction value.
    if l == None:
        f.write(prefix + 'if (x' + str(feature) + ') <= ' + str(threshold) + ') return ' + str(l_prediction) + '\n')
    else:
        # If the left child node exists, print the decision rule for this node and recursively call
        # the 'print_tree' function for the left subtree.
        f.write(prefix + 'if (x' + str(feature) + ') <= ' + str(threshold) + ')\n')
        print_tree(l, f, prefix + ' ')

    # Check if the right child node is None. If so, it means the current node is a leaf node on the right branch.
    # In this case, print the decision rule for the leaf node on the right branch and its prediction value.
    if r == None:
        f.write(prefix + 'else return ' + str(r_prediction) + '\n')
    else:
        # If the right child node exists, print the decision rule for this node and recursively call
        # the 'print_tree' function for the right subtree.
        f.write(prefix + 'else\n')
        print_tree(r, f, prefix + ' ')

# Open the 'test.txt' file in read mode ('r') and read its contents line by line.
# The file is expected to contain test data for the decision tree.
with open('test.txt', 'r') as f:
    test_data = [l.strip('\n').split(',') for l in f if '?' not in l]

# Open the 'tree.txt' file in write mode ('w') to write the decision tree's content.
# The 'tree.txt' file will contain the printed decision tree.
with open('tree.txt', 'w') as f:
    # Print the decision tree rooted at 'root' to the file 'tree.txt' using the 'print_tree' function.
    print_tree(root, f)
# Convert the test data from a list of strings to a NumPy array of integers for processing.
test_data = np.array(test_data).astype(int)  # test


# The following lines of code are specific for Q7 and Q9
# The tree_prediction function is used to predict the class label for a given data point 'x' using the decision tree
# rooted at the given node. The function begins by extracting the necessary information from the current node
# (feature, threshold, left and right predictions, left and right child nodes).

def tree_prediction(node, x):
    # Predicts the class label for a given data point 'x' using the decision tree rooted at 'node'.
    # Extract information from the current node for prediction purposes.
    feature = node.feature
    threshold = node.threshold
    l_prediction = node.l_prediction
    r_prediction = node.r_prediction
    l = node.l
    r = node.r
    if x[feature - 1] <= threshold:
        if l_prediction == x[-1]:
            node.correct += 1

        if l == None:
            return l_prediction
        else:
            return tree_prediction(l, x)
    else:
        if r_prediction == x[-1]:
            node.correct += 1
        if r == None:
            return r_prediction
        else:
            return tree_prediction(r, x)

# The 'tree_prediction' function is called for each data point 'x' in the 'test_data' list,
# and the predicted class labels are stored as strings in the 'predictions' list.
predictions = [str(tree_prediction(root, x)) for x in test_data]
predictions_str = ', '.join(predictions)
# Print the comma-separated string containing the predicted class labels for the test data.
print(predictions_str)


# The following lines of code are used for Q8 of the project.
# The prune function prunes the decision tree rooted at the given node to a specified depth (depth).
# Pruning involves reducing the depth of the tree by removing some branches (subtrees),
# effectively converting certain nodes into leaf nodes.
def prune(node, depth):
    # Prunes the decision tree rooted at 'node' to a specified 'depth'.

    # If the 'depth' parameter is 1, it means the current 'node' is at the desired pruning depth.
    # In this case, set both left and right child nodes to None, effectively converting the node into a leaf node.
    if depth == 1:
        node.l = None
        node.r = None
    # If the 'depth' is greater than 1, recursively prune the left and right subtrees.
    # Check if the left child node exists (is not None) and recursively prune it to the specified depth - 1.
    # Check if the right child node exists (is not None) and recursively prune it to the specified depth - 1.
    else:
        if node.l != None:
            prune(node.l, depth - 1)
        if node.r != None:
            prune(node.r, depth - 1)

# Prune the 'root' of the decision tree to a specified 'target_depth'.
prune(root, depth=target_depth)

# Save the pruned tree to a file named 'pruned_tree.txt' for further analysis or usage.
with open('pruned_tree.txt', 'w') as f:
    print_tree(root, f)

# Prune the 'root' of the decision tree again to ensure it remains at the desired 'target_depth'.
# Generate predictions for the test data using the pruned decision tree and convert them to strings.
# Join the predictions into a comma-separated string for easy printing or further processing.
# Print the predictions for the test data.
prune(root, depth=target_depth)
predictions = [str(tree_prediction(root, x)) for x in test_data]
predictions_str = ', '.join(predictions)
print(predictions_str)
