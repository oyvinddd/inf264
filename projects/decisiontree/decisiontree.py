import numpy as np
import statistics as stat
import random as rand
import math
import sys

# 1.1 Implement a decision tree learning algorithm from scratch

class Data:
    def __init__(self, index, mean, ml):
        self.index = index
        self.mean = mean
        # majority label
        self.ml = ml
    def __str__(self):
        return "index: %s, mean: %s, mcl: %s" % (self.index, self.mean, self.ml)

class Node:
    def __init__(self, label=None, data=None, parent=None, left=None, right=None):
        self.label = label
        self.data = data
        self.parent = parent
        self.left = left
        self.right = right
    def __str__(self):
        if self.is_leaf():
            if self.label == 0:
                return 'No'
            return 'Yes'
        return str(self.label)
    def is_leaf(self):
        return self.left == None and self.right == None

class DecisionTree:

    def __init__(self):
        self.tree = Node()

    def learn(self, X, y, impurity_measure='entropy', prune=False):
        # store the impurity measure on the instance for later use
        self.impurity_measure = impurity_measure
        # split data into training and pruning (validation) sets
        X_train, y_train, X_val, y_val = self._training_validation_split(X, y, prune)
        # call the (recursive) method to build a binary decision tree from the training data
        self._build_tree(X_train, y_train, self.tree)
        # if pruning is enabled, do reduced-error pruning on the tree
        if prune:
            self._prune_tree(X_val, y_val, self.tree)
    
    def predict(self, x):
        return self._traverse_tree(x, self.tree)

    def _traverse_tree(self, x, tree):
        # if we have reached a leaf, we are done
        if tree.is_leaf():
            return tree.label
        if x[tree.data.index] <= tree.data.mean:
            # if feature value is less than the mean, follow the left child node
            return self._traverse_tree(x, tree.left)
        # if feature is larger than the mean, follow the right child node
        return self._traverse_tree(x, tree.right)

    def _build_tree(self, x, y, node):
        if self._has_same_label(y):
            node.label = y[0]
            return
        if self._has_identical_features(x, y):
            label = self._most_common_label(y)
            node.label = label
            return
        # most common label (used for pruning)
        ml = self._majority_label(y)
        # get index (and mean value for later use) for the feature with the optimal split
        index, mean = self._optimal_split(x, y)
        # split the data into two separate sets (based on the feature)
        x1, y1, x2, y2 = self._split_data(x, y, index)
        # use feature index as the node name and the mean as the data
        node.label = index
        node.data = Data(index, mean, ml)
        # call the method recursively with the smaller data sets
        if len(y1) > 0:
            node.left = Node(parent=node)
            self._build_tree(x1, y1, node.left)
        if len(y2) > 0:
            node.right = Node(parent=node)
            self._build_tree(x2, y2, node.right)
    
    def _optimal_split(self, x, y):
        if self.impurity_measure == 'gini':
            return self._optimal_gini_split(x, y)
        return self._optimal_entropy_split(x, y)

    def _optimal_entropy_split(self, x, y):
        # first, calculate the total entropy before the split
        total = len(y)
        num_no = y.count(0)
        p_no = num_no / total
        p_yes = (total - num_no) / total
        h = -(p_no * math.log2(p_no) + p_yes * math.log2(p_yes))
        # then, calculate the i.g. for each feature (column)
        opt_ig, opt_index, opt_mean = 0, 0, 0
        for i in range(len(x[0])):
            h_f, mean = self._calc_entropy(x, y, i)
            ig = h - h_f
            # compare current i.g. to the max i.g.
            if ig > opt_ig:
                opt_ig = ig
                opt_index = i
                opt_mean = mean
        return opt_index, opt_mean
    
    def _optimal_gini_split(self, x, y):
        no_of_features = len(x[0])
        min_gini_val, idx_of_min_gini, opt_mean = 1, 0, 0
        for i in range(no_of_features):
            gini, mean = self._calc_gini_index(x, y, i)
            if gini < min_gini_val:
                min_gini_val = gini
                idx_of_min_gini = i
                opt_mean = mean
        return idx_of_min_gini, mean

    def _calc_entropy(self, x, y, i):
        # extract the column we care about from the matrix
        col = [e[i] for e in x]
        # calculate the mean value of the column
        mean = stat.mean(col)
        # variables for counting decisions based on if number is below or above mean
        below_mean_no, below_mean_yes, above_mean_no, above_mean_yes = 0, 0, 0, 0
        # loop through the numbers and count number of no/yes decisions
        for index, number in enumerate(col):
            if number <= mean:
                if y[index] == 0:
                    below_mean_no += 1
                else:
                    below_mean_yes += 1
            else:
                if y[index] == 0:
                    above_mean_no += 1
                else:
                    above_mean_yes += 1
        # total number of decisions for each split
        total_below = below_mean_no + below_mean_yes
        total_above = above_mean_no + above_mean_yes
        total = total_below + total_above
        # calculate probabilities
        p_below_no = below_mean_no / total_below
        p_below_yes = below_mean_yes / total_below
        p_above_no = above_mean_no / total_above
        p_above_yes = above_mean_yes / total_above
        p_total_below = total_below / total
        p_total_above = total_above / total
        # calculate entropy for each of the two splits
        ent_below = -(self._log_or_zero(p_below_no) + self._log_or_zero(p_below_yes))
        ent_above = -(self._log_or_zero(p_above_no) + self._log_or_zero(p_above_yes))
        # return the sum of the two entropies times their weights
        total_ent = p_total_below * ent_below + p_total_above * ent_above
        return total_ent, mean
    
    # 1.2 - Gini index
    def _calc_gini_index(self, x, y, i):
        # get the column with the feature values we want to calculate
        feature = [e[i] for e in x]
        # calculate the mean (this is our splitting criteria)
        mean = stat.mean(feature)
        # count the number of no/yes decisions for data points below and above the mean
        num_below_no, num_below_yes, num_above_no, num_above_yes = 0, 0, 0, 0
        for index, value in enumerate(feature):
            if value <= mean:
                if y[index] == 0:
                    num_below_no += 1
                else:
                    num_below_yes += 1
            else:
                if y[index] == 0:
                    num_above_no += 1
                else:
                    num_above_yes += 1
        # total number of decisions for each split
        total_below = num_below_no + num_below_yes
        total_above = num_above_no + num_above_yes
        # calculate the separate gini impurities for the data points below and above the mean
        gini_below = 1 - math.pow(num_below_no / total_below, 2) - math.pow(num_below_yes / total_below, 2)
        gini_above = 1 - math.pow(num_above_no / total_above, 2) - math.pow(num_above_yes / total_above, 2)
        # calculate weighted average of the two gini impurities
        w_below = total_below / (total_below + total_above)
        w_above = total_above / (total_above + total_below)
        return gini_below * w_below + gini_above * w_above, mean

    # 1.3 - Add reduced-error pruning
    def _prune_tree(self, X, y, tree):
        if tree.is_leaf():
            return self._count_label_errors(tree.label, y)
        if not X:
            return 0
        x1, y1, x2, y2 = self._split_data(X, y, tree.data.index)
        # accuracy of the left and right subtrees
        err_l = self._prune_tree(x1, y1, tree.left)
        err_r = self._prune_tree(x2, y2, tree.right)
        # accuracy of the majority label
        err_m = self._count_label_errors(tree.data.ml, y)
        # prune subtree if accuracy of majority label is greater
        if err_m < err_l + err_r:
            # set the majority label as the node label and delete children
            tree.label = tree.data.ml
            tree.left = None
            tree.right = None
            return err_m
        return err_l + err_r

    def _count_label_errors(self, label, y):
        return len(y) - y.count(label)
    
    def _training_validation_split(self, X, y, prune):
        X_val, y_val = [], []
        if prune:
            threshold = len(X) / 3
            while len(X_val) < threshold:
                random_index = rand.randrange(len(X))
                X_val.append(X.pop(random_index))
                y_val.append(y.pop(random_index))
        return X, y, X_val, y_val
    
    def _log_or_zero(self, percentage):
        if percentage == 0:
            return 0
        return percentage * math.log2(percentage)
    
    def _split_data(self, x, y, split_index):
        x1, y1, x2, y2 = [], [], [], []
        col = [e[split_index] for e in x]
        mean = stat.mean(col)
        for index, row in enumerate(x):
            # get the value we are splitting on
            value = row[split_index]
            if value <= mean:
                x1.append(row)
                y1.append(y[index])
            else:
                x2.append(row)
                y2.append(y[index])
        return x1, y1, x2, y2
    
    def _has_same_label(self, y):
        for decision in y:
            if decision is not y[0]:
                return False
        return True
    
    def _has_identical_features(self, X, y):
        for i in range(len(X[0])):
            col = [e[i] for e in X]
            for value in col:
                if value is not col[0]:
                    return False
        return True
    
    def _majority_label(self, y):
        no_count = 0
        for d in y:
            if d == 0:
                no_count += 1
        if no_count >= len(y) / 2:
            return 0
        return 1

# utility function for loading data form file
def data_from_file(filename):
    x, y = [], []
    path = sys.path[0] + '/data/' + filename
    with open(path, 'r') as reader:
        for line in reader.readlines():
            parts = line.rstrip().split(',')
            # we assume the decision column is binary (either 0 or 1)
            y.append(int(parts.pop()))
            # we assume all feature values are floating point
            x.append([float(i) for i in parts])
    return x, y

def print_tree(node, level=0):
    if node != None:
        print_tree(node.right, level + 1)
        print(' ' * 5 * level + '->', node)
        print_tree(node.left, level + 1)

X, Y = data_from_file('data_banknote_authentication.csv')

dt = DecisionTree()
dt.learn(X, Y, prune=False)
print_tree(dt.tree)