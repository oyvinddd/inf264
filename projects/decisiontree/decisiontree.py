import numpy as np
import statistics as stat
import math
import sys

# 1.1 Implement a decision tree learning algorithm from scratch

class Data:
    def __init__(self, feature_index, mean):
        self.feature_index = feature_index
        self.mean = mean

class Node:
    def __init__(self, label=None, data=None, left=None, right=None):
        self.label = label
        self.data = data
        self.left = left
        self.right = right
    def __str__(self):
        if self.label != None:
            return self.label
        return 'Null'
    def is_leaf(self):
        return self.left is None and self.right is None


class DecisionTree:

    def __init__(self):
        self.tree = Node()

    def learn(self, X, y, impurity_measure='entropy'):
        self._build_tree(X, y, self.tree)
    
    def predict(self, x):
        yield

    def _build_tree(self, x, y, node):
        if self._has_same_label(y):
            if y[0] == 0:
                node.label = 'No'
            else:
                node.label = 'Yes'
            return
        if self._has_identical_features(x, y):
            label = self._most_common_label(y)
            if label == 0:
                node.label = 'No'
            else:
                node.label = 'Yes'
            return
        # get index of the feature with optimal i.g.
        index, mean = self._optimal_ig_index(x, y)
        # split the data into two separate sets (based on the feature)
        x1, y1, x2, y2 = self._split_data(x, y, index)
        # use feature index as the node name and the mean as the data
        node.label = "split index: " + str(index)
        node.data = Data(index, mean)
        # call the method recursively with the smaller data sets
        if x1 and y1:
            node.left = Node()
            self._build_tree(x1, y1, node.left)
        if x2 and y2:
            node.right = Node()
            self._build_tree(x2, y2, node.right)

    def _optimal_ig_index(self, x, y):
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
            if ig > opt_ig:
                opt_ig = ig
                opt_index = i
                opt_mean = mean
        return opt_index, opt_mean

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
    
    def _log_or_zero(self, percentage):
        if percentage == 0:
            return 0
        return percentage * math.log2(percentage)
    
    def _split_data(self, x, y, split_index):
        x1, y1, x2, y2 = [], [], [], []
        col = [e[split_index] for e in x]
        mean = stat.mean(col)
        for index, row in enumerate(x):
            # remove the value we are checking from the row
            value = row.pop(split_index)
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
        if not X:
            return False
        for i in range(len(X[0])):
            col = [e[i] for e in X]
            for value in col:
                if value is not col[0]:
                    return False
        return True
    
    def _most_common_label(self, y):
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

def print_t(t, h):
    if t == None:
        return
    print("Node at height: " + str(h) + ", " + str(t))
    h += 1
    print_t(t.left, h)
    print_t(t.right, h)

X, Y = data_from_file('data_banknote_authentication.txt')
dt = DecisionTree()
dt.learn(X, Y)

# tree = Node('0')
# tree.left = Node('1')
# tree.right = Node('2')
# tree.left.left = Node('3')
# tree.left.right = Node('4')
# tree.right.left = Node('5')
# tree.right.right = Node('6')
t = dt.tree
print_t(t, 0)