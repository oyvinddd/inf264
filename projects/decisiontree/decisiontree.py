import numpy as np
import statistics as stat
import math
import sys

# 1.1 Implement a decision tree learning algorithm from scratch

class Node:

    def __init__(self, label, left=None, right=None):
        self.label = label
        self.left = left
        self.right = right

    def __str__(self):
        return self.label

class DecisionTree:

    def __init__(self):
        self.tree = None

    def learn(self, X, y, impurity_measure='entropy'):
        self._build_tree(X, y, self.tree)

    def _build_tree(self, x, y, node):
        if self._has_unique_label(y):
            print("unique label")
            return
        if self._has_identical_features(x, y):
            print("has identical features")
            return
        # calculate the i.g. and get index of the optimal feature
        opt_index = self._optimal_ig_index(x, y)
        # split the data by the feature into two separate sets
        x1, y1, x2, y2 = self._split_data(x, y, opt_index)

    def _optimal_ig_index(self, x, y):
        # first, calculate the total entropy before the split
        total = len(y)
        num_no = y.count(0)
        p_no = num_no / total
        p_yes = (total - num_no) / total
        ent = -(p_no * math.log2(p_no) + p_yes * math.log2(p_yes))
        # then, calculate the i.g. for each feature (column)
        opt_ig, opt_index = 0, 0
        for i in range(len(x[0])):
            ig = ent - self._calc_entropy(x, y, i)
            if ig > opt_ig:
                opt_ig = ig
                opt_index = i
        return opt_index

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
        ent_below = -(p_below_no * math.log2(p_below_no) + p_below_yes * math.log2(p_below_yes))
        ent_above = -(p_above_no * math.log2(p_above_no) + p_above_yes * math.log2(p_above_yes))
        # return the sum of the two entropies times their weights
        return p_total_below * ent_below + p_total_above * ent_above
    
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
    
    def _has_unique_label(self, y):
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


X, Y = data_from_file('data_banknote_authentication.txt')
dt = DecisionTree()
dt.learn(X, Y)

print(dt.tree)
dt.test_build()
print(dt.tree)