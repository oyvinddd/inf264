import numpy as np
import statistics as stat
import math
import sys

# 1.1 Implement a decision tree learning algorithm from scratch

class Node:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children

def optimal_ig_index(x, y):
    # first, calculate the total entropy before the split
    total = len(y)
    num_no = y.count(0)
    p_no = num_no / total
    p_yes = (total - num_no) / total
    ent = -(p_no * math.log2(p_no) + p_yes * math.log2(p_yes))
    # then, calculate the i.g. for each feature (column)
    opt_ig, opt_index = 0, 0
    for i in range(len(x[0])):
        ig = ent - calc_entropy(x, y, i)
        if ig > opt_ig:
            opt_ig = ig
            opt_index = i
    return opt_index


def calc_entropy(x, y, i):
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


# Utility functions

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
index = optimal_ig_index(X, Y)
print("optimal informaiton gain index: " + str(index))
