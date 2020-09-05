import numpy as np
import math
import sys

# 1.1 Implement a decision tree learning algorithm from scratch

class Node():
    def __init__(self, label, children=None):
        self.label = label
        self.children = children

def learn(X, y, impurity_measure='entropy'):
    if has_same_label(y):
        return Node(y[0], None)
    if has_identical_features(X):
        yield

def has_same_label(y):
    if len(y) == 1:
        return True
    for label in y:
        if label != y[0]:
            return False
    return True

def has_identical_features(x):
    if len(x) == 1:
        return True
    for data in x:
        if data != x[0]:
            return False
    return True


def calc_entropy(y, label):
    y_len = len(y)
    label_cnt = y.count(label)
    p_a = label_cnt/y_len
    p_b = (y_len-label_cnt)/y_len
    return -(p_a * math.log2(p_a) + p_b * math.log2(p_b))

def calc_entropy(x, y, idx):
    col = [row[idx] for row in x]
    values, decisions = {}, {}
    total = len(y)
    for index, key in enumerate(col):
        # increment the counter the given decision value
        if key in values:
            values[key] += 1
        else:
            values[key] = 1
        # increment the number of 'no' decisions
    print("VALUES")
    print(values)
    print("DEC")
    print(decisions)
    return col



# Utility functions

def data_from_file(filename):
    x, y = [], []
    path = sys.path[0] + '/data/' + filename
    # data = np.genfromtxt(path, delimiter=',')
    with open(path, 'r') as reader:
        for line in reader.readlines():
            parts = line.rstrip().split(',')
            y.append(parts.pop())
            x.append(parts)
    return x, y

X, Y = data_from_file('tennis.csv')
l = calc_entropy(X, Y, 3)
#print(l)