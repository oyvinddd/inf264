import numpy as np
import math
import sys

# 1.1 Implement a decision tree learning algorithm from scratch

class Node():
    def __init__(self, label, children=None):
        self.label = label
        self.children = children

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

# def calc_ent():



# Utility functions

def data_from_file(filename):
    x, y = [], []
    path = sys.path[0] + '/data/' + filename
    # data = np.genfromtxt(path, delimiter=',')
    with open(path, 'r') as reader:
        for line in reader.readlines():
            parts = line.rstrip().split(',')
            # we assume the decision column is binary (either 0 or 1)
            y.append(int(parts.pop()))
            # we assume all feature values are floating point
            x.append([float(i) for i in parts])
    return x, y

X, Y = data_from_file('data_banknote_authentication.txt')
print(X)