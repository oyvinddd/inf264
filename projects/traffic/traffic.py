import sklearn as sk
import pandas as pd # Dont need this I think
import numpy as np

# Task 1: Prediction traffic
# In this project, we practice constructing regression models by predicting traffic.

# Quantities: # of cars crossing, # of cars towards city center and # of cars towards danmarksplass
# Features: year, month, day and time

# load data set and split it into feature and quantity sets
traffic_data = np.genfromtxt('data.csv', delimiter=',', skip_header=1, dtype=int)
X = traffic_data[:, :4]
y = traffic_data[:, 4:7]


# clues on choosing features in regression model in hw3e
