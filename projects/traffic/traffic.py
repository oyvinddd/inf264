import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import datetime as dt
import holidays

# Task 1: Predicting traffic
# In this project, we practice constructing regression models by predicting traffic.

# Quantities: # of cars crossing, # of cars towards city center and # of cars towards danmarksplass
# Features: year, month, day and time

class Model:
    def __init__(self, model):
        self.model = model
    def learn(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self
    def predict(self, dps):
        new_dp = self._map_datapoints(dps)
        print("new dP: " + str(new_dp))
        return self.model.predict(new_dp)
    def _map_datapoints(self, dps):
        new_dps = []
        for dp in dps:
            if len(dp) != 4:
                continue
            date = dt.datetime(dp[0], dp[1], dp[2], hour=dp[3])
            is_weekend = date_is_weekend(date)
            is_holiday = date_is_holiday(date)
            new_dps.append([dp[3], is_weekend, is_holiday])
        return new_dps



# hyper parameter = (e.g.) K_nn

# what could influence the number of cars?
#   - if it's a weekday or not (categorical feature)
#   - if it's a holiday or not (categorical feature)
#   - is it winter?
#   - what quarter is it?

# 1.2 Preprocessing
def preprocess_data(X):
    # create new features from the existing feature set:
    new_features = {
        'hour': [],         # numerical feature
        'is_weekend': [],   # categorical/binary feature
        'is_holiday': []    # categorical/binary feature
    }
    for index, row in X.iterrows():
        new_features['hour'].append(row['Fra_time'])
        date = date_from_row(row)
        is_weekend = date_is_weekend(date)
        is_holiday = date_is_holiday(date)
        new_features['is_weekend'].append(is_weekend)
        new_features['is_holiday'].append(is_holiday)
    X_new = pd.DataFrame(new_features, columns=['hour', 'is_weekend', 'is_holiday'])
    return X_new

# helper function for loading data from file into a pandas DataFrame
def load_data(filename):
    # load data and slice it into feature and quantity sets
    data = pd.read_csv(filename, delimiter=',')
    X = data.loc[:, 'År':'Fra_time']
    y = data.loc[:, 'Volum til SNTR':'Volum totalt']
    return X, y

def load_and_preprocess_data(filename):
    yield

# helper function that returns a datetime object from a DataFrame row
def date_from_row(row):
    year, month, day, hour = row['År'], row['Måned'], row['Dag'], row['Fra_time']
    return dt.datetime(year, month, day, hour=hour)

# helper function for checking if a given date is a weekday or weekend
def date_is_weekend(date):
    if date.isoweekday() not in range(1, 6):
        return True
    return False

# helper function for checking if a given date is a norwegian holiday
def date_is_holiday(date):
    if date in holidays.Norway(years=[date.year], include_sundays=False):
        return True
    return False

# helper function for visualizing car volume with respect to the selected features
def plot_data(X, y):
    colors = ['r', 'g', 'b']
    for col in range(X.shape[1]):
        plt.figure(1, figsize=(24, 16))
        if col < X.shape[1] - 1:
            plot_idx = col+1
        else:
            plot_idx = 4
        plt.subplot(5, 3, plot_idx)
        plt.scatter(X.iloc[:, col], y.iloc[:, 2], marker='o', c=colors[col])
        plt.xlabel(X.columns[col])
        plt.ylabel('Total volume')
    plt.suptitle("Total car volume with respect to each of the features")
    plt.show()

# 1.3 Modelling and evaluation

# function for learning a linear regression model
def learn_lr():
    yield
    # lr = LinearRegression()
    # lr.fit(X_train, y_train)
    # p = lr.predict(X_test)

# function for learning a regression tree model
def learn_models(X, y):
    # extreact column with the total volume of cars
    y_total = y.loc[:, 'Volum totalt']

    # split the data set into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_total, test_size=0.3, shuffle=True, random_state=667)
    
    # dt = DecisionTreeRegressor(criterion="mse")
    # dt = dt.fit(X_train, y_train)
    
    dp = [2015,12,17,19] # 2015,12,17,19,106,142,248
    # pr = dt.predict([dp])
    # print(pr)

    md = Model(DecisionTreeRegressor(criterion="mse"))
    md = md.learn(X_train, y_train)
    print(md.predict([dp]))



# run the program

# load data
X, y = load_data('data_small.csv')
# preprocess data
X_new = preprocess_data(X)
# call function for learning models
learn_models(X_new, y)
