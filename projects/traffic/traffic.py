from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from random import randint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import holidays

class Model:
    def __init__(self, model, name=None):
        self.model = model
        self.name = name
    def learn(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self
    def predict(self, dps):
        new_dp = self._map_datapoints(dps)
        return self.model.predict(new_dp)
    def score(self, X, y):
        return self.model.score(X, y)
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
        # 'day_of_week': [],  # numerical feature
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
    return pd.DataFrame(new_features, columns=['hour', 'is_weekend', 'is_holiday'])

# helper function for loading data from file into a pandas DataFrame
def load_data(filename):
    # load data and slice it into feature and quantity sets
    data = pd.read_csv(filename, delimiter=',')
    X = data.loc[:, 'År':'Fra_time']
    y = data.loc[:, 'Volum til SNTR':'Volum totalt']
    return X, y

# wrapper function to return preprocessed data from csv
def load_and_preprocess_data(filename):
    X, y = load_data(filename)
    X_new = preprocess_data(X)
    return X_new, y

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

# function for learning and making predictions on a given model
def execute_model(model, X, y, unseen_datapoints):
    print('Executing model ' + model.name + '...')
    # split the quantities into separate sets
    y_sntr = y.loc[:, 'Volum til SNTR']
    y_dnp = y.loc[:, 'Volum til DNP']
    y_total = y.loc[:, 'Volum totalt']
    # split the data set into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_total, test_size=0.3, shuffle=True, random_state=randint(100, 300))
    # learn model using the training data
    md = model.learn(X_train, y_train)
    # make predictions on unseen data
    predictions = md.predict(unseen_datapoints)
    print("Predictions: " + str(predictions))
    # score of the given model
    score = md.score(X_test, y_test)
    print('Score: ' + str(score))

    # make prediciton on training data
    # predictions = md.predict(X_train)
    # good_train_predicitons = (predictions == y_train)
    # train_accuracy = np.sum(good_train_predicitons) / len(X_train)


#######################
# EXECUTE THE PROGRAM #
#######################

# create our three models
linear_regression = Model(LinearRegression(), name='Linear Regression')
regression_tree = Model(DecisionTreeRegressor(criterion="mse"), name='Regression Tree')
neural_network = Model(MLPRegressor(), name='Neural Network')
# load data from file and preprocess
X, y = load_and_preprocess_data('data.csv')
# create a set of unseen data we want to predict
datapoints = [
    [2015,12,17,19] # 2015,12,17,19,106,142,248
]
# learn each of the three models separately
execute_model(linear_regression, X, y, datapoints)
execute_model(regression_tree, X, y, datapoints)
execute_model(neural_network, X, y, datapoints)