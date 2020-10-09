from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from random import randint
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
        return self.model.predict(dps)
    def map_and_predict(self, dps):
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
            hour_of_day = dp[3]
            is_weekend = date_is_weekend(date)
            is_holiday = date_is_holiday(date)
            new_dps.append([hour_of_day, is_weekend, is_holiday])
        return new_dps

# hyper parameter = (e.g.) K_nn

# 1.2 Preprocessing
def preprocess_data(X):
    # create new features from the existing feature set:
    new_features = {
        'hour_of_day': [],  # numerical feature
        'is_weekend': [],   # categorical/binary feature
        'is_holiday': []    # categorical/binary feature
    }
    for index, row in X.iterrows():
        date = date_from_row(row)
        is_weekend = date_is_weekend(date)
        is_holiday = date_is_holiday(date)
        new_features['hour_of_day'].append(row['Fra_time'])
        new_features['is_weekend'].append(is_weekend)
        new_features['is_holiday'].append(is_holiday)
    return pd.DataFrame(new_features, columns=['hour_of_day', 'is_weekend', 'is_holiday'])

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

# 1.3 Modelling and evaluation

def train_and_evaluate_model(model, X, y):
    print('Training and evaluating model %s...' % model.name)
    # split the data set into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=randint(100, 300))
    # apply training data on the model
    model = model.learn(X_train, y_train)
    # to evaluate the performance of the model, we make predictions on the same data set we used for training
    y_pred = model.predict(X_test)
    # return the score of the model base on the mean accuracy
    return model.score(X_test, y_test)


#######################
# EXECUTE THE PROGRAM #
#######################

# create our three models
linear_regression = Model(LinearRegression(), name='Linear Regression')
logistic_regression = Model(LogisticRegression(), name='Logistic Regression')
regression_tree = Model(DecisionTreeRegressor(criterion="mse"), name='Regression Tree')
neural_network = Model(MLPRegressor(), name='Neural Network')

# load data from file and preprocess
X, y = load_and_preprocess_data('data.csv')
y_total = y.loc[:, 'Volum totalt']
y_sntr = y.loc[:, 'Volum til SNTR']
y_dnp = y.loc[:, 'Volum til DNP']

# learn and evaluate each of the three models separately
acc_lr = train_and_evaluate_model(linear_regression, X, y_total)
acc_rt = train_and_evaluate_model(regression_tree, X, y_total)
acc_nn = train_and_evaluate_model(neural_network, X, y_total)
print(acc_lr, acc_rt, acc_nn)
# execute_model(linear_regression, X, y, datapoints)
#execute_model(logistic_regression, X, y, datapoints)
# execute_model(regression_tree, X, y, datapoints)
# execute_model(neural_network, X, y, datapoints)

