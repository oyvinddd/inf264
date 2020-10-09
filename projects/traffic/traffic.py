from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from random import randint
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
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
        'is_holiday': [],   # categorical/binary feature
        'is_gsh': []        # categorical/binary feature (gsh = general staff holiday) 
    }
    for index, row in X.iterrows():
        date = date_from_row(row)
        is_weekend = date_is_weekend(date)
        is_holiday = date_is_holiday(date)
        is_gsh = date_is_gsh(date)
        new_features['hour_of_day'].append(row['Fra_time'])
        new_features['is_weekend'].append(is_weekend)
        new_features['is_holiday'].append(is_holiday)
        new_features['is_gsh'].append(is_gsh)
    return pd.DataFrame(new_features, columns=['hour_of_day', 'is_weekend', 'is_holiday', 'is_gsh'])

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

# helper function for checking if a given date is a Norwegian general staff holiday
def date_is_gsh(date):
    # the first element returned from isocalendar() is the week number
    week_num = date.isocalendar()[1]
    if week_num in [28, 29, 30]:
        return True
    return False

# 1.3 Modelling and evaluation

# helper function for picking the best model amongst all three models
def select_best_model(X_train, y_train, X_val, y_val):
    # create our three models
    sv = Model(SVR(kernel='rbf'), name='Support Vector Machine')
    rt = Model(DecisionTreeRegressor(criterion="mse"), name='Regression Tree')
    nn = Model(MLPRegressor(hidden_layer_sizes=(300,), activation='relu', solver='adam', max_iter=500), name='Neural Network')
    # learn and evaluate each of the three models separately
    sv, acc_sv = train_and_evaluate_model(sv, X_train, y_train, X_val, y_val)
    rt, acc_rt = train_and_evaluate_model(rt, X_train, y_train, X_val, y_val)
    nn, acc_nn = train_and_evaluate_model(nn, X_train, y_train, X_val, y_val)
    print(acc_sv, acc_rt, acc_nn)
    # determine which model had the best accuracy and return it
    if acc_sv < acc_rt or acc_sv < acc_nn:
        if acc_rt < acc_nn:
            return nn, acc_nn # neural network won
        return rt, acc_rt # regression tree won
    return sv, acc_sv # support vector machine won

# helper function for training and evaluating a given model based on training and validation data
def train_and_evaluate_model(model, X_train, y_train, X_val, y_val):
    print('Training and evaluating model %s...' % model.name)
    # apply training data on the model
    model = model.learn(X_train, y_train)
    # return model, as well as the score of the model base on the mean accuracy
    return model, model.score(X_val, y_val)

# function for running and instance. An instance is just the scenario we're currently looking at (sentrum, sntr or dnp)
def run_instance(X, y, instance_desc):
    print(instance_desc)
    # split the data into train and a concatenation of validation and test sets with a ratio of 0.7/0.3
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y,  test_size=0.3, shuffle=True, random_state=666)
    # further split set into validation and test sets with a ratio of 0.5/0.5
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, shuffle=True, random_state=667)
    # call function for determining the best model based on the training data
    best_model, best_acc = select_best_model(X_train, y_train, X_val, y_val)
    print('Best model was %s' % best_model.name)
    # now we can estimate the performance of the best model on unseen (test) data
    print('Estimating performance of %s on unseen data...' % best_model.name)
    y_pred = best_model.predict(X_test)
    score = best_model.score(X_test, y_test)
    print('Score: ' + str(score))

# Unfinished - function for visualizing prediction
def vizualize_prediction(X, y, y_pred):
    yield
    # visualize prediction
    # X_grid = X_test.reshape((len(X_test), 1))
    # print(X_test.loc[:, 'hour_of_day'])
    # plt.scatter(X.loc[:, 'Fra_time'], y, color='red')
    # plot predicted data 
    # plt.plot(X_grid, y_pred, color = 'blue') 
    # plt.show()


#######################
# EXECUTE THE PROGRAM #
#######################

# load and preprocess data from file
X, y = load_data('data.csv')
X_new = preprocess_data(X)

# split the different quantities
y_total = y.loc[:, 'Volum totalt']
y_sntr = y.loc[:, 'Volum til SNTR']
y_dnp = y.loc[:, 'Volum til DNP']

# run our three instances (total, sntr and dnp)
run_instance(X_new, y_total, 'Running instance for total # of cars...')
run_instance(X_new, y_sntr, 'Running instance for # of cars towards SNTR...')
run_instance(X_new, y_dnp, 'Running instance for # of cars towards DNP...')

