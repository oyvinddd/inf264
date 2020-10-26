from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import time

"""Load data from file and do preprocessing"""
def load_and_preprocess_data(images_file, labels_file, should_downsample=False):
    # read images and their respective classes from csv
    X = np.loadtxt(open(images_file, 'rb'), delimiter=',', dtype=int)
    y = np.loadtxt(open(labels_file, 'rb'), delimiter=',', dtype=int)
    X_new = []
    for image_row in X:
        # preprocess each image by going through each pixel and giving it the value
        # of 1 for values higher than 255/2 and 0 for values lower. This way we can
        # treat each pixel in the image as a binary feature in our model.
        image_row = to_binary_values(image_row)
        # we can choose to resize our images to half the size of the original images.
        # this makes our program much more efficient in terms of running time, but it
        # does have a slight negative effect on model precision.
        if should_downsample:
            X_new.append(downsample_image(image_row))
    if should_downsample:
        return X_new, y
    return X, y

"""Set each pixel in an image to either 0 or 1 depending on the original RGB value"""
def to_binary_values(image_row):
    threshold = 255/2
    for index, col in enumerate(image_row):
        if image_row[index] > threshold:
            image_row[index] = 1
        else:
            image_row[index] = 0
    return image_row

"""Downsample (resize) image to 1/4 of the size of the original image"""
def downsample_image(image_row):
    if type(image_row) is not np.array:
        image_row = np.array(image_row)
    image_row = image_row.reshape(28, 28)
    input_size, output_size = 28, 14
    bin_size = input_size // output_size
    downsampled_image = image_row.reshape((output_size, bin_size, output_size, bin_size, 1)).max(3).max(1)
    return downsampled_image.reshape(196)

"""Create and initialize all three candidate models"""
def create_candidate_models(X, y):
    # split data set into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=667)
    # decision tree
    dt = DecisionTreeClassifier(
        ccp_alpha=0.002,
        random_state=40,
        criterion='entropy'
    )
    # neural network
    nn = MLPClassifier(max_iter=500)
    # support vector machine
    vm = SVC(kernel='rbf', decision_function_shape='ovr')
    models = [
        ('Decision Tree', dt),
        ('Neural Network', nn),
        ('Support Vector Machine', vm),
        ('K-Nearest Neighbour', KNeighborsClassifier(n_neighbors=20))
    ]
    return models

"""Used for determining the correct alpha value for CCP"""
def plot_decision_tree_alphas(X, y):
    # split data set into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=667)
    # create decision tree classifier
    tree = DecisionTreeClassifier(random_state=40)
    # do cost complexity pruning
    path = tree.cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas
    alphas = alphas[:-1]
    train_acc, test_acc = [], []
    # create a decision tree for each of our alpha values and store the training and testing accuracies
    for alpha in alphas:
        tree = DecisionTreeClassifier(random_state=0, ccp_alpha=alpha)
        tree = tree.fit(X_train, y_train)
        y_train_pred = tree.predict(X_train)
        y_test_pred = tree.predict(X_test)
        train_acc.append(accuracy_score(y_train, y_train_pred))
        test_acc.append(accuracy_score(y_test, y_test_pred))
    # graphically plot the accuracies of the trees using the training and testing datasets
    fig, ax = plt.subplots()
    ax.set_xlabel('alpha')
    ax.set_ylabel('accuracy')
    ax.set_title('Accuracy vs alpha for training and testing sets')
    ax.plot(alphas, train_acc, marker='o', label='train', drawstyle='steps-post')
    ax.plot(alphas, test_acc, marker='o', label='test', drawstyle='steps-post')
    ax.legend()
    plt.show()

"""Preprocess data points and do preditions using a given model"""
def preprocess_and_predict(model, datapoints, should_downsample=False):
    new_datapoints = []
    # convert pixel values to binary so it matches the training set values
    for datapoint in datapoints:
        datapoint = to_binary_values(datapoint)
        if should_downsample:
            new_datapoints.append(downsample_image(datapoint))
    # return a list of predicitons
    if should_downsample:
        return model.predict(new_datapoints)
    return model.predict(datapoints)

"""We use cross-validation to help us select the best model"""
def select_best_model(models, X, y, no_of_folds=10):
    best_score, best_model = 0, None
    # the avg. testing accuracy (aka the cross validated acc.) is used as the estimate of out of sample acc.
    # the cross_val_score function takes care of splitting the data set, so we pass it the whole X and y
    # use 10-fold cross validation
    for name, model in models:
        scores = cross_val_score(model, X, y, cv=no_of_folds, scoring='accuracy')
        # calculate the mean score from all the 10 scores
        mean_score = scores.mean()
        print('Mean score for model %s: %f' % (name, mean_score))
        if mean_score > best_score:
            best_model = model
            best_score = mean_score
    # return the model with the best mean score
    print('Chosen model: %s' % best_model)
    return best_model

"""Helper function for displaying image to user"""
def show_image(datapoint):
    if type(datapoint) is not np.array:
        datapoint = np.array(datapoint)
    # first, reshape it to a res x res pixel matrix
    res = int(math.sqrt(len(datapoint)))
    datapoint = datapoint.reshape(res, res)
    # use pyplot to show the image
    plt.imshow(datapoint, cmap='Greys')
    plt.show()



#########################
###  EXECUTE PROGRAM  ###
#########################

# we want to benchmark program execution time, so keep track of start time
start_time = time.time()
# flag for determining if we want to downsample image or not (from 28*28 to 14*14)
should_downsample_image = True

# load data and do preprocessing
X, y = load_and_preprocess_data(
    images_file='handwritten_digits_images_small.csv',
    labels_file='handwritten_digits_labels_small.csv',
    should_downsample=should_downsample_image
)

# create our 3 candidate models
models = create_candidate_models(X, y)

# select the model with the best CV mean score
best_model = select_best_model(models, X, y, no_of_folds=5)

# now that we have our best model, we can fit it with the data
best_model = best_model.fit(X, y)

image = [
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,129,103,155,116,155,233,155,214,255,
    231,149,19,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,250,254,253,253,253,253,254,253,253,
    254,253,224,32,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,123,188,188,188,188,95,89,89,89,89,
    177,253,235,19,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,253,254,74,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32,99,241,253,207,24,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,89,220,253,252,157,24,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,47,161,252,253,233,120,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,47,242,253,246,
    170,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,86,253,254,242,115,73,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35,226,253,253,253,253,244,123,26,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,28,87,148,190,249,249,253,229,103,3,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,6,94,188,253,190,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,4,146,254,70,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,80,253,163,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,168,254,164,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,50,232,253,132,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,
    232,253,221,30,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,47,207,254,253,84,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,93,249,253,231,107,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,60,142,253,253,226,39,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
]
# make predicitons on unseen data
predictions = preprocess_and_predict(
    model=best_model, 
    datapoints=[image], 
    should_downsample=should_downsample_image
)

print('Predictions: %s' % str(predictions))
print('Program execution time: %fs' % (time.time() - start_time))

