from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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
def create_candidate_models(X, y, tune_params=True):
    # split data set into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=667)
    # decision tree
    dt = create_and_init_dt(X_train, y_train, tune_params)
    # neural network
    nn = create_and_init_nn(X_train, y_train, tune_params)
    # support vector machine
    vm = create_and_init_svc(X_train, y_train, tune_params)
    models = [
        ('Decision Tree', dt),
        ('Neural Network', nn),
        ('Support Vector Machine', vm)
    ]
    return models

"""Use grid search to tune a decision tree model and return it"""
def create_and_init_dt(X_train, y_train, tune_params):
    if not tune_params:
        return DecisionTreeClassifier(random_state=40)
    # define the hyperparameters we want to evaluate
    tuning_params = {'criterion': ['gini', 'entropy']}
    # use grid search to find the best combination of hyperparameters for our model
    clf = GridSearchCV(DecisionTreeClassifier(random_state=40), tuning_params, cv=3, return_train_score=False)
    clf.fit(X_train, y_train)
    best_params = clf.best_params_
    # create our final model and configure it with the best params
    decision_tree = DecisionTreeClassifier(
        ccp_alpha=0.001,
        random_state=40,
        criterion=best_params['criterion']
    )
    return decision_tree

"""Use grid search to tune a neural network model and return it"""
def create_and_init_nn(X_train, y_train, tune_params):
    if not tune_params:
        return MLPClassifier(max_iter=500)
    # define the hyperparameters we want to evaluate
    tuning_params = {
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'learning_rate': ['constant','adaptive']
    }
    # use grid search to find the best combination of hyperparameters for our model
    clf = GridSearchCV(MLPClassifier(), tuning_params, cv=3, return_train_score=False)
    clf.fit(X_train, y_train)
    best_params = clf.best_params_
    # create our final model and configure it with the best params
    neural_network = MLPClassifier(
        activation=best_params['activation'],
        learning_rate=best_params['learning_rate'],
        solver=best_params['solver'],
        max_iter=500
    )
    return neural_network

"""Use grid search to tune a support vector model and return it"""
def create_and_init_svc(X_train, y_train, tune_params):
    if not tune_params:
        return SVC()
    # define the hyperparameters we want to evaluate
    tuning_params = {'C': [1, 2, 5, 10, 20], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'linear', 'poly']}
    # use grid search to find the best combination of hyperparameters for our model
    clf = GridSearchCV(SVC(), tuning_params, cv=3, return_train_score=False)
    clf.fit(X_train, y_train)
    best_params = clf.best_params_
    # create our final model and configure it with the best params
    support_vector_machine = SVC(
        gamma=best_params['gamma'],
        kernel=best_params['kernel'],
        C=best_params['C']
    )
    return support_vector_machine

"""Used for determining the correct alpha value for CCP through visual inspection"""
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

"""Preprocess data points and do predictions using a given model"""
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
    # the avg. testing accuracy (aka the cross validated acc.) is used as the estimate 
    # of out of sample accuracy. The cross_val_score function takes care of splitting 
    # the data set, so we pass it the whole X and y sets. A 10-fold cross-validation is 
    # generally known to perform well, so use 10 as a default value.
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

"""Helper function for loading a single png image from file"""
def load_image_from_file(filename):
    image_data = plt.imread(filename)
    image_data = image_data.reshape(784, 3)
    new_image = []
    for p in image_data:
        r, g , b = p[0], p[1], p[2]
        # we only care about one channel, so combine all RGB values
        grayscale = (1 - ((r + b + g) / 3))
        # normalize (pixel value ranges from 0 - 255)
        grayscale = grayscale * 255
        new_image.append(grayscale)
    return new_image

"""Helper function for displaying image to user"""
def show_image(datapoint):
    if type(datapoint) is not np.array:
        datapoint = np.array(datapoint)
    # first, reshape it to a n*n pixel matrix
    n = int(math.sqrt(len(datapoint)))
    datapoint = datapoint.reshape(n, n)
    # use pyplot to show the image
    plt.imshow(datapoint, cmap='Greys')
    plt.show()



#########################
###  EXECUTE PROGRAM  ###
#########################

# we want to benchmark program execution time, so keep track of start time
start_time = time.time()
# flag for determining if we want to downsample images during preprocessing or not (from 28*28 to 14*14)
should_downsample_images = True
# flag for determining if we should tune the model parameters or not
should_tune_params = False

# load data and do preprocessing
X, y = load_and_preprocess_data(
    images_file='handwritten_digits_images_small.csv',
    labels_file='handwritten_digits_labels_small.csv',
    should_downsample=should_downsample_images
)

# create our 3 candidate models
models = create_candidate_models(X, y, tune_params=should_tune_params)

# select the model with the best CV mean score
best_model = select_best_model(models, X, y, no_of_folds=5)

# now that we have our best model, we can fit it with the data
best_model = best_model.fit(X, y)

# load unseen data (digits written on paper by myself)
img_three = load_image_from_file('three.png')
img_five = load_image_from_file('five.png')
img_nine = load_image_from_file('nine.png')

# predict the class of the above images
predictions = preprocess_and_predict(
    model=best_model, 
    datapoints=[img_three, img_five, img_nine], 
    should_downsample=should_downsample_images
)

print('Predictions: %s' % str(predictions))
print('Program execution time: %fs' % (time.time() - start_time))

