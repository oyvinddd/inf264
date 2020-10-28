from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import math
import time

"""Load data from file and do preprocessing"""
def load_and_preprocess_data(images_file, labels_file, should_downsample=False):
    print('Step 1. Loading and Preprocessing...')
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
def create_candidate_models(X_train, y_train, tune_params=True):
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
def create_and_init_dt(X_train, y_train, tune_params, random_seed=667):
    if not tune_params:
        return DecisionTreeClassifier(random_state=random_seed)
    # define the hyperparameters we want to evaluate
    tuning_params = {
        'ccp_alpha': [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0],
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random']
    }
    # use grid search to find the best combination of hyperparameters for our model
    clf = GridSearchCV(DecisionTreeClassifier(random_state=random_seed), tuning_params, cv=3, return_train_score=False)
    clf.fit(X_train, y_train)
    best_params = clf.best_params_
    print('Using the following parameters for decision tree: %s' % str(best_params))
    # create our final model and configure it with the best params
    decision_tree = DecisionTreeClassifier(
        ccp_alpha=best_params['ccp_alpha'],
        criterion=best_params['criterion'],
        random_state=random_seed
    )
    return decision_tree

"""Use grid search to tune a neural network model and return it"""
def create_and_init_nn(X_train, y_train, tune_params, random_seed=667):
    if not tune_params:
        return MLPClassifier(max_iter=500, random_state=random_seed)
    # define the hyperparameters we want to evaluate
    tuning_params = {
        'activation': ['tanh', 'relu', 'logistic', 'identity'],
        'solver': ['sgd', 'adam'],
        'learning_rate': ['constant','adaptive']
    }
    # use grid search to find the best combination of hyperparameters for our model
    clf = GridSearchCV(MLPClassifier(max_iter=500, random_state=random_seed), tuning_params, cv=3, return_train_score=False)
    clf.fit(X_train, y_train)
    best_params = clf.best_params_
    print('Using the following parameters for neural network: %s' % str(best_params))
    # create our final model and configure it with the best params
    neural_network = MLPClassifier(
        activation=best_params['activation'],
        learning_rate=best_params['learning_rate'],
        solver=best_params['solver'],
        random_state=random_seed,
        max_iter=500
    )
    return neural_network

"""Use grid search to tune a support vector model and return it"""
def create_and_init_svc(X_train, y_train, tune_params, random_seed=667):
    if not tune_params:
        return SVC(random_state=random_seed)
    # define the hyperparameters we want to evaluate
    tuning_params = {
        'gamma': ['scale', 'auto'], 
        'kernel': ['rbf', 'linear', 'poly'],
        'C': [1, 2, 5, 10, 20]
    }
    # use grid search to find the best combination of hyperparameters for our model
    clf = GridSearchCV(SVC(random_state=random_seed), tuning_params, cv=3, return_train_score=False)
    clf.fit(X_train, y_train)
    best_params = clf.best_params_
    print('Using the following parameters for support vector machine: %s' % str(best_params))
    # create our final model and configure it with the best params
    support_vector_machine = SVC(
        gamma=best_params['gamma'],
        kernel=best_params['kernel'],
        C=best_params['C'],
        random_state=random_seed
    )
    return support_vector_machine

"""Used for determining the correct alpha value for CCP through visual inspection"""
def plot_decision_tree_alphas(X, y):
    # split data set into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=667)
    # create decision tree classifier
    tree = DecisionTreeClassifier(random_state=667)
    # do cost complexity pruning
    path = tree.cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas
    alphas = alphas[:-1]
    train_acc, test_acc = [], []
    # create a decision tree for each of our alpha values and store the training and testing accuracies
    for alpha in alphas:
        tree = DecisionTreeClassifier(random_state=667, ccp_alpha=alpha)
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
def preprocess_and_predict(model, datapoints, convert_to_binary=False, should_downsample=False):
    new_datapoints = []
    # convert pixel values to binary so it matches the training set values
    if convert_to_binary:
        [to_binary_values(dp) for dp in datapoints]
    if should_downsample:
        [downsample_image(dp) for dp in datapoints]
    # return a list of predictions
    return model.predict(datapoints)

"""We use cross-validation to help us select the best model"""
def select_best_model(models, X, y, no_of_folds=10):
    print('Step 2. Model selection...')
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

"""Evaluate the given model by calculating accuracy score"""
def evaluate_model(model, X_train, X_test, y_train, y_test, preprocess=False):
    print('Step 3. Model evaluation...')
    # fit model wit training data
    model = model.fit(X_train, y_train)
    # do predicitons on unseen data (testing set)
    y_pred = preprocess_and_predict(
        model=model, 
        datapoints=X_test,
        convert_to_binary=preprocess,
        should_downsample=preprocess
    )
    # calculate the accuracy classification score on the test set
    return accuracy_score(y_test, y_pred)

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
should_tune_params = True

# load data and do preprocessing
X, y = load_and_preprocess_data(
    images_file='handwritten_digits_images.csv',
    labels_file='handwritten_digits_labels.csv',
    should_downsample=should_downsample_images
)

# split data set into training and testing sets
random_seed = 667
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=random_seed)

# create our 3 candidate models and hand them some training data to work with
models = create_candidate_models(X_train, y_train, tune_params=should_tune_params)

# select the model with the best CV mean score
best_model = select_best_model(models, X_train, y_train, no_of_folds=5)

# evaluate performance of the best model on unseen data
acc_score = evaluate_model(best_model, X_train, X_test, y_train, y_test)
print('Accuracy score : %f' % acc_score)
print('Program execution time: %fs' % (time.time() - start_time))

# load unseen data (digits written on paper by myself)
# img_three = load_image_from_file('three.png')
# img_five = load_image_from_file('five.png')
# img_nine = load_image_from_file('nine.png')

