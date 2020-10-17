from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# NOTES:
# gradient descent for hp?
#
# Try support vector machine
# - probably won't need to reshape before we want to show image to user
# https://www.youtube.com/watch?v=7aK7tVdcVbY&ab_channel=CodeHeroku



def load_and_preprocess_data(images_file, labels_file):
    # read images and their respective classes from csv
    X = np.loadtxt(open(images_file, 'rb'), delimiter=',', dtype=int)
    y = np.loadtxt(open(labels_file, 'rb'), delimiter=',', dtype=int)
    # preprocess each image by going through each pixel and giving it the value
    # of 1 for values higher than 255/2 and 0 for values lower. This way we can
    # treat each pixel in the image as a binary feature in our model.
    for row in X:
        row = to_binary_values(row)
    return X, y

def to_binary_values(image_row):
    for index, col in enumerate(image_row):
        if image_row[index] > 255/2:
            image_row[index] = 1
        else:
            image_row[index] = 0
    return image_row

def train_validation_test_split(X, y):
    # do train, validation test split
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=667)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, shuffle=True, random_state=668)
    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_and_predict(model, datapoints):
    # convert pixel values to binary so it matches the training set values
    for dp in datapoints:
        dp = to_binary_values(dp)
    return model.predict(datapoints)


# helper function for displaying image to user
def show_image(X_row):
    if type(X_row) is not np.array:
        X_row = np.array(X_row)
    # first, reshape it to a 28x28 pixel matrix
    X_row = X_row.reshape(28, 28)
    # use pyplot to show the image
    plt.imshow(X_row, cmap='Greys')
    plt.show()



#########################
###  EXECUTE PROGRAM  ###
#########################

# load data and do preprocessing
X, y = load_and_preprocess_data(
    images_file='handwritten_digits_images_small.csv',
    labels_file='handwritten_digits_labels_small.csv'
)

# split data into training, validation and test sets
X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split(X, y)
print(X_train[0])

# support vector machine classifier
md = SVC(kernel='linear')
md = md.fit(X_train, y_train)

predicitons = md.predict(X_val)
print(predicitons)

datapoints = [
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    ,183,238,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,182
    ,254,83,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,182,255
    ,83,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,206,239,22
    ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,85,254,234,0,0,0
    ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,135,254,234,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,197,88,3,0,0,0,0,185,254,178,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,254,67,0,0,0,22,243,235,19,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,254,143,0,0,0,133,254,176,0,0,0
    ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,45,243,254,41,0,0,9,205,254,113,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,114,254,230,25,0,0,83,254,242,45
    ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,60,253,254,253,163,146,146,208,
    254,137,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64,250,254,254,254,254,
    254,254,254,254,247,159,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,84,254,246,
    174,187,223,180,248,254,254,251,154,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    24,73,52,0,0,0,38,240,254,117,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,53,254,251,57,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    ,0,81,254,213,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,160,254,
    208,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,72,254,254,86,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,67,254,214,3,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0]]

print('Prediction: ' + str(preprocess_and_predict(md, datapoints)))
