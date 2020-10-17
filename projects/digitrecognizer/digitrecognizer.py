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
# - preprocess data by setting pixels to either 1 or 0 based on a threshold
# - this way we can use 28*28 binary features
# - probably won't need to reshape before we want to show image to user
# https://www.youtube.com/watch?v=7aK7tVdcVbY&ab_channel=CodeHeroku



def load_and_preprocess_data():
    # read images and their respective classes from csv
    X = np.loadtxt(open('handwritten_digits_images.csv', 'rb'), delimiter=',')
    y = np.loadtxt(open('handwritten_digits_labels.csv', 'rb'), delimiter=',')
    # preprocess each image by going through each pixel and giving it the value
    # of 1 for values higher than 255/2 and 0 for values lower. This way we can
    # treat each pixel in the image as a binary feature in our model.
    for row in X:
        for index, col in enumerate(row):
            if row[index] > 255/2:
                row[index] = 1
            else:
                row[index] = 0
    # reshape the data set to pixel matrices
    # if reshape:
    #     X = X.reshape(X.shape[0], 28, 28)
    return X, y

# helper function for displaying image to user
def show_image(X_row):
    # first, reshape it to a 28x28 pixel matrix
    X_row = X_row.reshape(28, 28)
    # use pyplot to show the image
    plt.imshow(X_row, cmap='Greys')
    plt.show()

def create_smaller_files():
    images, labels = [], []
    counter = 0
    with open('handwritten_digits_images.csv', 'r') as reader:
        for line in reader.readlines():
            if counter % 10 == 0:
                images.append(line)
            counter += 1
    with open('handwritten_digits_images_small.csv', 'w') as writer:
        writer.writelines(images)
    
    counter = 0
    with open('handwritten_digits_labels.csv', 'r') as reader:
        for line in reader.readlines():
            if counter % 10 == 0:
                labels.append(line)
            counter += 1
    with open('handwritten_digits_labels_small.csv', 'w') as writer:
        writer.writelines(labels)


###################
# EXECUTE PROGRAM #
###################

create_smaller_files()
exit(1)
X, y = load_and_preprocess_data()

# do train, validation test split
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=667)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, shuffle=True, random_state=668)

# support vector machine classifier
md = SVC(kernel='linear')
md = md.fit(X_train, y_train)

predicitons = md.predict(X_val)
