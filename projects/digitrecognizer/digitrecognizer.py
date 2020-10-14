import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# function for loading and reshaping image data
def read_data_and_reshape():
    # read images from file
    X = np.loadtxt(open('handwritten_digits_images.csv', 'rb'), delimiter=',')
    y = np.loadtxt(open('handwritten_digits_labels.csv', 'rb'), delimiter=',')
    # reshape each image
    X = X.reshape(X.shape[0], 28, 28)
    return X, y

# function for displaying image to user
def show_image(img):
    plt.imshow(X[0], cmap='Greys')
    plt.show()



###################
# EXECUTE PROGRAM #
###################

X, y = read_data_and_reshape()