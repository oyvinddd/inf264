import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets, model_selection

# 1. Load the iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target

seed = 666
# separate training data from validation and test
X_train, X_val_test, Y_train, Y_val_test = model_selection.train_test_split(X, Y, test_size=0.7, shuffle=True, random_state=seed)

seed = 221
# separate validation and test data
X_val, X_test, Y_val, Y_test = model_selection.train_test_split(X_train, Y_train, test_size=0.5, shuffle=True, random_state=seed)

# 2. Perform an k-NN classification for each k in 1, 5, 10, 20, 30

N_train = len(X_train)
N_val = len(X_val)
N_test = len(X_test)

print("Length of data sets is as follows: ", N_train, N_val, N_test)

# Plot params
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['red', 'darkcyan', 'darkblue'])

def plot_iris(X_train, Y_train, X_val_test, Y_val_test):
    print(X_val_test[:,0], Y_val_test)
    ax = plt.gca()
    # plot validation and testing points
    ax.scatter(X_val_test[:,0], Y_val_test, c=Y_val_test, cmap=cmap_light, edgecolor='k', s=20, zorder=2)
    # plot the training data in bold colors
    # ax.scatter(X_train, Y_train, c=Y_train, cmap=cmap_bold, edgecolor='k', s=20, zorder=2)
    # add labels to the x/y axxis
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    return ax

# Plot iris data
plot_iris(X_train, Y_train, X_val_test, Y_val_test)