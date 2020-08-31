import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets, model_selection

# 1. Load the iris dataset
iris = datasets.load_iris()     # load iris dataset
X = iris.data[:, :2]            # store the first two features
Y = iris.target                 # store the labels

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
print("Datapoints used for training:   ", N_train)
print("Datapoints used for validation: ", N_val)
print("Datapoints used for testing:   ", N_test)

# Plot params
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['red', 'darkcyan', 'darkblue'])



def plot_iris(X_train, Y_train, X_val_test, Y_val_test):
    ax = plt.gca()
    # plot validation and testing points
    ax.scatter(X_val_test[:,0], X_val_test[:,1], c=Y_val_test, cmap=cmap_light, edgecolor='k', s=20, zorder=2)
    # plot the training data in bold colors
    ax.scatter(X_train[:,0], X_train[:,1], c=Y_train, cmap=cmap_bold, edgecolor='k', s=20, zorder=2)
    # add labels to the x/y axxis
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    return ax



def draw_knn_boundaries(knn, h=0.02): # h = Step size in the mesh
    """
    Draw boundaries as decided by the trained knn
    """
    ax = plt.gca()
    [xmin, xmax] = ax.get_xlim()
    [ymin, ymax] = ax.get_ylim()
    # Generate the axis associated to the first feature: 
    x_axis = np.arange(xmin, xmax, h)
    # Generate the axis associated to the 2nd feature: 
    y_axis = np.arange(ymin, ymax, h)
    # Generate a meshgrid (2D grid) from the 2 axis:
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    # Vectorize the grids into column vectors:
    x_grid_vectorized = x_grid.flatten()
    x_grid_vectorized = np.expand_dims(x_grid_vectorized, axis=1)
    y_grid_vectorized = y_grid.flatten()
    y_grid_vectorized = np.expand_dims(y_grid_vectorized, axis=1)
    # Concatenate the vectorized grids
    grid = np.concatenate((x_grid_vectorized, y_grid_vectorized), axis=1)
    # Now you can use 'grid' as data to classify by the knn 

    # Predict concatenated features to get the decision boundaries:
    decision_boundaries = ... #TODO!

    # Reshape the decision boundaries into a 2D matrix:
    decision_boundaries = decision_boundaries.reshape(x_grid.shape)
    plt.pcolormesh(x_grid, y_grid, decision_boundaries, cmap=cmap_light, zorder=1)
    return ax



# Main work here:
def knn_on_iris(k, X_train, Y_train, X_val, Y_val):
    """
    Train a knn and plot its boundaries on iris data
    """

    # --------------------
    # Plot iris data
    # --------------------
    plot_iris(X_train, Y_train, X_val, Y_val)

    # --------------------
    # Train the knn
    # --------------------

    # Create an instance of the KNeighborsClassifier class for current value of k:
    k_NN = KNeighborsClassifier(n_neighbors=k)
    # Train the classifier with the training data
    k_NN.fit(X_train, Y_train)

    # --------------------
    # Draw knn boundaries
    # --------------------
    draw_knn_boundaries(k_NN)
    plt.title("k-NN classification on Iris, k = " + str(k_NN.get_params().get("n_neighbors")))
    plt.show()

    # --------------------
    # Model accuracy:
    # --------------------

    # Accuracy on train set:
    train_predictions = k_NN.predict(X_train)
    good_train_predictions = (train_predictions == Y_train)
    train_accuracy = np.sum(good_train_predictions) / len(X_train)
    # Accuracy on test set:
    val_predictions = k_NN.predict(X_val)
    good_val_predictions = (val_predictions == Y_val)
    val_accuracy = np.sum(good_val_predictions) / len(X_val)
    
    return (k_NN, train_accuracy, val_accuracy)



### k-NN on the Iris dataset for different values of k:
# Create vectors to store the results for each k:
train_accuracies = []
val_accuracies = []

# Train a knn for each value of k in k_list
k_list = [1, 5, 10, 20, 30]
for k in k_list:
    knn, train_acc, val_acc = knn_on_iris(k, X_train, Y_train, X_val, Y_val)
    print("K-nn trained with k = ", k)
    print("Train accuracy: ", train_acc, " ----- ", "Validation accuracy: ", val_acc)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

# Plot accuracy curves:
plt.plot(k_list, train_accuracies)
plt.plot(k_list, val_accuracies)
plt.ylim(0, 1)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='best')
plt.title("k-NN accuracy curves on Iris")

# Display plots:
plt.show()