from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precisthetaion_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import warnings

—-------
## Avoid printing out warnings
with warnings.catch_warnings():
     warnings.filterwarnings("ignore")
     bc = load_breast_cancer()
     X_main, y_main = bc.data, bc.target


# 1.Explore and import Breast Cancer Wisconsin dataset:
# a)Explore the dataset by using the Scikit Learn library and Numpy
scaler = StandardScaler()
X_main = scaler.fit_transform(X_main)

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X_main, y_main, test_size=0.3, random_state=42)

# Transpose the data to match the input format of weights
X_train = X_train.T
X_test = X_test.T
y_train = y_train.reshape(1, y_train.shape[0])
y_test = y_test.reshape(1, y_test.shape[0])

# 2.In the assignment, you will use gradient descent to find the weights 
# for the logisticregression problem and apply it to the dataset

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss_function (y, y_dash):
    loss = - (y * np.log(y_dash)) - ((1 - y) * np.log(1 - y_dash))
    return loss

def cost_func(y, y_dash):
    m = len(y)
    cost = 0
    for i in range(m):
        cost += loss_function(y[i], y_dash[i])
    cost = cost / m
    return cost

def gradient_descent (x_1, y_1, lr, w,b):
    m, n= x_1.shape
    # Initialize weights and bias to zeros
    weights = np.zeros((n, 1))
    bias = 0
    # Store loss history to visualize convergence
    loss_history = []
    for i in range(iterations):
        # Forward propagation
        z = np.dot(weights.T, x_1) + bias
        y_hat = sigmoid(z)
        # Compute cost/loss
        loss =  loss_function(y, y_hat)
        loss_history.append(loss)
        # Backward propagation (gradient calculation)
        dw = (1/m) * np.dot(X, (y_hat - y).T)
        db = (1/m) * np.sum(y_ hat - y)
        # Update parameters 
        weights =weights - learning_rate * dw
        bias = bias - learning_rate * db
    return weights, bias, loss_history

def predict ():
    return



# 3.The initial hyper-parameters for this assignment are:
# a)Threshold=0.5
# b)Learning rate=0.5
# c)Run your algorithm for 5000 iterations to update weights



# 4.Report the coefficient vector w



# 5.For the test dataset, determine the
# a)Precision
# b)Recall
# c)F1 score
# d)Confusion matrix.

# 6.Plot the log loss on every 100th iteration of your gradient descent, 
# with the iterationnumber on the horizontal axis and the objective value on the vertical axis




#7.Use the test set as a validation set and see if you can find a better setting of thehyperparameters. Report the best values you found.