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

# —-------
## Avoid printing out warnings
with warnings.catch_warnings():
     warnings.filterwarnings("ignore")
     bc = load_breast_cancer()
     X_main, y_main = bc.data, bc.target


# 1.Explore and import Breast Cancer Wisconsin dataset:
# a)Explore the dataset by using the Scikit Learn library and Numpy
scaler = StandardScaler()
X_main = scaler.fit_transform(X_main)
#Add columns of 1 to account for bias, and then you will only need the theta
X_b = np.c_[np.ones((X_main.shape[0], 1)), X_main]

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X_b, y_b, test_size=0.3, random_state=42)
print (X_train)
print (X_test)
print(y_train)
print(y_test)

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

def gradient_descent (x_1, y_1, lr,iterations):
    m, n= x_1.shape
    # Initialize weights and bias to zeros
    weights = np.zeros((n, 1))
    #bias = 0
    # Store loss history to visualize convergence
    loss_history = []
    for i in range(iterations):
        # Forward propagation
        z = np.dot(x_1, weights.T) 
        y_hat = sigmoid(z)
        # Compute cost/loss
        loss = loss_function(y, y_hat)
        loss_history.append(loss)
        # Backward propagation (gradient calculation)
        gradients = (1/m) * np.dot(x_1.T, (y_hat - y_1))
        # Update parameters 
        weights =weights - lr * gradients
    return weights, loss_history

def predict(X, w):
    z = np.dot(X, w)
    A = sigmoid(z)
    return (A >= 0.5).astype(int)

# 3.The initial hyper-parameters for this assignment are:
# a)Threshold=0.5
# b)Learning rate=0.5
# c)Run your algorithm for 5000 iterations to update weights
lr1=0.5
iterations1=5000
w_result, loss_result = gradient_descent (X_train, y_train, lr1,iterations1)
binary_predictions = predict(X_test,w_result)

# 4.Report the coefficient vector w
print ("The coefficient vector w was : ", w_result)

# 5.For the test dataset, determine the
# a)Precision
# b)Recall
# c)F1 score
# d)Confusion matrix..
# Evaluate the model - Make sure the correct variables are passed 
precision = precision_score(y_test, binary_predictions)
recall = recall_score(y_test, binary_predictions)
f1 = f1_score(y_test, binary_predictions)
accuracy = accuracy_score(y_test, binary_predictions)
conf_matrix = confusion_matrix(y_test, binary_predictions)
# Display the results
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")

# 6.Plot the log loss on every 100th iteration of your gradient descent, 
# with the iterationnumber on the horizontal axis and the objective value on the vertical axis
# Plot the log loss over iterations
plt.plot(range(0, iterations, 100), log_losses)
plt.xlabel("Iterations")
plt.ylabel("Log Loss")
plt.title("Log Loss over Iterations")
plt.show()  

#7.Use the test set as a validation set and see if you can find a better setting of thehyperparameters. Report the best values you found.
w_result1, loss_result1 = gradient_descent (X_test, y_test, lr1,iterations1)
binary_predictions1 = predict(X_train,w_result1)
print ("The coefficient vector w was : ", w_result1)

precision = precision_score(y_train, binary_predictions1)
recall = recall_score(y_train, binary_predictions1)
f1 = f1_score(y_train, binary_prediction1s)
accuracy = accuracy_score(y_train, binary_predictions1)
conf_matrix = confusion_matrix(y_train, binary_predictions1)

