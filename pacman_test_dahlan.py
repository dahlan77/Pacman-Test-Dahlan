import csv
import math
import numpy as np
from matplotlib import pyplot as plt
import random
import pandas as pd
def loadcsvdataset(filename):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        headers = dataset[0]
        dataset = dataset[1: len(dataset)]
        return dataset, headers
dataset, headers = loadcsvdataset('advertising.csv')
print("Data Advertise")
print(headers)
print(dataset)
print("Dataset Size")
print(len(dataset), "X", len(dataset[0]))
dataset = np.array(dataset)
dataset = dataset.astype(float)

X = dataset[:,0:-3]
#taking columns with index 2 to 6 as features in X
Y = dataset[:, -1]
#taking the last column i.e. 'price per unit area' as target
one = np.ones((len(X),1))
X = np.append(one, X, axis=1)
#reshape Y to a column vector
Y = np.array(Y).reshape((len(Y),1))
print(X.shape)
print(Y.shape)
def train_test_split(X, Y, split):

    #randomly assigning split% rows to training set and rest to test set
    indices = np.array(range(len(X)))
    
    train_size = round(split * len(X))

    random.shuffle(indices)

    train_indices = indices[0:train_size]
    test_indices = indices[train_size:len(X)]

    X_train = X[train_indices, :]
    X_test = X[test_indices, :]
    Y_train = Y[train_indices, :]
    Y_test = Y[test_indices, :]
    
    return X_train,Y_train, X_test, Y_test
split = 0.8
X_train, Y_train, X_test, Y_test = train_test_split(X, Y, split)

print ("TRAINING SET")
print("X_train.shape: ", X_train.shape)
print("Y_train.shape: ", Y_train.shape)

print("TESTING SET")
print("X_test.shape: ", X_test.shape)
print("Y_test.shape: ", Y_test.shape)

def normal_equation(X, Y):
    beta = np.dot((np.linalg.inv(np.dot(X.T,X))), np.dot(X.T,Y))

    return beta


def predict(X_test, beta):
    return np.dot(X_test, beta)
beta = normal_equation(X_train, Y_train)
predictions = predict(X_test, beta)

print(predictions.shape)
def metrics(predictions, Y_test):

    #calculating mean absolute error
    mae = np.mean(np.abs(predictions-Y_test))

    #calculating root mean square error
    mse = np.square(np.subtract(Y_test,predictions)).mean() 
    rmse = math.sqrt(mse)

    #calculating r_square
    rss = np.sum(np.square((Y_test- predictions)))
    mean = np.mean(Y_test)
    sst = np.sum(np.square(Y_test-mean))
    r_square = 1 - (rss/sst)
    

    return mae, rmse, r_square
mae, rmse, r_square = metrics(predictions, Y_test)
print("Mean Absolute Error: ", mae)
print("Root Mean Square Error: ", rmse)
print("R square: ", r_square)

#GRADIENT DESCENT#

theta = np.zeros((2, 1))
m=200
def cost_function(X, Y, theta):
    y_pred = np.dot(X, theta)
    sqrd_error = (y_pred - Y) ** 2
    cost = 1 / (2 * m) * np.sum(sqrd_error)
    
    return cost
cost_function(X, Y, theta)
def gradient_descent(X, Y, theta, alpha, iter):
    costs = []
    
    for i in range(iter):
        y_pred = np.dot(X, theta)
        
        
        der = np.dot(X.transpose(), (y_pred - Y)) / m
        theta -= alpha * der
        costs.append(cost_function(X, Y, theta))
        
    return theta, costs
theta, costs = gradient_descent(X, Y, theta, alpha=0.000038, iter=4000000)
print("theta = ",theta)
print("cost = ",costs[-1])
y_pred = np.dot(X, np.round(theta, 3))

dic = {'Sales (Actual)': Y.flatten(),
       'Sales (Predicted)': np.round(y_pred, 1).flatten()}

df1 = pd.DataFrame(dic) 

print(df1)
def predict(tv_ads):
    X = np.array([1, tv_ads]).reshape(1, 2)
    y_pred = np.dot(X, theta)
    
    return y_pred[0, 0]
print(predict(220))
R_squared_gradient = 0.9 ** 2
MSE_gradient = ((Y - y_pred) ** 2).sum() / m
RMSE_gradient = np.sqrt(MSE_gradient)
print('R^2 Gradient: ', np.round(R_squared_gradient, 2))
print('RMSE Gradient: ', np.round(RMSE_gradient, 2))
print('MSE Gradient: ', np.round(MSE_gradient, 2))