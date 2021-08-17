import csv
import math
import numpy as np
from matplotlib import pyplot as plt
import random
import pandas as pd

df=pd.read_csv('carprice.csv',index_col="ID")
df = df.fillna(df.mean())
X = df['Power_bhp']
Y = df['Price']
m=X.values.size
X = np.append(np.ones((m, 1)), X.values.reshape(m, 1), axis=1)
Y = np.array(Y).reshape((len(Y),1))
theta = np.zeros((2, 1))
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