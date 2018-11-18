#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 09:37:08 2018

@author: patrickorourke
"""

# Assignment for the dataset "Auto MPG"

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr   
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split



# Units - "Miles per galllon", "number", "Meters", "unit of power", "Newtons" . "Meters per sec sqr"
columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']


EPOCH = 100
# STEP 1 - GATHERING DATA

# Function to read textfile dataset and load it as a Pandas DataFrame
def loadData(file,columns):
    df = pd.read_table(file, delim_whitespace=True)
    df.columns = columns
    return df

def missingValues(dataset):
    # Identify any missing values in the dataset
    missing = dataset.isnull().sum()  
    #print("Features with missing value: ",missing)
    # Replace any missing value in the dataset with its respective column's mean 
    data.fillna(data.mean(),inplace=True)
    return data

def correlation(data):
    correlation = []
    for i in range(0,7):
        j = pearsonr(data.iloc[:,i],data.iloc[:,9])
        correlation.append(j)
    return correlation

# activation function
def sigmoid(z):
    return 1/(1+np.exp(-z))

#derivative of sigmoid
def sigmoidPrime(s):
  return s * (1 - s)

# forward propagation through our network
def forward(X):
    # dot product of X (input) and first set of weights
    z = np.dot(X, W1)
    # activation function
    forw1 = sigmoid(z)
    # dot product of hidden layer (forw1) and second set of 4x1 weights
    z3 = np.dot(forw1, W2) 
    # final activation function
    forw2 = sigmoid(z3) 
    
    return forw1, forw2

# backward propagate through the network
def backward(X, y, forw1, forw2, W1, W2):
    # error in output
    forw2_error = y - forw2
    # applying derivative of sigmoid to error
    forw2_delta = 2*forw2_error*sigmoidPrime(forw2) 
    
    # adjusting first set (input --> hidden) weights
    
    # z2 error: how much our hidden layer weights contributed to output error
    forw1_error = forw2_delta.dot(W2.T) 
    # applying derivative of sigmoid to z2 error
    forw1_delta = forw1_error*sigmoidPrime(forw1)
        
    # adjusting first set (input --> hidden) weights
    W1 += X.T.dot(forw1_delta) 
    # adjusting second set (hidden --> output) weights
    W2 += forw1.T.dot(forw2_delta)
    
    return forw2_error ** 2
    
def train (X, y):
    forw1, forw2 = forward(X)
    error = backward(X, y, forw1, forw2, W1, W2)
    print("Error is ", error)
    
if __name__ == "__main__":
    
    file = "/Users/patrickorourke/Documents/Auto_MPG/auto_mpg_data_original.txt"
    # Label the columsn of the Pandas DataFrame
    columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']
    data = loadData(file,columns)
    
    # STEP 2 - PREPARING THE DATA
    
    # Examine the dataset
    data.head()
    
    data = missingValues(data)

    # Create additional feature, power-to-weight ratio - higher the value, better the performance of the car
    power_to_weight = data['horsepower']/data['weight']
    
    W1 = np.random.randn(7, 4)
    
    W2 = np.random.randn(4, 1)
    
    # As each column in the Pandas dataframe is a Pandas Series, add the 'power to weight'column with the folowing code, using the existing indexing:
    data['power to weight'] = pd.Series(power_to_weight, index=data.index)
    
    train, test = train_test_split(data, test_size=0.2)
    
    ys_train = np.array(train.iloc[:,9].values)
    
    ys_test = np.array(test.iloc[:,9].values)

    train = train.iloc[:,0:7]
    
    test = test.iloc[:,0:7]

    train_losses, test_losses = [], []
    for e in range(EPOCH):
        
        epoch_train_losses =  []
        for i in range(train.shape[0]):
            
            x = np.array(train.iloc[i].values).reshape(1, 7)
            y = ys_train[i].reshape(1,)
    
            l1, l2 = forward(x)
    
            train_loss = backward(x, y, l1, l2, W1, W2)
            epoch_train_losses.append(train_loss)
        
        train_losses.append(np.mean(epoch_train_losses))
        
        epoch_test_losses = []
        for i in range(test.shape[0]):
            
            x = np.array(train.iloc[i].values).reshape(1, 7)
            y = ys_test[i].reshape(1,)
    
            l1, l2 = forward(x)
            
            test_loss = (y - l2) ** 2
            epoch_test_losses.append(test_loss)
        
        test_losses.append(np.mean(epoch_test_losses))
        
    
    plt.plot(train_losses, label='train')
    plt.plot(test_losses, label='test')
    plt.title('Plot of training and test losses over epochs')
    plt.xlabel("Epochs", 'r')
    plt.ylabel("Epoch Loss",'b')
    plt.legend()      
       
    
    
    
    
    
    
        
        
        
        
    
        
        
        
       
        
        
    
    
    
    
    
    
    
        
        
    
        

        
        
    
        
        
    
    
   
        
    
    
    
