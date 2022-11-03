#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 10:33:39 2022

@author: weiweitao
"""

import numpy as np
import pandas as pd
import random
from statistics import mode

def train_test_split(df, ratio, targetvar):
    n = df.shape[0]
    ntrain = int(np.ceil(ratio*n))

    trainidx = random.sample(range(n), ntrain)
    testidx = list(set(range(n)) - set(trainidx))

    traindf = df.iloc[trainidx]
    testdf = df.iloc[testidx]
    
    return traindf.drop(columns = targetvar), traindf[targetvar], testdf.drop(columns = targetvar), testdf[targetvar] 

def euclideandist(x1, x2):
    return np.sum((x1 - x2)**2)

def knn(xi, Xtrain, num_neighbors):
    dist = Xtrain.apply(lambda x: euclideandist(x, xi), axis = 1)
    ypred = mode(ytrain[list(dist.sort_values()[:num_neighbors].index.values)])
    return ypred

def accuracy(ypred, y):
    return np.mean(np.array(ypred) == np.array(y))
    
df = pd.read_csv('./data/iris.txt', sep = ',')
num_neighbors = 5
Xtrain, ytrain, Xtest, ytest = train_test_split(df, 0.7, 'Iris-setosa')

ypred = []
for i in range(Xtest.shape[0]):
    ypred.append(knn(Xtest.iloc[i], Xtrain,  num_neighbors))
    
print('The prediction accuracy is: ', accuracy(ypred, ytest))
    




