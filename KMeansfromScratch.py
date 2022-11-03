#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 10:06:28 2022

@author: weiweitao
"""

################# build k means from scratch
import matplotlib.pyplot as plt
import numpy as np

## inputs
X = np.array([[1, 2],
              [1.5,1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9,11]])

k = 2
tol = 0.001
maxiter = 300

## euclidea function
def euclideandist(x1, x2):
    return np.sum((x1 - x2)**2)

## initialize the center
center = X[:k].copy()
res = {}

## main iteration - update center and result
for iteri in range(maxiter):
    optimized = False
    
    # initialize the result
    for ki in range(k):
        res[ki] = []
        
    ## classify different points to classes
    for xi in X:
        dist = [euclideandist(xi, ci) for ci in center]
        res[np.argmin(dist)].append(xi)
    
    ## update the center position
    precenter = center.copy()
    for ki in range(k):
        center[ki] = sum(res[ki])/len(res[ki])
        
    if euclideandist(precenter, center) < tol**2:
        optimized = True
        break

### plot the results
for c in center:
    plt.scatter(c[0], c[1], marker="o", color="k", s=150, linewidths=5)

colors = ['red', 'blue']
for ki in range(k):
    color = colors[ki]
    for featureset in res[ki]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)
        
plt.show()


################# build KNN from scratch
