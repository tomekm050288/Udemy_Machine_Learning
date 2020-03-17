# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 22:02:31 2020

@author: tm
"""

import numpy as np
 
X = np.arange(-25, 25, 1).reshape(10,5)
 
ones = np.ones((X.shape[0], 1))
ones 
X_1 = np.append(X.copy(), ones, axis=1)
X_1 
w = np.random.rand(X_1.shape[1])
 
 
def predict(x, w):     
    total_stimulation = np.dot(x, w)       
    y_pred = 1 if total_stimulation > 0 else -1
    return y_pred

y = np.array([1, -1, -1, 1, -1, 1, -1, -1, 1, -1])
eta = 0.01
