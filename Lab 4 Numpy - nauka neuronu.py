# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 21:52:01 2020

@author: tm
"""
import numpy as np
 
X = np.arange(-25, 25, 1).reshape(10,5)
X 
ones = np.ones((X.shape[0], 1))
ones 
X_1 = np.append(X.copy(), ones, axis=1)
X_1 
w = np.random.rand(X_1.shape[1])
w
 
 
def predict(x, w):     
    total_stimulation = np.dot(x, w)       
    y_pred = 1 if total_stimulation > 0 else -1
    return y_pred

y = np.array([1, -1, -1, 1, -1, 1, -1, -1, 1, -1])
eta = 0.01


epochs = 10

for i in range(epochs):
    for x, y_target in zip(X_1,y):
        y_pred = predict(x,w)
        delta_w = eta*(y_target - y_pred)*x
        w += delta_w
        print(w)









