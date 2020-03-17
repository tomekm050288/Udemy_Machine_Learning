# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 21:30:30 2020

@author: tm
"""
import numpy as np

X = np.arange(-25,25).reshape(10,5)
X
ones = np.ones((X.shape[0],1))
ones

X_1 = np.append(X,ones,axis=1)
X_1

w = np.random.rand(X_1.shape[1])
w

def predict(x,w):
    total_stimulation = np.dot(x, w)
    if total_stimulation > 0:
        return 1
    return 0

predict(w, X_1[0,])

for row in range(X_1.shape[0]):
    print(predict(w,X_1[row]))
    
for x in X_1:
    y_pred = predict(x, w)
    print(y_pred)