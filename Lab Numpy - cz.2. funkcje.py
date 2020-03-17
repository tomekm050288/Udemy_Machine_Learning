# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 19:40:03 2020

@author: tm
"""
import numpy as np

X = np.arange(1,26).reshape(5,5)
X

Ones = np.ones((5,5))
Ones

result = X * Ones
result

result2 = np.dot(X,Ones)
result2

Ones = np.ones(X.shape)
Ones

diag = np.zeros(X.shape)
diag

np.fill_diagonal(diag, 1)
diag

np.dot(X,diag)
X

np.where(X > 10, 1, 0)

np.where(X % 2 == 0, 1, 0)

np.where(X % 2 == 0, X, X + 1)
X

X_bis = np.where(X > 10, 2*X, 0)
X_bis
np.count_nonzero(X_bis)


x = np.array([[10,20,30], [40,50,60]])
y = np.array([[100], [200]])

print(np.append(x, y, axis=1))
print(np.append(x, x, axis=0))












