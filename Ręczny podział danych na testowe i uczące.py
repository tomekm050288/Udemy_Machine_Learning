# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 16:07:30 2020

@author: tm
"""

import numpy as np

arr = np.arange(5,30,2)
arr

boolArr = arr < 10
boolArr

newArr = arr[boolArr]
newArr

newArr = arr[arr < 20]
newArr

newArr = arr[arr % 3 == 0]
newArr

newArr = arr[arr > 10 & arr < 20]
newArr

arr = np.arange(24).reshape(4,6)
arr

arr[0]
arr[0][1]
arr[0][2:3]
arr[0,2:4]
arr[0,:]
arr[:,1]
arr[0:2,3]
arr[:2,2]
arr[:2,2:3]
arr[:,-1]

arr = np.arange(50).reshape(10,5)
arr

split_level = 0.2
num_rows = arr.shape[0]
num_rows
split_border = split_level * num_rows
split_border

X = arr[:int(split_border),:]
y = arr[int(split_border):,:]


np.random.shuffle(arr)

arr

y = arr[:int(split_border),:]
X = arr[int(split_border):,:]


np.random.shuffle(arr)
arr
 
arr[:round(split_border), :]
arr[round(split_border):, :]

# tworzenie zbiorów target i feature

data = np.arange(500).reshape(100,5)
data
np.random.shuffle(data)
data
 
split_level = 0.2
num_rows = data.shape[0]
split_border = split_level * num_rows 
split_border 
X_train = data[round(split_border):,:-1]
X_test  = data[:round(split_border),:-1]
y_train = data[round(split_border):,-1]
y_test  = data[:round(split_border),-1]


from sklearn.model_selection import train_test_split
 
X_train, X_test, y_train, y_test = train_test_split(
        data[:, :-1], data[:, -1], test_size=0.2, shuffle = True)


data.shape
X_train.shape
X_test.shape
y_train.shape
y_test.shape

