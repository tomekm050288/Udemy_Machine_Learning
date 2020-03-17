# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:37:39 2020

@author: tm
"""


import numpy as np
 
data = np.array([[10, 7, 4], [3, 2, 1]])
data

np.mean(data)
np.mean(data,axis=0)
np.mean(data,axis=1)

np.average(data)
np.average(data,axis = 0)
np.average(data,axis = 1)

np.average(data,axis=1,weights=[0,1,1])
np.average(data,axis=1,weights=[2,3,5])

np.var(data)
np.var(data,axis = 0)
np.var(data,axis = 1)

np.std(data)
np.std(data,axis = 0)
np.std(data,axis = 1)

data = np.zeros((2, 1000000))
data[0, :] = 1.0
data[1, :] = 0.1
np.mean(data, dtype=np.float32)
np.mean(data, dtype=np.float64)


data = np.zeros((2, 10))
data[0, :] = 1.0
data[1, :] = 0.1
np.mean(data, dtype=np.float32)
np.mean(data, dtype=np.float64)


