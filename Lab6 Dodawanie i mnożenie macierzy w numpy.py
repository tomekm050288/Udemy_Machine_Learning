# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:28:51 2020

@author: tm
"""


import time
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
 
 
num_iterations = 100
time_results_loop = []
 
for iteration in range(1, num_iterations+1):
    
    start_time = time.time()
    
    data = np.arange(0,10000*iteration, 1)
    
    my_sum = 0
    for i in data:
        my_sum += i
    
    end_time = time.time()
    
    print('{} - :{}'.format(iteration, end_time - start_time))    
    time_results_loop.append(end_time - start_time)
    
print(time_results_loop)


num_iterations = 100
time_results_np = []
 
for iteration in range(1, num_iterations+1):
 
    start_time = time.time()
    
    data = np.arange(0,10000*iteration, 1)
    my_sum = np.sum(data)
    
    end_time = time.time()
    
    print('{} - :{}'.format(iteration, end_time - start_time))    
    time_results_np.append(end_time - start_time)



a = np.arange(0.1,1,0.1)
a
np.sum(a)


fig = plt.figure()
plt.scatter(range(num_iterations), time_results_loop, s=10, c='b', marker="s", label='loop')
plt.scatter(range(num_iterations), time_results_np, s=10, c='r', marker="o", label='numpy')
plt.legend(loc='upper left');
plt.show()



num_iterations = 100
time_results_loop = []
 
for iteration in range(1, num_iterations+1):
    
    start_time = time.time()
    
    data1 = np.ones(shape=(10*iteration, 10*iteration), dtype=np.float)
    data2 = np.ones(shape=(10*iteration, 10*iteration), dtype=np.float)
    data3 = np.zeros(shape=(10*iteration, 10*iteration), dtype=np.float)
    
    for i in range(data3.shape[0]):
        for j in range(data3.shape[1]):
            data3[i,j] = data1[i,j] + data2[i,j]
    
    end_time = time.time()
    
    print('{} - :{}'.format(iteration, end_time - start_time))    
    time_results_loop.append(end_time - start_time)
    
print(time_results_loop)



num_iterations = 100
time_results_np = []
 
for iteration in range(1, num_iterations+1):
 
    start_time = time.time()
    
    data1 = np.ones(shape=(10*iteration, 10*iteration), dtype=np.float)
    data2 = np.ones(shape=(10*iteration, 10*iteration), dtype=np.float)
    data3 = np.zeros(shape=(10*iteration, 10*iteration), dtype=np.float)
    
    data3 = data1 + data2
    
    end_time = time.time()
    
    print('{} - :{}'.format(iteration, end_time - start_time))    
    time_results_np.append(end_time - start_time)




num_iterations = 40
time_results_loop = []
 
for iteration in range(1, num_iterations+1):
    
    start_time = time.time()
    
    data1 = np.ones(shape=(10*iteration, 10*iteration), dtype=np.float)
    data2 = np.ones(shape=(10*iteration, 10*iteration), dtype=np.float)
    data3 = np.zeros(shape=(10*iteration, 10*iteration), dtype=np.float)
    
    for i in range(data1.shape[0]):
        for j in range(data1.shape[1]):
            data3[i,j] = sum([data1[i, v] * data2[v, j] for v in range(data1.shape[1])])
    end_time = time.time()
    
    print('{} - :{}'.format(iteration, end_time - start_time))    
    time_results_loop.append(end_time - start_time)
    
   
num_iterations = 100
time_results_np = []
 
for iteration in range(1, num_iterations+1):
 
    start_time = time.time()
    
    data1 = np.ones(shape=(10*iteration, 10*iteration), dtype=np.float)
    data2 = np.ones(shape=(10*iteration, 10*iteration), dtype=np.float)
    data3 = np.zeros(shape=(10*iteration, 10*iteration), dtype=np.float)
    
    data3 = data1.dot(data2)
    
    end_time = time.time()
    
    print('{} - :{}'.format(iteration, end_time - start_time))    
    time_results_np.append(end_time - start_time)
