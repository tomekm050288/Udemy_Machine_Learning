# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 21:43:48 2020

@author: tm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


 data = pd.read_csv(r"C:\Users\Tomasz\Desktop\udemy cheatsheet\Machine Learning\Dane Kurs\high-school-sat-gpa\high_school_sat_gpa.csv",
                    sep=' ', usecols=['math_SAT','verb_SAT','high_GPA'])
     
 data.head()
 data.dtypes
 
 plt.figure(figsize=(10,6))
 plt.scatter(data['math_SAT'],data['high_GPA'],color = 'red')
 plt.show()
 
lr_math = LinearRegression()
lr_math.fit(data['math_SAT'].values.reshape(-1,1), data['high_GPA'].values.reshape(-1,1))

x_min = data['math_SAT'].min()
x_min
x_max = data['math_SAT'].max()
x_max

 plt.figure(figsize=(10,6))
 plt.scatter(data['math_SAT'],data['high_GPA'],color = 'red')
 plt.plot([x_min,x_max],lr_math.predict([[x_min],[x_max]]), color='red')
 plt.show()
 
 
plt.figure(figsize=(10,6))
plt.scatter(data['math_SAT'],data['verb_SAT'],color = 'red')
plt.show()
 
lr_math = LinearRegression()
lr_math.fit(data['math_SAT'].values.reshape(-1,1), data['verb_SAT'].values.reshape(-1,1))

x_min = data['math_SAT'].min()
x_min
x_max = data['math_SAT'].max()
x_max

plt.figure(figsize=(10,6))
plt.scatter(data['math_SAT'],data['verb_SAT'],color = 'red')
plt.plot([x_min,x_max],lr_math.predict([[x_min],[x_max]]), color='red')
plt.show()
 
student_john = np.array([600,650]).reshape(1,2)

 
lr = LinearRegression()
lr.fit(data[['math_SAT','verb_SAT']].values, data['high_GPA'].values.reshape(-1,1))
 
student_john = np.array([600,650]).reshape(1,2)
lr.predict(student_john)






















 