# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:07:14 2020

@author: tm
"""

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.linear_model import LinearRegression

auto = pd.read_csv(r"C:\Users\Tomasz\Desktop\udemy cheatsheet\Machine Learning\Dane Kurs\auto-mpg\auto-mpg.csv")
auto.head()
auto.shape

X = auto.iloc[:,1:-1]
X = X.drop('horsepower', axis = 1)
y = auto[['mpg']]

lr = LinearRegression()

lr.fit(X,y)
lr.score(X,y)

my_car1 = [4, 160, 190, 12, 90, 1]
my_car2 = [4, 200, 260, 15, 83, 1]
 
cars = [my_car1, my_car2]

mpg_predict = lr.predict(cars)
print(mpg_predict)
