# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 21:04:28 2020

@author: tm
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

data = pd.read_csv(r"C:\Users\Tomasz\Desktop\udemy cheatsheet\Machine Learning\Dane Kurs\breast_cancer.csv")

X = data[['area_mean', 'area_se', 'texture_mean', 'concavity_worst', 'concavity_mean']]
y = data['diagnosis']
y = y.apply(lambda x: 1 if x == 'M' else 0)

scaler = StandardScaler()

scaler.fit(X)
X_std = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2)

perceptron = Perceptron(eta0=0.01, max_iter=100)

perceptron.fit(X_train,y_train)

y_pred = perceptron.predict(X_test)


good = y_test[y_test == y_pred].count()
good
total = y_test.count()
total
print('result: {}'.format(100*good/total))
