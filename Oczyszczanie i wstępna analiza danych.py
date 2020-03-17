# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 16:49:11 2020

@author: tm
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# można zrezygnować z poniższej linii, żeby wykresy oglądać w osobnym oknie
%matplotlib inline

cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS',
        'RAD','TAX','PTRATIO','B','LSTAT','MEDV']
 
data = pd.read_csv(r"C:\Users\Tomasz\Desktop\udemy cheatsheet\Machine Learning\Dane Kurs\housing\housing.data",
                   sep=' +', engine='python', header=None, 
                   names=cols)

plt.figure()
sns.boxplot(x=data['CHAS'],y=data['MEDV'])
plt.plot()

plt.figure()
sns.barplot(x=data['CHAS'],y=data['MEDV'])
plt.plot()

plt.figure()
sns.barplot(x=data['CRIM'], y=data['MEDV'])
plt.plot()

corr_matrix = np.corrcoef(data.values.T)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(data=corr_matrix,
            annot=True,
            square = True,
            fmt='.2f',
            xticklabels=cols,
            yticklabels=cols)


sns.pairplot(data,height = 1.5)


cols = ['LSTAT', 'RM', 'PTRATIO', 'INDUS','TAX','NOX','MEDV']
 
fig, ax = plt.subplots(figsize=(12,12))
sns.pairplot(data[cols])
plt.show()





