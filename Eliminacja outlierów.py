# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:28:47 2020

@author: tm
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline
from sklearn.linear_model import LinearRegression
from scipy import stats
 
cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS',
        'RAD','TAX','PTRATIO','B','LSTAT','MEDV']
 
data = pd.read_csv(r"C:\Users\Tomasz\Desktop\udemy cheatsheet\Machine Learning\Dane Kurs\housing\housing.data",
                   sep=' +', engine='python', header=None, 
                   names=cols)

data1 = data.loc[:,['LSTAT','MEDV']]
len(data1)

plt.figure(figsize = (10,10))
sns.boxplot(data1['LSTAT'])
plt.show()

X = data1['LSTAT'].values.reshape(-1,1)
y = data1['MEDV'].values.reshape(-1,1)


lr1 = LinearRegression()
lr1.fit(X,y)
lr1.score(X,y)

# removing outliers using z-score

z = np.abs(stats.zscore(data1))

threshold = 3


data1_o_z = data1[(z < threshold).all(axis = 1)]
len(data1_o_z)


plt.figure()
sns.barplot(data1_o_z["LSTAT"])
plt.show()


X = data1_o_z['LSTAT'].values.reshape(-1,1)
y = data1_o_z['MEDV'].values.reshape(-1,1)
lr1_o_z = LinearRegression()
lr1_o_z.fit(X,y)
lr1_o_z.score(X,y)


# checking IQR method
Q1 = data1.quantile(0.25)
Q3 = data1.quantile(0.75)
IQR = Q3 - Q1
IQR


outlier_condition = ((data1 < Q1 - 1.5*IQR) | (data1 > Q3 + 1.5 *IQR))

data1_o_iqr = data1[~ outlier_condition.any(axis=1)]
len(data1_o_iqr)


plt.figure()
sns.boxplot(data1_o_iqr["LSTAT"])
plt.show()


X = data1_o_iqr['LSTAT'].values.reshape(-1,1)
y = data1_o_iqr['MEDV'].values.reshape(-1,1)

lr1_o_iqr = LinearRegression()
lr1_o_iqr.fit(X,y)
lr1_o_iqr.score(X,y)
















