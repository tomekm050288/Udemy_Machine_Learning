# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 17:44:14 2020

@author: tm
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#from scipy import stats


cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS',
        'RAD','TAX','PTRATIO','B','LSTAT','MEDV']
 
data = pd.read_csv(r"C:\Users\Tomasz\Desktop\udemy cheatsheet\Machine Learning\Dane Kurs\housing\housing.data",
                   sep=' +', engine='python', header=None, 
                   names=cols)

data = data.loc[:,['LSTAT','MEDV']]

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3-Q1


outlier_condition = ((data < Q1 - 1.5*IQR) | (data > Q3 + 1.5 *IQR))
data = data[~ outlier_condition.any(axis=1)]
len(data)

X = data["LSTAT"].values.reshape(-1,1)
y = data["MEDV"].values.reshape(-1,1)

plt.figure()
plt.scatter(X,y)
plt.plot()

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

scaler.fit(y)
y = scaler.transform(y)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

lr = LinearRegression()
lr.fit(X_train,y_train)

y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

from sklearn.metrics import mean_absolute_error

mae_train = mean_absolute_error(y_train,y_pred_train)
mae_train

mae_test = mean_absolute_error(y_test,y_pred_test)
mae_test


from sklearn.metrics import mean_squared_error

mse_train = mean_squared_error(y_train,y_pred_train)
mse_train

mse_test = mean_squared_error(y_test,y_pred_test)
mse_test


from sklearn.metrics import r2_score

r2_train = r2_score(y_train,y_pred_train)
r2_test = r2_score(y_test,y_pred_test)
r2_train
r2_test

# bez eliminacji outlierów

cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS',
        'RAD','TAX','PTRATIO','B','LSTAT','MEDV']
 
data = pd.read_csv(r"C:\Users\Tomasz\Desktop\udemy cheatsheet\Machine Learning\Dane Kurs\housing\housing.data",
                   sep=' +', engine='python', header=None, 
                   names=cols)

data = data.loc[:,['LSTAT','MEDV']]


X = data["LSTAT"].values.reshape(-1,1)
y = data["MEDV"].values.reshape(-1,1)

plt.figure()
plt.scatter(X,y)
plt.plot()


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

lr = LinearRegression()
lr.fit(X_train,y_train)

y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

from sklearn.metrics import mean_absolute_error

mae_train = mean_absolute_error(y_train,y_pred_train)
mae_train

mae_test = mean_absolute_error(y_test,y_pred_test)
mae_test


from sklearn.metrics import mean_squared_error

mse_train = mean_squared_error(y_train,y_pred_train)
mse_train

mse_test = mean_squared_error(y_test,y_pred_test)
mse_test


from sklearn.metrics import r2_score

r2_train = r2_score(y_train,y_pred_train)
r2_test = r2_score(y_test,y_pred_test)
r2_train
r2_test



# wszystkie kolumny ze skalowniem i eliminacją outlierów

cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS',
        'RAD','TAX','PTRATIO','B','LSTAT','MEDV']
 
data = pd.read_csv(r"C:\Users\Tomasz\Desktop\udemy cheatsheet\Machine Learning\Dane Kurs\housing\housing.data",
                   sep=' +', engine='python', header=None, 
                   names=cols)


Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3-Q1


outlier_condition = ((data < Q1 - 1.5*IQR) | (data > Q3 + 1.5 *IQR))
data = data[~ outlier_condition.any(axis=1)]
len(data)

X = data.loc[:, data.columns!='MEDV']
y = data['MEDV'].values.reshape(-1,1)

plt.figure()
plt.scatter(X,y)
plt.plot()

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

scaler.fit(y)
y = scaler.transform(y)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

lr = LinearRegression()
lr.fit(X_train,y_train)

y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

from sklearn.metrics import mean_absolute_error

mae_train = mean_absolute_error(y_train,y_pred_train)
mae_train

mae_test = mean_absolute_error(y_test,y_pred_test)
mae_test


from sklearn.metrics import mean_squared_error

mse_train = mean_squared_error(y_train,y_pred_train)
mse_train

mse_test = mean_squared_error(y_test,y_pred_test)
mse_test


from sklearn.metrics import r2_score

r2_train = r2_score(y_train,y_pred_train)
r2_test = r2_score(y_test,y_pred_test)
r2_train
r2_test



# wszystkie kolumny bez skalowania i eliminacji outlierów

cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS',
        'RAD','TAX','PTRATIO','B','LSTAT','MEDV']
 
data = pd.read_csv(r"C:\Users\Tomasz\Desktop\udemy cheatsheet\Machine Learning\Dane Kurs\housing\housing.data",
                   sep=' +', engine='python', header=None, 
                   names=cols)



X = data.loc[:, data.columns!='MEDV']
y = data['MEDV'].values.reshape(-1,1)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

lr = LinearRegression()
lr.fit(X_train,y_train)

y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

from sklearn.metrics import mean_absolute_error

mae_train = mean_absolute_error(y_train,y_pred_train)
mae_train

mae_test = mean_absolute_error(y_test,y_pred_test)
mae_test


from sklearn.metrics import mean_squared_error

mse_train = mean_squared_error(y_train,y_pred_train)
mse_train

mse_test = mean_squared_error(y_test,y_pred_test)
mse_test


from sklearn.metrics import r2_score

r2_train = r2_score(y_train,y_pred_train)
r2_test = r2_score(y_test,y_pred_test)
r2_train
r2_test










