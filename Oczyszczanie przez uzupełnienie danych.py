# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 17:12:44 2020

@author: tm
"""

import pandas as pd
import numpy as np

diam_org = pd.read_csv(r"C:\Users\Tomasz\Desktop\udemy cheatsheet\Machine Learning\Dane Kurs\diamonds\diamonds.csv", usecols = ['color','price'])

diam = diam_org.copy()

diam_mean_original = diam.groupby(by = 'color').mean()

import random

missing_data = random.sample(range(0,diam.shape[0]),5)
diam.loc[missing_data]

diam.loc[missing_data,'price'] = np.NaN

diam.loc[missing_data]

diam_mean_start = diam.groupby(by = 'color').mean()

diff = diam_mean_original - diam_mean_start

filter_nan = diam['price'].isnull()
diam.loc[filter_nan,:]

diam.loc[filter_nan, 'color'].map(diam_mean_start['price'])
diam.loc[filter_nan, 'price'] = diam.loc[filter_nan, 'color'].map(diam_mean_start['price'])
# checking results of 'repair' operation
diam.loc[filter_nan, 'price']


diam_mean_repair = diam.groupby(by = 'color').mean()
diff2 = diam_mean_start - diam_mean_repair
diff3 = diam_mean_original - diam_mean_repair
diff4 = diam_mean_original - diam_mean_start

######################################################################
diam2 = diam_org.copy()
diam_mean_start2 = diam2['price'].mean()

diam2.loc[missing_data,'price'] = np.NaN
diam_mean_start2 = diam2.groupby(by = 'color').mean()
filter_nan2 = diam2['price'].isnull()
diam2.loc[filter_nan2, 'price'] = diam_mean_start2
diam_mean_repair2 = diam2.groupby(by = 'color').mean()


diam_mean_original - diam_mean_start
diam_mean_start2 - diam_mean_repair2
diam_mean_original - diam_mean_start2
diam_mean_original - diam_mean_repair2













