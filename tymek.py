# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:42:34 2023

@author: tymot
"""

import pandas as pd

train = pd.read_csv('training.csv') # split for test and train set is ready
test = pd.read_csv('test.csv')




train.info()
train.head()


train["PurchDate"] = pd.to_datetime(train["PurchDate"])





