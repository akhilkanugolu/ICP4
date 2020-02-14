# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:43:20 2020

@author: akhil
"""
import pandas as pd
train=pd.read_csv("train_preprocessed.csv")
test=pd.read_csv("test_preprocessed.csv")
correlation = train.corr(method='pearson')

#