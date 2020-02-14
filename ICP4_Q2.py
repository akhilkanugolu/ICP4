# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:56:59 2020

@author: akhil
"""
import numpy as np
import pandas as pd

glass=pd.read_csv('glass.csv')

X=glass.iloc[:,:-1].values
y=glass.iloc[:,9].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

acc_naive= round(classifier.score(X_train, y_train) * 100, 2)
print("svmaccuracy is:", acc_naive)

from sklearn.metrics import classification_report
classification_report(y_test,y_pred)



