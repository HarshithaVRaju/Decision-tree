# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:30:36 2022

@author: Harshitha
"""
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
dataset=load_iris()
print(dataset.data)
print(dataset.data.shape)
print(dataset.target)
X=pd.DataFrame(dataset.data,columns=dataset.feature_names)
X
Y=dataset.target
Y
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X,Y,test_size=0.25,random_state=0)
print(xtrain.shape)
print(xtest.shape)
accuracy=[]
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 

for i in range(1,10):
    model = DecisionTreeClassifier(max_depth = i, random_state= 0 )
    model.fit(xtrain, ytrain)
    pred = model.predict(xtest)
    print(accuracy_score(ytest, pred))
#from sklearn.svm import SVC
#model=SVC(kernel='linear')
model.fit(xtrain,ytrain)
pred=model.predict(xtest)
from sklearn.metrics import accuracy_score,classification_report
print(accuracy_score(ytest, pred))
classification_report(ytest,pred)
new["x"].corr()
