# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 12:36:45 2018

@author: Hitesh
"""
#Data PreProcessing
#Importing the Libraries
import numpy as np #library that contain mathematical tools
import matplotlib.pyplot as plt #Library useful for plotting charts
import pandas as pd #Library to import and manage datasets 

#IMPORTING THE DATASETS
dataset=pd.read_csv('Data.csv')
#creating feature matrix 
X=dataset.iloc[:,:-1].values# all the lines , all columns except last column hence -1
Y=dataset.iloc[:,3].values #dependent variable hence index- 3rd column

#Taking care of Missing Data
from sklearn.preprocessing import Imputer #Class that allows replacing missing data
imputer=Imputer(missing_values="NaN",
                strategy="mean",
                axis=0) #(value to replace,strategy,axis 0 means Impute along column,1 means along rows ) 

imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

#Categorial Data
#data where categories are present like country and purchased in our data.
#we need to encode text labels to numerical labels

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
#LabelEncoder just transforms country to 0 and  1 
#and hence machine learning model equation can think that countries have different order which would be wrong
#and hence we use OneHotEncoder

onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()