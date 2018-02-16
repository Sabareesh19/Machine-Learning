# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:05:23 2018

@author: sabar
"""

#importing libraries
import numpy as np #mainly used for performing all mathematical operations
import matplotlib.pyplot as plt #used to plot all charts in python. pylpot is a sublibrary in matplotlib
import pandas as pd #to manage datasets and import datasets

#import datasets
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values #-1indicates all the columns we areconsidering except last one,iloc is used to represent the integer position from 0 to length -1
#importing the dependent variable for the last column
y = dataset.iloc[:, 3].values # '3' is the index of the purchased column,values get all the items in the dataset/excel

#missing values
from sklearn.preprocessing import Imputer #Imputer is the class that handles the missing data
#create the object fro the class
impute = Imputer(missing_values="NaN",strategy="median",axis=0)
#to fit in the cloums where the data is mssing
impute.fit(X[:,1:3]) #HERE 3 SPECIFIES THE LAST column<dont get confused we are specifying the upper bound>
#to perform the mean function
X[:,1:3] = impute.transform(X[:,1:3] ) #transform computes the mean

#Categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#The label encoder is used to convert the text into numbers
labelencoder_X = LabelEncoder()
#fittransform used to fit the column that need to to be converted to numbers
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#creating object for onehotencoder
onehotencoder = OneHotEncoder(categorical_features = [0])
#to break one column into 3 different columns to avoid ambiguity
X = onehotencoder.fit_transform(X).toarray()
#The index 2 i.e 3 need to be broken similarly to differentiate between yes and no.
labelencoder_y = LabelEncoder()
y= labelencoder_y.fit_transform(y)

#Splitting the train set and the dataset
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

#feature scaling#Mainly done inorder to avoid domainant values 
#between the X,y variables and done using finding euclidian distance
#estimating standard deviation,normlization

#import the library Standardscalar
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
#Use fit transform for the X variable(tRAIN) and only transform for the test(AS ITS ALREADY FIT)
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
 
