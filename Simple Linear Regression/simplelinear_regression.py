# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 16:22:47 2018

@author: sabar
"""
#Simple regression
#importing libraries
import numpy as np #mainly used for performing all mathematical operations
import matplotlib.pyplot as plt #used to plot all charts in python. pylpot is a sublibrary in matplotlib
import pandas as pd #to manage datasets and import datasets

#import datasets
dataset = pd.read_csv('Salary_Data.csv')
#X is the indepedent variable , matrix
X = dataset.iloc[:, :-1].values #-1indicates all the columns we areconsidering except last one,iloc is used to represent the integer position from 0 to length -1
#Matrix of vectors for the dependent variables
y = dataset.iloc[:, 1].values # '3' is the index of the purchased column,values get all the items in the dataset/excel

#Splitting the train set and the dataset
from sklearn.cross_validation import train_test_split
X_train_reg,X_test_reg,y_train_reg,y_test_reg = train_test_split(X,y,test_size = 1/3, random_state = 0)

#feature scaling#Mainly done inorder to avoid domainant values 
#between the X,y variables and done using finding euclidian distance
#estimating standard deviation,normlization

#import the library Standardscalar
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
#Use fit transform for the X variable(tRAIN) and only transform for the test(AS ITS ALREADY FIT)
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Import linear model library from sklearn.This is used to make the 
#machin elaern / predict the salary values based on the number of
#experiene the user has.
from sklearn.linear_model import LinearRegression
#regressor is the meachine which learn the salary and the experience
#LR is the function which calls the object itself
regressor = LinearRegression() #to learn the corerelations
#Now to fit the values of features X_train and vector values y_train 
#to understand the machine learn the salary
regressor.fit(X_train_reg,y_train_reg) #fit is method to fit the traing set


#predicting the test results
#y_pred = regressor.predict(X_test_reg)
y_pred = regressor.predict(X_test_reg)

#Visualizing the traing test results
plt.scatter(X_train_reg,y_train_reg, color = 'red')
plt.plot(X_train,regressor.predict(X_test_reg), color = 'blue')
plt.title('Salary vs Experience')
plt.xlabel('Experience')
plt.ylabel('Slary')
plt.show()

#Visualizing the test results
plt.scatter(X_test_reg,y_test_reg, color = 'red')
plt.plot(X_train,regressor.predict(X_test_reg), color = 'blue')
plt.title('Salary vs Experience')
plt.xlabel('Experience')
plt.ylabel('Slary')
plt.show()

