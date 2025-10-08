# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 17:09:48 2025

@author: anams
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

#load dataset
df = pd.read_csv('C:/5-python_crisp-ml(q)/understanding_data_assignments_datasets/Company_Data.csv')
print('initial shape:', df.shape)
#rows : 400 , cols : 11
df.head()

#check missing values
print('missing values:\n',df.isnull().sum())
#no missing values are present

#removing duplicates    
df.drop_duplicates(inplace=True)
print('after removing duplicates:', df.shape)
#no duplicates are present

#outlier treatment
#plot boxplots for each numeric cols
df.dtypes
sns.boxplot(df.CompPrice) #outliers are present
sns.boxplot(df.Income)
sns.boxplot(df.Advertising)
sns.boxplot(df.Population)
sns.boxplot(df.Price) #outliers are present
sns.boxplot(df.Age)
sns.boxplot(df.Education)
sns.boxplot(df.Sales) #outliers are present
from feature_engine.outliers import Winsorizer

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['CompPrice']
    )

df = winsor.fit_transform(df)
sns.boxplot(df['CompPrice'])

#for Price
winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['Price']
    )

df = winsor.fit_transform(df)
sns.boxplot(df['Price'])

#for Sales
winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['Sales']
    )

df = winsor.fit_transform(df)
sns.boxplot(df['Sales'])  

#log transformation
#log transformation on :Advertising (+vely skewed)
sns.histplot(df.Advertising, kde=True)
df['Advertising_log'] = np.log1p(df['Advertising'])
sns.histplot(df.Advertising_log, kde=True)

#boxcox transformation (only for strictly +ve values)
#boxcox : can be used for both skewed
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method = 'box-cox')

#box-cox works for strictly +ve values 
#apply box-cox to Price col (slightly left skewed)
#since all values of 'Price' are +ve, apply box-cox 
df['Price_boxcox'] = pt.fit_transform(df[['Price']])
#before
sns.histplot(df['Price'], kde=True).set_title('org. Price ')
#after
sns.histplot(df['Price_boxcox'].dropna(), kde=True).set_title('after boxcox Price ')


#Standardization (mean = 0, std = 1)
#Apply to all num cols i.e. 'Income', 'Population', 'Price_boxcox', 
#'CompPrice', 'Age', 'Advertising_log', 'Sales', 'Education'
scaler_std = StandardScaler()
df['Income'] = scaler_std.fit_transform(df[['Income']])
df['Population'] = scaler_std.fit_transform(df[['Population']])
df['Price_boxcox'] = scaler_std.fit_transform(df[['Price_boxcox']])
df['Age'] = scaler_std.fit_transform(df[['Age']])
df['Advertising_log'] = scaler_std.fit_transform(df[['Advertising_log']])
df['Sales'] = scaler_std.fit_transform(df[['Sales']])
df['Education'] = scaler_std.fit_transform(df[['Education']])
df['CompPrice'] = scaler_std.fit_transform(df[['CompPrice']])

#zero variance feature removal : no feature with 0 var.

#dummy variables (one-hot encoding)
#convert categorical variables into dummies
# create dummies directly as int (preferred)
df = pd.get_dummies(df, drop_first=True)
df[df.select_dtypes(include=['bool', 'uint8']).columns] = df.select_dtypes(include=['bool', 'uint8']).astype(int)
print('shape after dummies:', df.shape)

#Drop raw columns for which skewness transformation was done.
df.drop(columns=['Price', 'Advertising'], inplace=True)

df.to_csv('Company_Data_prep.csv', index=False)
import os
os.getcwd() 
