# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 17:08:14 2025

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

#load titanic dataset
df = pd.read_csv('C:/5-python_crisp-ml(q)/understanding_data_assignments_datasets/crime_data.csv')
print('initial shape:', df.shape) #(50, 5)
df.head()
df.dtypes
'''
One-hot encoding for multi-class categoricals

Scaling for continuous features

SMOTE for target imbalance

'''
#check missing values
print('missing values:\n',df.isnull().sum())
#no missing values

#removing duplicates    
df.drop_duplicates(inplace=True)
print('after removing duplicates:', df.shape) 
#no duplicates present

#outlier treatment
#plot boxplots for each non-bin numeric cols : Murder,Assault,UrbanPop,Rape  
#avoid binary or categorical features (State)
sns.boxplot(df.Murder) 
sns.boxplot(df.Assault)
sns.boxplot(df.UrbanPop) 
sns.boxplot(df.Rape ) #outliers are present

from feature_engine.outliers import Winsorizer

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['Rape']
    )

df = winsor.fit_transform(df)
sns.boxplot(df.Rape  )

#check skewness 
df.hist(figsize=(10,8), color='skyblue', edgecolor='black')    
plt.suptitle('hist of num features')
plt.tight_layout()
plt.show()

#log transformation
#log transformation on :Rape (+vely skewed)

sns.histplot(df.Rape , kde=True) 
df['Rape_log'] = np.log1p(df['Rape'])
sns.histplot(df.Rape_log, kde=True)

#Standardization : all num cols

#Standardization (mean = 0, std = 1)
scaler_std = StandardScaler()
df['Murder'] = scaler_std.fit_transform(df[['Murder']])
df['Assault'] = scaler_std.fit_transform(df[['Assault']])
df['Rape_log'] = scaler_std.fit_transform(df[['Rape_log']])
df['UrbanPop'] = scaler_std.fit_transform(df[['UrbanPop']])

#zero variance feature removal : no cols with 0 variance
df.drop(columns=['Rape'], inplace=True)

df.to_csv('crime_data_prep.csv',encoding='utf-8')
import os
os.getcwd() 
