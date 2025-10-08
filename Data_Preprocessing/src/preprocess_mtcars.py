# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 18:15:55 2025

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
df = pd.read_csv('C:/5-python_crisp-ml(q)/understanding_data_assignments_datasets/mtcars.csv')
print('initial shape:', df.shape) #(32, 11)
df.head()
df.dtypes
#check bin cols
binary_cols = [col for col in df.columns if df[col].nunique() == 2]
print(binary_cols) #['vs', 'am']
#don't apply winsorization,scaling, boxcox or log transformation on 
#binary cols
'''
One-hot encoding for multi-class categoricals

Scaling for continuous features

SMOTE for target imbalance

StandardScaler : disp, hp, drat, wt, qsec (continous cols)
One-hot encoding : gear(3, 4, 5), carb(1, 2, 3, 4, 6, 8), 
                   cyl(4, 6, 8) (discrete values)
'''
#check missing values
print('missing values:\n',df.isnull().sum())
#no missing values

#removing duplicates    
df.drop_duplicates(inplace=True)
print('after removing duplicates:', df.shape) 
#no duplicates present

#outlier treatment
#plot boxplots for each non-bin numeric cols :mpg, disp, hp, 
#drat, wt, qsec, cyl
#avoid binary or categorical features (am, vs, gear, carb)
sns.boxplot(df.mpg) #outliers are present
sns.boxplot(df.disp)
sns.boxplot(df.hp) #outliers are present
sns.boxplot(df.drat) 
sns.boxplot(df.wt) #outliers are present
sns.boxplot(df.qsec) #outliers are present
sns.boxplot(df.cyl) 
from feature_engine.outliers import Winsorizer

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['mpg']
    )

df = winsor.fit_transform(df)
sns.boxplot(df.mpg )

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['hp']
    )

df = winsor.fit_transform(df)
sns.boxplot(df.hp )

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['wt']
    )

df = winsor.fit_transform(df)
sns.boxplot(df.wt )

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['qsec']
    )

df = winsor.fit_transform(df)
sns.boxplot(df.qsec)



#log transformation
#log transformation on :mpg, disp, hp, drat, qsec (+vely skewed)
sns.histplot(df.mpg, kde=True)
df['mpg_log'] = np.log1p(df['mpg'])
sns.histplot(df.mpg_log, kde=True)

sns.histplot(df.disp, kde=True)
df['disp_log'] = np.log1p(df['disp'])
sns.histplot(df.disp_log, kde=True)

sns.histplot(df.hp, kde=True)
df['hp_log'] = np.log1p(df['hp'])
sns.histplot(df.hp_log, kde=True)

sns.histplot(df.drat, kde=True)
df['drat_log'] = np.log1p(df['drat'])
sns.histplot(df.drat_log, kde=True)

sns.histplot(df.qsec, kde=True)
df['qsec_log'] = np.log1p(df['qsec'])
sns.histplot(df.qsec_log, kde=True)

#drop original columns of log transformed columns 
df.drop(columns=['mpg', 'disp', 'hp', 'drat', 'qsec'], inplace=True)

#apply one-hot encoding to gear, carb, cyl cols
df = pd.get_dummies(df, columns=['gear', 'carb', 'cyl'], drop_first=True)
df[df.select_dtypes(include=['bool', 'uint8']).columns] = df.select_dtypes(include=['bool', 'uint8']).astype(int)
print('shape after dummies:', df.shape)

#check skewness 
df.hist(figsize=(10,8), color='skyblue', edgecolor='black')    
plt.suptitle('hist of num features')
plt.tight_layout()
plt.show()
#Standardization :disp_log, hp_log, drat_log, qsec_log, wt

#Standardization (mean = 0, std = 1)
scaler_std = StandardScaler()
df['disp_log'] = scaler_std.fit_transform(df[['disp_log']])
df['hp_log'] = scaler_std.fit_transform(df[['hp_log']])
df['drat_log'] = scaler_std.fit_transform(df[['drat_log']])
df['wt'] = scaler_std.fit_transform(df[['wt']])
df['qsec_log'] = scaler_std.fit_transform(df[['qsec_log']])

#zero variance feature removal : no cols with 0 variance
#no SMOTE as it is regression dataset
'''
use linear regression model

so finally after preprocessing mtcars dataset for linear regression 
with target col as mpg the cols are as follows : 
    mpg -winsorized,log transformed;	
    cyl-onehot encoding;	
    disp-log transformed,Standardization;	
    hp-winsorized,log transformed,Standardization;	
    drat-log transformed,Standardization;	
    wt-winsorized,Standardization;	
    qsec-winsorized,log transformed,Standardization;	
    vs-as it is(bin int);	
    am-as it is(bin int);	
    gear-onehot encoding;	
    carb-onehot encoding.
'''
df.to_csv('mtcars_prep.csv', index=False)
import os
os.getcwd() 
