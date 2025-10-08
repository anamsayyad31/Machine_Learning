# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 17:07:46 2025

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
df = pd.read_csv('C:/5-python_crisp-ml(q)/understanding_data_assignments_datasets/heart disease.csv')
print('initial shape:', df.shape) #(303, 14)
df.head()
df.dtypes
#check bin cols
binary_cols = [col for col in df.columns if df[col].nunique() == 2]
print(binary_cols) #['sex', 'fbs', 'exang', 'target']
#don't apply winsorization,scaling, boxcox or log transformation on 
#binary cols
#check 3 class cols
three_class_cols = [col for col in df.columns if df[col].nunique() == 3]
print(three_class_cols) #['restecg', 'slope']
# for multi-class imbalance use encoding 
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
print('after removing duplicates:', df.shape) #removed 1 entry

#outlier treatment
#plot boxplots for each non-bin numeric cols :'age', 'cp', 'trestbps', 
#'chol', 'restecg', 'thalach','oldpeak', 'slope', 'ca', 'thal'
sns.boxplot(df.age)
sns.boxplot(df.cp)
sns.boxplot(df.trestbps) #outliers are present
sns.boxplot(df.chol) #outliers are present
sns.boxplot(df.restecg)
sns.boxplot(df.thalach) #outliers are present
sns.boxplot(df.oldpeak) #outliers are present
sns.boxplot(df.slope)
sns.boxplot(df.ca) #outliers are present
sns.boxplot(df.thal) #outliers are present
from feature_engine.outliers import Winsorizer

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['trestbps']
    )

df = winsor.fit_transform(df)
sns.boxplot(df.trestbps )

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['chol']
    )

df = winsor.fit_transform(df)
sns.boxplot(df.chol )

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['thalach']
    )

df = winsor.fit_transform(df)
sns.boxplot(df.thalach )

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['oldpeak']
    )

df = winsor.fit_transform(df)
sns.boxplot(df.oldpeak )

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['ca']
    )

df = winsor.fit_transform(df)
sns.boxplot(df.ca)

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['thal']
    )

df = winsor.fit_transform(df)
sns.boxplot(df.thal)

#log transformation
#log transformation on :old_peak, trestbps (+vely skewed)
sns.histplot(df.oldpeak, kde=True)
df['oldpeak_log'] = np.log1p(df['oldpeak'])
sns.histplot(df.oldpeak_log, kde=True)

sns.histplot(df.trestbps, kde=True)
df['trestbps_log'] = np.log1p(df['trestbps'])
sns.histplot(df.trestbps_log, kde=True)


#apply one-hot encoding to restecg ,slope cols
df = pd.get_dummies(df, columns=['restecg', 'slope', 'cp'], drop_first=True)
df[df.select_dtypes(include=['bool', 'uint8']).columns] = df.select_dtypes(include=['bool', 'uint8']).astype(int)
print('shape after dummies:', df.shape)

#boxcox transformation (only for strictly +ve values)
#boxcox : age, thal
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method = 'box-cox')

#box-cox works for strictly +ve values 
#apply box-cox to thalach col (slightly left skewed)
#since all values of 'Price' are +ve, apply box-cox 
df['thalach_boxcox'] = pt.fit_transform(df[['thalach']])
#before
sns.histplot(df['thalach'], kde=True).set_title('org. thalach ')
#after
sns.histplot(df['thalach_boxcox'].dropna(), kde=True).set_title('after boxcox thalach ')

#check skewness 
df.hist(figsize=(10,8), color='skyblue', edgecolor='black')    
plt.suptitle('hist of num features')
plt.tight_layout()
plt.show()
#Standardization :age,cp_log,trestbps_log,chol,thalach_boxcox,
#oldpeak_log,ca,thal_boxcox

#Standardization (mean = 0, std = 1)
scaler_std = StandardScaler()
df['age'] = scaler_std.fit_transform(df[['age']])
df['trestbps_log'] = scaler_std.fit_transform(df[['trestbps_log']])
df['chol'] = scaler_std.fit_transform(df[['chol']])
df['thalach_boxcox'] = scaler_std.fit_transform(df[['thalach_boxcox']])
df['oldpeak_log'] = scaler_std.fit_transform(df[['oldpeak_log']])
df['ca'] = scaler_std.fit_transform(df[['ca']])
df['thal'] = scaler_std.fit_transform(df[['thal']])

#zero variance feature removal : no cols with 0 variance

#SMOTE(only apply to continuous features not bin cols)
df.target.value_counts()
'''
target
1    164
0    138
Name: count, dtype: int64
inference : since imbalance is less apply model without smote first.
use logistic regression model

if performance is poor apply class_weight param:
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)
'''
df.drop(columns=['trestbps', 'thalach', 'oldpeak'], inplace=True)

# shift target column to end
target = df.pop("target")
df["target"] = target

df.to_csv('heart disease_prep.csv', index=False)
import os
os.getcwd() 
