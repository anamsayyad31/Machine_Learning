# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 17:04:50 2025

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
df = pd.read_excel('C:/5-python_crisp-ml(q)/understanding_data_assignments_datasets/EastWestAirlines.xlsx')
print('initial shape:', df.shape) #(3999, 12)
df.head()

#check missing values
print('missing values:\n',df.isnull().sum())
#no missing values

#removing duplicates    
df.drop_duplicates(inplace=True)
print('after removing duplicates:', df.shape)
#no duplicates

df.dtypes
#outlier treatment
#plot boxplots for each numeric cols
sns.boxplot(df.Balance ) #outliers are present
sns.boxplot(df.cc1_miles )
sns.boxplot(df.cc2_miles) #outliers are present
sns.boxplot(df.cc3_miles) #outliers are present
sns.boxplot(df.Bonus_miles) #outliers are present
sns.boxplot(df.Bonus_trans ) #outliers are present
sns.boxplot(df.Flight_miles_12mo) #outliers are present
sns.boxplot(df.Flight_trans_12 ) #outliers are present
sns.boxplot(df.Days_since_enroll)

#drop cols with low var.
df.drop(columns=['cc2_miles', 'cc3_miles'], inplace=True)

from feature_engine.outliers import Winsorizer

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['Balance']
    )

df = winsor.fit_transform(df)
sns.boxplot(df.Balance )

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['Bonus_miles']
    )

df = winsor.fit_transform(df)
sns.boxplot(df.Bonus_miles )  

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['Bonus_trans']
    )

df = winsor.fit_transform(df)
sns.boxplot(df.Bonus_trans)

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['Flight_miles_12mo']
    )

df = winsor.fit_transform(df)
sns.boxplot(df.Flight_miles_12mo)

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['Flight_trans_12']
)

df = winsor.fit_transform(df)
sns.boxplot(df.Flight_trans_12 )

#since Input columns ['Qual_miles'] have low variation for method 'iqr'
#use log transformation instead
sns.histplot(df.Qual_miles, kde=True)
df['Qual_miles_log'] = np.log1p(df['Qual_miles'])
sns.histplot(df.Qual_miles_log, kde=True)

#check skewness 
df.hist(figsize=(10,8), color='skyblue', edgecolor='black')    
plt.suptitle('hist of num features')
plt.tight_layout()
plt.show()

#log transformation
#log transformation on :Balance, cc1_miles, Bonus_trans, Flight_trans_12 (+vely skewed)
sns.histplot(df.Balance, kde=True)
df['Balance_log'] = np.log1p(df['Balance'])
sns.histplot(df.Balance_log, kde=True)

sns.histplot(df.cc1_miles, kde=True)
df['cc1_miles_log'] = np.log1p(df['cc1_miles'])
sns.histplot(df.cc1_miles_log, kde=True)

sns.histplot(df.Bonus_trans, kde=True)
df['Bonus_trans_log'] = np.log1p(df['Bonus_trans'])
sns.histplot(df.Bonus_trans_log, kde=True)

sns.histplot(df.Flight_trans_12, kde=True)
df['Flight_trans_12_log'] = np.log1p(df['Flight_trans_12'])
sns.histplot(df.Flight_trans_12_log, kde=True)

#Normalization & Standardization
#Standardization (mean = 0, std = 1) :Balance_log, cc1_miles_log, 
#Bonus_trans_log, Flight_trans_12_log, Qual_miles_log,Bonus_miles, 
#Flight_miles_12mo, Days_since_enroll 
scaler_std = StandardScaler()
df['Balance_log'] = scaler_std.fit_transform(df[['Balance_log']])
df['cc1_miles_log'] = scaler_std.fit_transform(df[['cc1_miles_log']])
df['Bonus_trans_log'] = scaler_std.fit_transform(df[['Bonus_trans_log']])
df['Flight_trans_12_log'] = scaler_std.fit_transform(df[['Flight_trans_12_log']])
df['Qual_miles_log'] = scaler_std.fit_transform(df[['Qual_miles_log']])
df['Bonus_miles'] = scaler_std.fit_transform(df[['Bonus_miles']])
df['Flight_miles_12mo'] = scaler_std.fit_transform(df[['Flight_miles_12mo']])
df['Days_since_enroll'] = scaler_std.fit_transform(df[['Days_since_enroll']])

#zero variance feature removal : no col with 0 var.

#SMOTE
from imblearn.over_sampling import SMOTE

# Split X and y first
X = df.drop(['Award?', 'ID#'], axis=1)
y = df['Award?']

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print('before:',y.value_counts())
print('after:',pd.Series(y_resampled).value_counts())

df.drop(columns=['Balance', 'cc1_miles', 'Bonus_trans', 'Flight_trans_12', 'Qual_miles'], inplace=True)

# Move 'Award?' to the end
award_col = df.pop('Award?')
df['Award?'] = award_col

df.to_excel('EastWestAirlines_prep.xlsx', index=False)
import os
os.getcwd() 

