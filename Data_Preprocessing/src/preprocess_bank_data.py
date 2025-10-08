# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 17:06:06 2025

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
df = pd.read_csv('C:/5-python_crisp-ml(q)/understanding_data_assignments_datasets/bank_data.csv')
print('initial shape:', df.shape) #(45211, 32)
df.head()
df.dtypes # all int cols
#check bin cols
binary_cols = [col for col in df.columns if df[col].nunique() == 2]
print(binary_cols) #26 bin int cols
'''
['default', 'housing', 'loan', 'poutfailure', 
 'poutother', 'poutsuccess', 'poutunknown', 'con_cellular', 
 'con_telephone', 'con_unknown', 'divorced', 'married', 
 'single', 'joadmin.', 'joblue.collar', 'joentrepreneur', 
 'johousemaid', 'jomanagement', 'joretired', 'joself.employed', 
 'joservices', 'jostudent', 'jotechnician', 'jounemployed', 
 'jounknown', 'y']

'''
#don't apply winsorization,scaling, boxcox or log transformation on 
#binary cols
'''
One-hot encoding for multi-class categoricals

Scaling for continuous features

SMOTE for target imbalance

continous cols : age, balance,duration
discrete cols :campaign,pdays,previous
'''
#check missing values
print('missing values:\n',df.isnull().sum())
#no missing values

#removing duplicates    
df.drop_duplicates(inplace=True)
print('after removing duplicates:', df.shape) 
#1 duplicate present

#outlier treatment
#plot boxplots for each non-bin numeric cols 
sns.boxplot(df.age) #outliers are present
sns.boxplot(df.balance) #outliers are present
sns.boxplot(df.duration) #outliers are present
sns.boxplot(df.campaign) #outliers are present
sns.boxplot(df.pdays) #outliers are present
sns.boxplot(df.previous) #outliers are present

#winsorization (only for continous data)
from feature_engine.outliers import Winsorizer

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['age']
    )

df = winsor.fit_transform(df)
sns.boxplot(df.age )

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['balance']
    )

df = winsor.fit_transform(df)
sns.boxplot(df.balance )

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['duration']
    )

df = winsor.fit_transform(df)
sns.boxplot(df.duration )

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['campaign']
    )

df = winsor.fit_transform(df)
sns.boxplot(df.campaign)

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['pdays']
    )

df = winsor.fit_transform(df)
sns.boxplot(df.pdays)

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['previous']
    )

df = winsor.fit_transform(df)
sns.boxplot(df.previous)

#check skewness 
df.hist(figsize=(10,8), color='skyblue', edgecolor='black')    
plt.suptitle('hist of num features')
plt.tight_layout()
plt.show()

#log transformation
#log transformation on :age, balance, duration, campaign, pdays, previous
sns.histplot(df.age, kde=True)
df['age_log'] = np.log1p(df['age'])
sns.histplot(df.age_log, kde=True)
print("Any negatives?", (df.previous < 0).sum())
sns.histplot(df.balance, kde=True)
df['balance_log'] = np.log1p(df['balance'])
sns.histplot(df.balance_log, kde=True)

sns.histplot(df.duration, kde=True)
df['duration_log'] = np.log1p(df['duration'])
sns.histplot(df.duration_log, kde=True)

sns.histplot(df.campaign, kde=True)
df['campaign_log'] = np.log1p(df['campaign'])
sns.histplot(df.campaign_log, kde=True)

sns.histplot(df.pdays, kde=True)
df['pdays_log'] = np.log1p(df['pdays'])
sns.histplot(df.pdays_log, kde=True)

sns.histplot(df.previous, kde=True)
df['previous_log'] = np.log1p(df['previous'])
sns.histplot(df.previous_log, kde=True)

#appling one-hot encoding is not required

#Standardization :age, balance, duration (for continous normal distribution)

#Standardization (mean = 0, std = 1)
scaler_std = StandardScaler()
df['age_log'] = scaler_std.fit_transform(df[['age_log']])

df['balance_log'] = scaler_std.fit_transform(df[['balance_log']]) #error due to nan & infs

#since balance_log col contains nans & infs so log transform only positive values
df['balance_log'].isna().sum()        # 3716
np.isinf(df['balance_log']).sum()     # 50

#Log-transform only positive balances
df['balance_log'] = df['balance'].apply(lambda x: np.log(x) if x > 0 else np.nan)
df['balance_log'] = scaler_std.fit_transform(df[['balance_log']])

df['duration_log'] = scaler_std.fit_transform(df[['duration_log']])

#normalization : campaign, pdays, previous (for discrete num cols)
#Normalization (min = 0, max = 1)
scaler_minmax = MinMaxScaler()
df['campaign_log'] = scaler_minmax.fit_transform(df[['campaign_log']])
df['pdays_log'] = scaler_minmax.fit_transform(df[['pdays_log']]) #error due to infs
np.isinf(df['pdays_log']).sum()     # 36953
# Step 1: Replace -1 with NaN before log-transform
df['pdays_log'] = df['pdays'].apply(lambda x: np.log(x) if x > 0 else np.nan)
df['pdays_log'] = scaler_minmax.fit_transform(df[['pdays_log']])

df['previous_log'] = scaler_minmax.fit_transform(df[['previous_log']])

#check missing values
print('missing values:\n',df.isnull().sum())
#drop pdays_log cols with too many missing values
df.drop(columns=['pdays_log'], inplace=True)
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    df[col].fillna(df[col].mean(), inplace=True) 
    
#zero variance feature removal : no cols with 0 variance
#SMOTE 
df.y.value_counts()
from imblearn.over_sampling import SMOTE

# Split X and y first
X = df.drop(['y'], axis=1)
y = df['y']

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print('before:',y.value_counts())
print('after:',pd.Series(y_resampled).value_counts())
