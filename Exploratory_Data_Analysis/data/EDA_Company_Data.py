# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 18:07:20 2025

@author: anams
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

#load dataset
df = pd.read_csv('C:/5-python_crisp-ml(q)/understanding_data_assignments_datasets/Company_Data.csv')

df.dtypes
#ShelveLoc, Urban, US columns are categorical  
#Sales : float 
#rest of the columns are int datatype

#shape, size
print(df.shape) #rows : 400 ,cols : 11
print(df.size) #rows x cols : 4400

df.describe()
'''
Sales : dist =normal dist. ,range = 16.27 ;  
CompPrice :  dist =normal dist. ,range = 98 ;    
Income : dist =normal dist. ,range = 99 ; 
Advertising : dist =slightly right skewed dist. ,range = 29 ; 
Population : dist =slightly left skewed dist. ,range = 499 ;        
Price : dist =normal dist. ,range = 164 ;        
Age : dist =normal dist. ,range = 55 ;  
Education : dist =normal dist. ,range = 8
'''

#1st moment : mean (central tendency)
mean_values = df.mean(numeric_only=True)
print('\nMean (1st moment):\n', mean_values)
'''
Mean (1st moment):
 Sales            7.496325
CompPrice      124.975000
Income          68.657500
Advertising      6.635000
Population     264.840000
Price          115.795000
Age             53.322500
Education       13.900000

inference:
1. price (115.80) is lower than competitor price (125.00) → competitive 
pricing strategy
2. Income is moderate (68.66) → customers may be price-conscious
3. Ad cost is relatively low (6.64) vs. Sales (7.50) → check if low 
spend still yields good returns(an opportunity to optimize ad budget)
'''


var_values = df.var(numeric_only=True)
print('\nVariance (2nd moment): \n', var_values)
'''
Variance (2nd moment): 
 Sales              7.975626
CompPrice        235.147243
Income           783.218239
Advertising       44.227343
Population     21719.813935
Price            560.584436
Age              262.449618
Education          6.867168

inference:
variance(how widely features are spread) must be high/medium
  High Variance Features - Population, Income, Price
  Moderate Variance Features - Advertising, Age, CompPrice
  Low Variance Features - Sales, Education
1. Education can't be used for model development as variance is too low
2. Sales is target variable so can't be used as a feature
3. Rest all numerical cols can be used for model but Population column may
   need scaling
'''
std_values = df.std(numeric_only=True)
print('\nSD (2nd moment): \n', std_values)
#higher SD low peakedness 
'''
SD (2nd moment): 
 Sales            2.824115
CompPrice       15.334512
Income          27.986037
Advertising      6.650364
Population     147.376436
Price           23.676664
Age             16.200297
Education        2.620528
dtype: float64

inferences:
1. Higher SD → Lower Peakedness (Platykurtic) :
Population (147.38)
Income (27.99)
Price (23.68)

2. Lower SD → Higher Peakedness (Leptokurtic) :
Education (2.62)
Sales (2.82)   

3.Moderate SD → Mesokurtic :
Advertising (6.65)
Age (16.20)    
CompPrice (15.33)
'''
#skewness(symmetry)
skew_values = df.skew(numeric_only=True)
print('\nSkewness (3rd moment): \n', skew_values)
#all features have low skewness (close to 0:between −0.5 and +0.5), 
#indicating near-normal dist
'''
Skewness (3rd moment): 
 Sales          0.185560  slight rt skew
CompPrice     -0.042755   nearly symmetric
Income         0.049444   nearly symmetric
Advertising    0.639586   slight rt skew
Population    -0.051227   nearly symmetric
Price         -0.125286   nearly symmetric
Age           -0.077182   nearly symmetric
Education      0.044007   nearly symmetric
'''
#4th moment : kurtosis(peakedness)
kurt_values = df.kurtosis(numeric_only=True)
print('\nKurtosis (4th moment): \n', kurt_values)

'''
Sales −0.080877 → mesokurtic = close to normal 
CompPrice 0.041666 → mesokurtic = close to normal 
Income −1.085289 → strongly platykurtic = very flat 
Advertising −0.545118 → platykurtic = flatter distribution 
Population −1.202318 → strongly platykurtic = very flat 
Price 0.451885 → mesokurtic = close to normal 
Age −1.134392 → strongly platykurtic = very flat 
Education −1.298332 → strongly platykurtic = very flat
'''

sns.histplot(df.Sales, kde = True)   
sns.histplot(df.CompPrice, kde = True)           
sns.histplot(df.Income, kde = True)       
sns.histplot(df.Advertising, kde = True) #right skewed
sns.histplot(df.Population, kde = True)   
sns.histplot(df.Price, kde = True)  #slight lt skwed        
sns.histplot(df.Age, kde = True)       
sns.histplot(df.Education, kde = True)   


df.drop(columns = ['ShelveLoc', 'Urban', 'US']).hist(figsize=(10,8), color='skyblue', edgecolor='black')    
plt.suptitle('hist of num features')
plt.tight_layout()
plt.show()

'''
1.Sales

Dist: Nearly symmetric
Range: 0.00 – 16.27
Peak: Around 7.5
Kurtosis: Mesokurtic 
Insight: Sales show moderate variation and a fairly normal shape. 
Useful as a target variable, not a feature.

2.CompPrice

Dist: Nearly symmetric
Range: 77 – 175
Peak: Around 125
Kurtosis: Mesokurtic
Insight: Competitor pricing is centered and consistent. 
Can be a strong predictor for pricing strategy.

3.Income

Dist: Nearly symmetric
Range: 21 – 120
Peak: Around 65–70
Kurtosis: Strongly platykurtic
Insight: Income is widely spread with a flat distribution. 
May require scaling, useful for understanding customer affordability.

4.Advertising

Dist: Slightly right-skewed
Range: 0 – 29
Peak: Around 5–10
Kurtosis: Platykurtic
Insight: Low ad spend with wide spread. Opportunity to optimize 
budget for better ROI.

5.Population

Dist: Nearly symmetric
Range: 10 – 509
Peak: Broad
Kurtosis: Strongly platykurtic
Insight: Very high variance and flat distribution. Needs scaling, 
may influence regional targeting.

6.Price

Dist: Slightly left-skewed
Range: 24 – 188
Peak: Around 115
Kurtosis: Mesokurtic
Insight: Pricing is well-distributed and competitive. Useful for 
modeling customer response.

7.Age

Dist: Nearly symmetric
Range: 25 – 80
Peak: Around 50–55
Kurtosis: Strongly platykurtic
Insight: Age is broadly spread. May correlate with product preferences 
or brand loyalty.

8.Education

Dist: Nearly symmetric
Range: 10 – 18
Peak: Around 14
Kurtosis: Strongly platykurtic
Insight: Low variance and flat distribution. Not useful for modeling 
due to limited variance.
'''
#correlation matrix
sns.heatmap(df.corr(numeric_only=True),annot=True)
'''
CompPrice vs Price(0.58) : strong +ve correlation
Population vs Ad(0.27) : +ve correlation
Income vs Sales(0.15) : weak +ve correlation
Price vs Sales(-0.44) : -ve correlation
Age vs Sales(-0.23) : weak -ve correlation
'''
sns.set(style='whitegrid')

#1.Scatter Plot: Price vs Sales
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Price', y='Sales', data=df, hue='ShelveLoc')
plt.title('Price vs Sales')
plt.show()
#Negative correlation

#2.Scatter Plot: Advertising vs Sales
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Advertising', y='Sales', data=df)
plt.title('Advertising vs Sales')
plt.show()
#positive correlation

#3.Scatter Plot: Income vs Sales
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Income', y='Sales', data=df)
plt.title('Income vs Sales')
plt.show()
#weak positive correlation

#4.Scatter Plot: CompPrice vs Price
plt.figure(figsize=(6, 4))
sns.scatterplot(x='CompPrice', y='Price', data=df)
plt.title('CompPrice vs Price')
plt.show()
#strong positive correlation

#5.Scatter Plot: Age vs Sales
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Age', y='Sales', data=df)
plt.title('Age vs Sales')
plt.show()
#weak negative correlation
'''
1. Price vs Sales (hue: ShelveLoc):
indicates good shelf placement boosts sales irrespective of price.
shelveloc has positive correlation with sales.

2. Advertising vs Sales:
low ad spend (<10) shows wide spread in Sales.
high ad spend (>15) has higher Sales.

4. CompPrice vs Price:
most prices are close to competitor prices.

5. Age vs Sales:
Age <40: Sales slightly higher.
Age >60: Sales generally lower → older customers are less responsive.
'''
# Count plot for ShelveLoc
sns.countplot(x='ShelveLoc', data=df, palette='pastel')
plt.title('Product Count by Shelf Location')
plt.show()
#inference : shelf location medium is double than bad & good which 
#are nearly equal

# Bar plot for Urban
urban_counts = df['Urban'].value_counts()
urban_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Store Distribution: Urban vs Non-Urban')
plt.xlabel('Urban')
plt.ylabel('Count')
plt.show()
#inference : location of store in urban area is double that of non_urban 
#regions so major revenue is generated in urban regions

# Pie chart for US
us_counts = df['US'].value_counts()
us_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
plt.title('Store Distribution: US vs Non-US')
plt.ylabel('')
plt.show()
#inference : almost 65% stores are present in US