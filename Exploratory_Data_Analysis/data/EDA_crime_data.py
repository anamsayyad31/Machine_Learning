# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 14:54:05 2025

@author: anams
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

#load dataset
df = pd.read_csv('C:/5-python_crisp-ml(q)/understanding_data_assignments_datasets/crime_data.csv')

df.dtypes
#Only state column is categorical rest all are numerical cols 
#Murder, Rape : float 
#rest of the columns are int datatype

#shape, size
print(df.shape) #rows : 50 ,cols : 5
print(df.size) #rows x cols : 250

df.describe() #of num cols only
'''
Murder : dist =Slight right skewed dist. ,range = 16.6 ;  
Assault :  dist =rt. skewed dist. ,range = 292 ;    
UrbanPop : dist =slight left skewed dist. ,range = 59 ; 
Rape : dist =slightly right skewed dist. ,range = 38.7 ; 
'''

#1st moment : mean (central tendency)
mean_values = df.mean(numeric_only=True)
print('\nMean (1st moment):\n', mean_values)
'''
Mean (1st moment):
 Murder        7.788
Assault     170.760
UrbanPop     65.540
Rape         21.232

inference:
1. Assault has high average rate compared to rape & murder.

2. Most of the population (around 66%) are urban, which may correlate 
with higher crime rates due to population density.

3. Murder and Rape have lower means.
'''


var_values = df.var(numeric_only=True)
print('\nVariance (2nd moment): \n', var_values)
'''
Variance (2nd moment): 
 Murder        18.970465
Assault     6945.165714
UrbanPop     209.518776
Rape          87.729159

inference:
variance(how widely features are spread) must be high/medium
  High Variance Features - Assualt
  Moderate Variance Features - UrbanPop, Rape
  Low Variance Features - Murder
  
1. All columns can be used for model development.

2. UrbanPop and Assault cols may need scaling.

3. Assault is the most varied feature, possibly due to regional factors.

4. Murder has low variance, suggesting fewer states with very 
high or low murder rates.

5. UrbanPop and Rape show moderate spread, indicating diversity but not 
extreme outliers.
'''
std_values = df.std(numeric_only=True)
print('\nSD (2nd moment): \n', std_values)
#higher SD means low peakedness 
'''
SD (2nd moment): 
 Murder       4.355510
Assault     83.337661
UrbanPop    14.474763
Rape         9.366385528
dtype: float64

inferences:
1. Higher SD → Lower Peakedness (Platykurtic) :
Assault

2. Lower SD → Higher Peakedness (Leptokurtic) :
Murder 

3.Moderate SD → Mesokurtic :
UrbanPop 
Rape 
'''
#skewness(symmetry)
skew_values = df.skew(numeric_only=True)
print('\nSkewness (3rd moment): \n', skew_values)
#all features have low skewness (close to 0:between −0.5 and +0.5), 
#indicating near-normal dist
'''
Skewness (3rd moment): 
 Murder      0.393956  slight rt skew
Assault     0.234410   slight rt skew
UrbanPop   -0.226009   slight lt skew
Rape        0.801200   rt skew
'''
#4th moment : kurtosis(peakedness)
kurt_values = df.kurtosis(numeric_only=True)
print('\nKurtosis (4th moment): \n', kurt_values)

'''
Murder	-0.827	Platykurtic = flatter than normal
Assault	-1.054	Platykurtic = very flat	
UrbanPop	-0.738	Platykurtic = slightly flat	
Rape	0.354	Mesokurtic = close to normal	
'''
#histogram of each numerical column
sns.histplot(df.Murder, kde = True)   
sns.histplot(df.Assault, kde = True)          
sns.histplot(df.UrbanPop, kde = True)      
sns.histplot(df.Rape, kde = True)  

#combined hidtogram of all numerical columns
df.drop(columns = 'State').hist(figsize=(10,8), color='skyblue', edgecolor='black')    
plt.suptitle('hist of num features')
plt.tight_layout()
plt.show()

'''
1. Murder

Dist: Slightly right-skewed
Range: 0.8 – 17.4
Peak: Around 5–8
Kurtosis: Platykurtic (−0.827)
Insight: Murder rates are generally low with few extreme values. Distribution is 
flat, suggesting uniform spread. May not require transformation but could benefit 
from scaling.

2. Assault

Dist: Right-skewed
Range: 45 – 337
Peak: Around 100–150
Kurtosis: Strongly Platykurtic (−1.054)
Insight: Assault rates vary widely across states. High variance and flat 
distribution indicate regional disparities. Strong candidate for scaling and 
transformation.

3. UrbanPop

Dist: Slightly left-skewed
Range: 32 – 91
Peak: Around 60–70
Kurtosis: Slightly Platykurtic (−0.738)
Insight: Urban population is moderately spread with a slight lean toward lower 
values. Distribution is flat but not extreme. Useful for modeling urban influence 
on crime.

4. Rape

Dist: Moderately right-skewed
Range: 7.3 – 46.0
Peak: Around 20–25
Kurtosis: Mesokurtic (0.354)
Insight: Rape rates show moderate variation and a near-normal distribution. 
Slight skew and moderate tails suggest occasional outliers. Suitable for modeling 
with minimal preprocessing.
'''
# Set plot style
sns.set(style='whitegrid')

# 1. UrbanPop vs Assault
sns.scatterplot(data=df, x='UrbanPop', y='Assault', hue='Murder', palette='coolwarm')
plt.title('UrbanPop vs Assault (Hue: Murder)')
plt.show()
#Slight positive correlation

# 2. UrbanPop vs Rape
sns.scatterplot(data=df, x='UrbanPop', y='Rape', hue='Assault', palette='viridis')
plt.title('UrbanPop vs Rape (Hue: Assault)')
plt.show()
#Weak positive correlation.
 
# 3. Assault vs Murder
sns.scatterplot(data=df, x='Assault', y='Murder', hue='UrbanPop', palette='magma')
plt.title('Assault vs Murder (Hue: UrbanPop)')
plt.show()
#Moderate positive correlation

# 4. Rape vs Murder
sns.scatterplot(data=df, x='Rape', y='Murder', hue='UrbanPop', palette='plasma')
plt.title('Rape vs Murder (Hue: UrbanPop)')
plt.show()
#Weak correlation

# 5. Assault vs Rape
sns.scatterplot(data=df, x='Assault', y='Rape', hue='UrbanPop', palette='cubehelix')
plt.title('Assault vs Rape (Hue: UrbanPop)')
plt.show()
#Strong positive correlation
'''
1. UrbanPop vs Assault (Hue: Murder)
Low UrbanPop (<50):

Assault rates vary but mostly stay below 200.

Murder rates (color intensity) are generally low.

High UrbanPop (>70):

Assault rates often exceed 200.

Murder rates show moderate intensity.

Insight: States with higher urban populations tend to have more assaults. Murder rates don’t spike proportionally, suggesting different dynamics.

2. UrbanPop vs Rape (Hue: Assault)
UrbanPop 50–70:

Rape rates cluster around 15–30.

Assault hue varies, indicating mixed violence levels.

UrbanPop >75:

Rape rates can exceed 35.

Assault hue intensifies, showing higher violence.

Insight: Rape incidents rise slightly with urbanization, and high assault states often show elevated rape rates. Urban density may amplify vulnerability.

3. Assault vs Murder (Hue: UrbanPop)
Assault <150:

Murder rates mostly <10.

UrbanPop hue varies, no strong pattern.

Assault >200:

Murder rates range widely (up to 17).

UrbanPop hue shifts toward higher values.

Insight: High assault states may have higher murder rates, especially in urbanized regions. Suggests possible escalation of violence in dense areas.

4. Rape vs Murder (Hue: UrbanPop)
Rape <20:

Murder rates mostly <10.

UrbanPop hue is mixed.

Rape >30:

Murder rates vary widely.

UrbanPop hue leans high.

Insight: No strong correlation between rape and murder. However, states with high rape rates often have high urban populations, hinting at environmental factors.

5. Assault vs Rape (Hue: UrbanPop)
Assault <150:

Rape rates mostly <25.

UrbanPop hue is moderate.

Assault >200:

Rape rates often >30.

UrbanPop hue intensifies.

Insight: Strong positive correlation. States with high assault rates tend to have high rape rates, especially in urban areas. Indicates shared socio-economic or law enforcement challenges.
'''

# Pie chart for State
df['State'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(8,8))
plt.title('State Distribution')
plt.ylabel('')
plt.show()
#inference : all states are present in equal density(2%) 
