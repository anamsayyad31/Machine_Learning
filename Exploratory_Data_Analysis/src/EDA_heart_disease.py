# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 17:15:21 2025

@author: anams
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

#load dataset
df = pd.read_csv('C:/5-python_crisp-ml(q)/understanding_data_assignments_datasets/heart disease.csv')

df.dtypes
#Only oldpeak column is float datatype
#rest of the columns are int datatype

#shape, size
print(df.shape) #rows : 303 ,cols : 14
print(df.size) #rows x cols : 4242

df.describe() #of num cols only
'''
age       : dist = slight right skewed dist. ,range = 48.0 ;  
sex       : dist = bimodal binary dist. ,range = 1.0 ;  
cp        : dist = right skewed dist. ,range = 3.0 ;  
trestbps  : dist = slight right skewed dist. ,range = 106.0 ;  
chol      : dist = right skewed dist. ,range = 438.0 ;  
fbs       : dist = highly right skewed dist. ,range = 1.0 ;  
restecg   : dist = multimodal categorical dist. ,range = 2.0 ;  
thalach   : dist = slight left skewed dist. ,range = 131.0 ;  
exang     : dist = bimodal binary dist. ,range = 1.0 ;  
oldpeak   : dist = right skewed dist. ,range = 6.2 ;  
slope     : dist = multimodal categorical dist. ,range = 2.0 ;  
ca        : dist = right skewed dist. ,range = 4.0 ;  
thal      : dist = multimodal categorical dist. ,range = 3.0 ;  
target    : dist = bimodal binary dist. ,range = 1.0 ;  
'''

#1st moment : mean (central tendency)
mean_values = df.mean(numeric_only=True)
print('\nMean (1st moment):\n', mean_values)
'''
Mean (1st moment):
 age          54.366337
sex           0.683168
cp            0.966997
trestbps    131.623762
chol        246.264026
fbs           0.148515
restecg       0.528053
thalach     149.646865
exang         0.326733
oldpeak       1.039604
slope         1.399340
ca            0.729373
thal          2.313531
target        0.544554
dtype: float64


inference:
1. Average age of patients is around 54 years, indicating middle-aged individuals 
are most affected.

2. Majority of patients are male (mean sex ≈ 0.68), suggesting higher prevalence 
in men.

3. Mean cholesterol level (≈ 246) is relatively high, which may be a key risk 
factor.

4. Average resting blood pressure (≈ 131) is slightly elevated, hinting at common 
hypertension.

5. Mean thalach (≈ 150) shows moderately high max heart rate achieved, useful for 
cardiac stress analysis.

6. Mean value of target ≈ 0.54 implies that over half the patients in the dataset 
have heart disease.

7. Features like fbs, exang, and ca have low mean values, suggesting these 
conditions are less common across the population.
'''


var_values = df.var(numeric_only=True)
print('\nVariance (2nd moment): \n', var_values)
'''
Variance (2nd moment): 
 age           82.484558
sex            0.217166
cp             1.065132
trestbps     307.586453
chol        2686.426748
fbs            0.126877
restecg        0.276528
thalach      524.646406
exang          0.220707
oldpeak        1.348095
slope          0.379735
ca             1.045724
thal           0.374883
target         0.248836
dtype: float64

inference:
variance (how widely features are spread) must be high/medium  
  High Variance Features - chol, thalach, trestbps  
  Moderate Variance Features - age, oldpeak, cp, ca, slope  
  Low Variance Features - sex, fbs, restecg, exang, thal, target  

1. High variance in chol, thalach, and trestbps suggests strong 
individual differences in cholesterol, heart rate, and blood pressure.

2. Moderate variance in age, cp, oldpeak, ca, and slope indicates a 
balanced spread, useful for stratified modeling.

3. Low variance in binary and categorical features like sex, fbs, 
restecg, exang, thal, and target implies limited diversity, but they 
may still be predictive.

4. All features show measurable spread and can be considered for 
model development, though scaling may be needed for high-variance 
features.
'''
std_values = df.std(numeric_only=True)
print('\nSD (2nd moment): \n', std_values)
'''
SD (2nd moment): 
 age          9.082101
sex          0.466011
cp           1.032052
trestbps    17.538143
chol        51.830751
fbs          0.356198
restecg      0.525860
thalach     22.905161
exang        0.469794
oldpeak      1.161075
slope        0.616226
ca           1.022606
thal         0.612277
target       0.498835
dtype: float64


inference:
1. Higher SD → Lower Peakedness (Platykurtic) :  
   chol, thalach, trestbps  

2. Lower SD → Higher Peakedness (Leptokurtic) :  
   sex, fbs, exang, target  

3. Moderate SD → Mesokurtic :  
   age, cp, oldpeak, restecg, slope, ca, thal  
'''
#skewness(symmetry)
skew_values = df.skew(numeric_only=True)
print('\nSkewness (3rd moment): \n', skew_values)
'''
Skewness (3rd moment):  
age        -0.202463   slight lt skew  
sex        -0.791335   moderate lt skew  
cp          0.484732   slight rt skew  
trestbps    0.713768   moderate rt skew  
chol        1.143401   rt skew  
fbs         1.986652   strong rt skew  
restecg     0.162522   near symmetric  
thalach    -0.537410   moderate lt skew  
exang       0.742532   moderate rt skew  
oldpeak     1.269720   rt skew  
slope      -0.508316   moderate lt skew  
ca          1.310422   rt skew  
thal       -0.476722   moderate lt skew  
target     -0.179821   slight lt skew  
'''
#4th moment : kurtosis(peakedness)
kurt_values = df.kurtosis(numeric_only=True)
print('\nKurtosis (4th moment): \n', kurt_values)

'''
Kurtosis (4th moment):  
age        -0.542167   Platykurtic = slightly flat  
sex        -1.382961   Platykurtic = very flat  
cp         -1.193071   Platykurtic = very flat  
trestbps    0.929054   Mesokurtic = close to normal  
chol        4.505423   Leptokurtic = sharply peaked  
fbs         1.959678   Leptokurtic = moderately peaked  
restecg    -1.362673   Platykurtic = very flat  
thalach    -0.061970   Mesokurtic = close to normal  
exang      -1.458317   Platykurtic = very flat  
oldpeak     1.575813   Leptokurtic = moderately peaked  
slope      -0.627521   Platykurtic = slightly flat  
ca          0.839253   Mesokurtic = close to normal  
thal        0.297915   Mesokurtic = close to normal  
target     -1.980783   Platykurtic = extremely flat  
'''
sns.histplot(df.age, kde=True)
sns.histplot(df.sex, kde=True)
sns.histplot(df.cp, kde=True)
sns.histplot(df.trestbps, kde=True)
sns.histplot(df.chol, kde=True)
sns.histplot(df.fbs, kde=True)
sns.histplot(df.restecg, kde=True)
sns.histplot(df.thalach, kde=True)
sns.histplot(df.exang, kde=True)
sns.histplot(df.oldpeak, kde=True)
sns.histplot(df.slope, kde=True)
sns.histplot(df.ca, kde=True)
sns.histplot(df.thal, kde=True)
sns.histplot(df.target, kde=True)

df.hist(figsize=(10,8), color='skyblue', edgecolor='black')    
plt.suptitle('hist of num features')
plt.tight_layout()
plt.show()

'''
1. age  
Dist: Slightly right-skewed  
Range: 29 – 77  
Peak: Around 50–60  
Kurtosis: Platykurtic (−0.54)  
Insight: Age is moderately spread with a slight lean toward older individuals. 
Distribution is flat, suggesting uniform aging across patients. May benefit from 
scaling but not transformation.

2. sex  
Dist: Moderate left-skewed (binary)  
Range: 0 – 1  
Peak: At 1 (male)  
Kurtosis: Very Platykurtic (−1.38)  
Insight: Majority are male. Flat distribution due to binary nature. No transformation 
needed; useful for stratified analysis.

3. cp (chest pain type)  
Dist: Right-skewed  
Range: 0 – 3  
Peak: At 0  
Kurtosis: Very Platykurtic (−1.19)  
Insight: Most patients report typical angina. Distribution is flat and categorical. 
Encoding may be more useful than transformation.

4. trestbps (resting BP)  
Dist: Moderate right-skewed  
Range: 94 – 200  
Peak: Around 120–140  
Kurtosis: Mesokurtic (0.93)  
Insight: Blood pressure varies moderately. Distribution is near-normal with some 
outliers. Scaling recommended.

5. chol (cholesterol)  
Dist: Right-skewed  
Range: 126 – 564  
Peak: Around 200–250  
Kurtosis: Leptokurtic (4.50)  
Insight: Cholesterol shows strong outliers and sharp peak. High variance and 
peakedness make it a candidate for log transformation.

6. fbs (fasting blood sugar)  
Dist: Strong right-skewed  
Range: 0 – 1  
Peak: At 0  
Kurtosis: Leptokurtic (1.96)  
Insight: Most patients have normal fasting sugar. Binary with sharp peak. No 
transformation needed.

7. restecg  
Dist: Near symmetric  
Range: 0 – 2  
Peak: At 1  
Kurtosis: Very Platykurtic (−1.36)  
Insight: ECG results are evenly spread across categories. Flat distribution; 
categorical encoding preferred.

8. thalach (max heart rate)  
Dist: Moderate left-skewed  
Range: 71 – 202  
Peak: Around 140–160  
Kurtosis: Mesokurtic (−0.06)  
Insight: Heart rate is moderately spread and near-normal. Suitable for modeling 
with minimal preprocessing.

9. exang (exercise-induced angina)  
Dist: Moderate right-skewed  
Range: 0 – 1  
Peak: At 0  
Kurtosis: Very Platykurtic (−1.46)  
Insight: Most patients do not experience angina during exercise. Binary and flat; 
no transformation needed.

10. oldpeak  
Dist: Right-skewed  
Range: 0.0 – 6.2  
Peak: Around 0–1  
Kurtosis: Leptokurtic (1.57)  
Insight: ST depression shows sharp peak and long tail. May benefit from log or 
Box-Cox transformation.

11. slope  
Dist: Moderate left-skewed  
Range: 0 – 2  
Peak: At 2  
Kurtosis: Slightly Platykurtic (−0.63)  
Insight: Slope of ST segment is moderately flat. Categorical encoding preferred.

12. ca (major vessels)  
Dist: Right-skewed  
Range: 0 – 4  
Peak: At 0  
Kurtosis: Mesokurtic (0.84)  
Insight: Most patients have no major vessels colored. Moderate spread; may 
benefit from ordinal encoding.

13. thal  
Dist: Moderate left-skewed  
Range: 0 – 3  
Peak: At 2  
Kurtosis: Mesokurtic (0.29)  
Insight: Thalassemia types are moderately distributed. Encoding preferred over 
transformation.

14. target  
Dist: Slight left-skewed (binary)  
Range: 0 – 1  
Peak: At 1  
Kurtosis: Extremely Platykurtic (−1.98)  
Insight: Majority have heart disease. Binary and flat; no transformation needed.
'''
# Set plot style
sns.set(style='whitegrid')

# 1. Age vs Cholesterol
sns.scatterplot(data=df, x='age', y='chol', hue='target', palette='coolwarm')
plt.title('Age vs Cholesterol (Hue: Target)')
plt.show()
#Slight positive correlation


# 2. Age vs Max Heart Rate
sns.scatterplot(data=df, x='age', y='thalach', hue='target', palette='viridis')
plt.title('Age vs Max Heart Rate (Hue: Target)')
plt.show()
#Moderate negative correlation

 
# 3. Resting BP vs Cholesterol
sns.scatterplot(data=df, x='trestbps', y='chol', hue='target', palette='magma')
plt.title('Resting BP vs Cholesterol (Hue: Target)')
plt.show()
#Weak positive correlation


# 4. ST Depression vs Max Heart Rate
sns.scatterplot(data=df, x='oldpeak', y='thalach', hue='target', palette='plasma')
plt.title('ST Depression vs Max Heart Rate (Hue: Target)')
plt.show()
#Moderate negative correlation


# 5. Cholesterol vs ST Depression
sns.scatterplot(data=df, x='chol', y='oldpeak', hue='target', palette='cubehelix')
plt.title('Cholesterol vs ST Depression (Hue: Target)')
plt.show()
#Weak correlation
'''
1. Age vs Cholesterol (Hue: Target)
Age <50:

Cholesterol mostly <250

Target hue shows mix of 0 and 1, but fewer heart disease cases

Age >60:

Cholesterol often >250

Target hue leans toward 1 (presence of heart disease)

Insight: Older individuals tend to have higher cholesterol and are more likely to 
have heart disease. Age and cholesterol show mild correlation, with disease risk 
increasing in older age groups.

2. Age vs Max Heart Rate (Hue: Target)
Age <50:

Max heart rate often >150

Target hue favors 0 (no heart disease)

Age >60:

Max heart rate drops below 140

Target hue shifts toward 1

Insight: Max heart rate declines with age, and lower heart rates are more common 
in heart disease patients. Suggests age-related cardiac efficiency may influence 
disease risk.

3. Resting BP vs Cholesterol (Hue: Target)
Resting BP <130:

Cholesterol varies widely

Target hue mixed, no strong pattern

Resting BP >140:

Cholesterol often >250

Target hue leans toward 1

Insight: Higher resting blood pressure may coincide with elevated cholesterol and 
increased heart disease risk. However, correlation is weak and may require 
multivariate analysis.

4. ST Depression vs Max Heart Rate (Hue: Target)
Oldpeak <1:

Max heart rate often >150

Target hue favors 0

Oldpeak >2:

Max heart rate drops below 130

Target hue shifts toward 1

Insight: Higher ST depression (oldpeak) is associated with lower heart rate and 
increased heart disease presence. Indicates possible ischemic response during 
exercise.

5. Cholesterol vs ST Depression (Hue: Target)
Cholesterol <200:

ST depression mostly <1

Target hue favors 0

Cholesterol >250:

ST depression varies

Target hue leans toward 1

Insight: Weak correlation overall, but high cholesterol may coincide with elevated 
ST depression in heart disease patients. Suggests cholesterol alone isn’t a strong 
predictor.
'''
 
