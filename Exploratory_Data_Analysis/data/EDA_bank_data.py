# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 17:59:23 2025

@author: anams
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

#load dataset
df = pd.read_csv('C:/5-python_crisp-ml(q)/understanding_data_assignments_datasets/bank_data.csv')

df.dtypes
#there are no categorical columns so no count plot,pie & bar chart

#shape, size
print(df.shape) #(45211, 32) : rows,cols
print(df.size) #rows * cols : 1446752 

df.describe()
'''
Age : dist = normal dist. ,range = 77 ;
Default : dist = heavily left skewed dist. ,range = 1 ;
Balance : dist = right skewed dist. ,range = 110146 ;
Housing : dist = bimodal dist. ,range = 1 ;
Loan : dist = left skewed dist. ,range = 1 ;
Duration : dist = right skewed dist. ,range = 4918 ;
Campaign : dist = right skewed dist. ,range = 62 ;
Pdays : dist = right skewed dist. ,range = 872 ;
Previous : dist = right skewed dist. ,range = 275 ;
Poutfailure : dist = left skewed dist. ,range = 1 ;
Poutother : dist = left skewed dist. ,range = 1 ;
Poutsuccess : dist = left skewed dist. ,range = 1 ;
Poutunknown : dist = left skewed dist. ,range = 1 ;
Con_cellular : dist = left skewed dist. ,range = 1 ;
Con_telephone : dist = left skewed dist. ,range = 1 ;
Con_unknown : dist = left skewed dist. ,range = 1 ;
Divorced : dist = left skewed dist. ,range = 1 ;
Married : dist = bimodal dist. ,range = 1 ;
Single : dist = left skewed dist. ,range = 1 ;
Joadmin. : dist = left skewed dist. ,range = 1 ;
Joblue.collar : dist = left skewed dist. ,range = 1 ;
Joentrepreneur : dist = left skewed dist. ,range = 1 ;
Johousemaid : dist = left skewed dist. ,range = 1 ;
Jomanagement : dist = left skewed dist. ,range = 1 ;
Joretired : dist = left skewed dist. ,range = 1 ;
Joself.employed : dist = left skewed dist. ,range = 1 ;
Joservices : dist = left skewed dist. ,range = 1 ;
Jostudent : dist = left skewed dist. ,range = 1 ;
Jotechnician : dist = left skewed dist. ,range = 1 ;
Jounemployed : dist = left skewed dist. ,range = 1 ;
Jounknown : dist = left skewed dist. ,range = 1 ;
Y : dist = left skewed dist. ,range = 1 ;

except age,balance,duration,campaign,pdays,previous all cols are binary. 
'''
#1st moment : mean (central tendency)
mean_values = df.mean(numeric_only=True)
print('\nMean (1st moment):\n', mean_values)
'''
Mean (1st moment):
 age                  40.936210
default               0.018027
balance            1362.272058
housing               0.555838
loan                  0.160226
duration            258.163080
campaign              2.763841
pdays                40.197828
previous              0.580323
poutfailure           0.108403
poutother             0.040698
poutsuccess           0.033421
poutunknown           0.817478
con_cellular          0.647741
con_telephone         0.064276
con_unknown           0.287983
divorced              0.115171
married               0.601933
single                0.282896
joadmin.              0.114375
joblue.collar         0.215257
joentrepreneur        0.032890
johousemaid           0.027427
jomanagement          0.209197
joretired             0.050076
joself.employed       0.034925
joservices            0.091880
jostudent             0.020747
jotechnician          0.168034
jounemployed          0.028820
jounknown             0.006370
y                     0.116985
dtype: float64

Inference:

1. Age has a moderate average of ~41 years, suggesting a middle-aged population 
dominates the dataset.

2. Balance shows a high mean (~1362), but likely skewed due to outliers, 
indicating financial disparity.

3. Duration of contact (~258 seconds) suggests moderately long interactions, 
possibly reflecting engagement level.

4. Housing loans (~56%) and marital status (~60% married) show strong representation, 
hinting at financial and social stability.

5. Poutunknown has the highest mean among outcome types (~0.82), indicating most 
clients had unknown previous outcomes.

6. Cellular contact (~65%) is the dominant communication method, while telephone 
(~6%) is least used.

7. Job categories like blue-collar (~21%) and management (~21%) are more common, 
while student and unknown jobs are rare.

8. The target variable `y` has a low mean (~0.12), suggesting a small proportion 
of positive responses.
'''

var_values = df.var(numeric_only=True)
print('\nVariance (2nd moment): \n', var_values)
'''
Variance (2nd moment): 
 age                1.127581e+02
default            1.770202e-02
balance            9.270599e+06
housing            2.468876e-01
loan               1.345569e-01
duration           6.632057e+04
campaign           9.597733e+00
pdays              1.002577e+04
previous           5.305841e+00
poutfailure        9.665379e-02
poutother          3.904259e-02
poutsuccess        3.230482e-02
poutunknown        1.492110e-01
con_cellular       2.281778e-01
con_telephone      6.014627e-02
con_unknown        2.050533e-01
divorced           1.019090e-01
married            2.396149e-01
single             2.028702e-01
joadmin.           1.012955e-01
joblue.collar      1.689254e-01
joentrepreneur     3.180916e-02
johousemaid        2.667531e-02
jomanagement       1.654372e-01
joretired          4.756972e-02
joself.employed    3.370611e-02
joservices         8.344015e-02
jostudent          2.031717e-02
jotechnician       1.398019e-01
jounemployed       2.799042e-02
jounknown          6.329693e-03
y                  1.033016e-01
dtype: float64

inference:
variance(how widely features are spread) must be high/medium
  High Variance Features - Balance, Duration, Pdays 
  Moderate Variance Features - Age, Campaign, Previous
  Low Variance Features - rest all cols
  
1. Balance has the highest variance (~9.27 million), indicating extreme financial 
variability among individuals.

2. Duration and Pdays also show substantial spread, suggesting diverse engagement 
and historical contact patterns.

3. Features like jounknown and jostudent have very low variance, possibly 
contributing less to predictive power.

4. All features are usable for modeling, but high-variance ones may benefit from 
normalization to improve convergence.
'''
std_values = df.std(numeric_only=True)
print('\nSD (2nd moment): \n', std_values)
#higher SD means low peakedness 
'''
SD (2nd moment): 
 age                  10.618762
default               0.133049
balance            3044.765829
housing               0.496878
loan                  0.366820
duration            257.527812
campaign              3.098021
pdays               100.128746
previous              2.303441
poutfailure           0.310892
poutother             0.197592
poutsuccess           0.179735
poutunknown           0.386278
con_cellular          0.477680
con_telephone         0.245247
con_unknown           0.452828
divorced              0.319232
married               0.489505
single                0.450411
joadmin.              0.318269
joblue.collar         0.411005
joentrepreneur        0.178351
johousemaid           0.163326
jomanagement          0.406740
joretired             0.218105
joself.employed       0.183592
joservices            0.288860
jostudent             0.142538
jotechnician          0.373901
jounemployed          0.167303
jounknown             0.079559
y                     0.321406
dtype: float64

inferences:

1. Higher SD → Lower Peakedness (Platykurtic) :
Balance, Duration, Pdays

2. Moderate SD → Mesokurtic :
Age, Campaign, Previous

3. Lower SD → Higher Peakedness (Leptokurtic) :
rest all cols

'''
#skewness(symmetry)
skew_values = df.skew(numeric_only=True)
print('\nSkewness (3rd moment): \n', skew_values)
'''
Skewness (3rd moment): 
Age               0.684818   rt skew
Default           7.245375   highly rt skew
Balance           8.360308   highly rt skew
Housing          -0.224766   slight lt skew
Loan              1.852617   rt skew
Duration          3.144318   rt skew
Campaign          4.898650   rt skew
Pdays             2.615715   rt skew
Previous         41.846454   extremely rt skew
Poutfailure       2.519297   rt skew
Poutother         4.649199   rt skew
Poutsuccess       5.192072   rt skew
Poutunknown      -1.643851   lt skew
Con_cellular     -0.618604   lt skew
Con_telephone     3.553497   rt skew
Con_unknown       0.936454   rt skew
Divorced          2.411075   rt skew
Married          -0.416493   slight lt skew
Single            0.964070   rt skew
Joadmin.          2.423369   rt skew
Joblue.collar     1.385652   rt skew
Joentrepreneur    5.238320   rt skew
Johousemaid       5.787133   rt skew
Jomanagement      1.429986   rt skew
Joretired         4.125939   rt skew
Joself.employed   5.066613   rt skew
Joservices        2.825851   rt skew
Jostudent         6.724846   rt skew
Jotechnician      1.775767   rt skew
Jounemployed      5.632886   rt skew
Jounknown        12.409644   extremely rt skew
Y                 2.383480   rt skew
'''
#4th moment : kurtosis(peakedness)
kurt_values = df.kurtosis(numeric_only=True)
print('\nKurtosis (4th moment): \n', kurt_values)

'''
Kurtosis (4th moment): 
Age                 0.319570   Mesokurtic = close to normal
Default            50.497694   Leptokurtic = sharply peaked
Balance           140.751547   Leptokurtic = extremely peaked
Housing            -1.949566   Platykurtic = very flat
Loan                1.432253   Leptokurtic = moderately peaked
Duration           18.153915   Leptokurtic = sharply peaked
Campaign           39.249651   Leptokurtic = sharply peaked
Pdays               6.935195   Leptokurtic = peaked
Previous         4506.860660   Leptokurtic = extremely peaked
Poutfailure         4.347048   Leptokurtic = peaked
Poutother          19.615922   Leptokurtic = sharply peaked
Poutsuccess        24.958714   Leptokurtic = sharply peaked
Poutunknown         0.702278   Mesokurtic = close to normal
Con_cellular       -1.617401   Platykurtic = very flat
Con_telephone      10.627811   Leptokurtic = sharply peaked
Con_unknown        -1.123104   Platykurtic = flat
Divorced            3.813451   Leptokurtic = peaked
Married            -1.826614   Platykurtic = very flat
Single             -1.070617   Platykurtic = flat
Joadmin.            3.872890   Leptokurtic = peaked
Joblue.collar      -0.079971   Mesokurtic = close to normal
Joentrepreneur     25.441124   Leptokurtic = sharply peaked
Johousemaid        31.492300   Leptokurtic = sharply peaked
Jomanagement        0.044861   Mesokurtic = close to normal
Joretired          15.024033   Leptokurtic = sharply peaked
Joself.employed    23.671618   Leptokurtic = sharply peaked
Joservices          5.985698   Leptokurtic = peaked
Jostudent          43.225460   Leptokurtic = extremely peaked
Jotechnician        1.153398   Leptokurtic = moderately peaked
Jounemployed       29.730717   Leptokurtic = sharply peaked
Jounknown         152.005993   Leptokurtic = extremely peaked
Y                   3.681142   Leptokurtic = peaked	
'''
df.hist(figsize=(10,8), color='skyblue', edgecolor='black')    
plt.suptitle('hist of num features')
plt.tight_layout()
plt.show()

'''
(only continous cols included)
1. Age
Dist: Slightly right-skewed
Range: 18 – 95
Peak: Around 30–40
Kurtosis: Mesokurtic (0.32)
Insight: Age is moderately spread, with a natural concentration around middle-aged 
individuals. Slight skew and near-normal peakedness make it suitable for modeling 
without transformation.

2. Balance
Dist: Strongly right-skewed
Range: −8019 – 110146
Peak: Around 0–500
Kurtosis: Extremely Leptokurtic (140.75)
Insight: High concentration near lower values but extreme outliers present. Heavy 
tails suggest a need for log or robust transformation to reduce impact of outliers 
and improve model stability.

3. Duration
Dist: Right-skewed
Range: 0 – 4918
Peak: Around 100–200 seconds
Kurtosis: Strongly Leptokurtic (18.15)
Insight: Majority of calls are short with few long interactions. High skew and 
peakedness may affect model performance — consider log scaling or capping long 
durations.

4. Campaign
Dist: Strongly right-skewed
Range: 1 – 62
Peak: Around 1–2
Kurtosis: Very Leptokurtic (39.25)
Insight: Most clients were contacted very few times; a small number received 
frequent contact. Outliers may affect models — binning or transformation might 
help.

5. Pdays
Dist: Right-skewed
Range: −1 – 871
Peak: At −1 (not previously contacted)
Kurtosis: Leptokurtic (6.93)
Insight: Dominated by −1 values indicating no prior contact, with some high values. 
Consider treating −1 as a separate category or encoding it differently before 
modeling.

6. Previous
Dist: Extremely right-skewed
Range: 0 – 275
Peak: 0
Kurtosis: Extremely Leptokurtic (4506.86)
Insight: Most people were never previously contacted. Extreme kurtosis due to few 
repeated contacts. May be better treated as categorical or log-transformed for 
regression-based models.
'''
# Set plot style
sns.set(style='whitegrid')

# 1. Duration vs y (Hue: y)
sns.scatterplot(data=df, x='duration', y='y', hue='y', palette='coolwarm')
plt.title('Duration vs Response (Hue: y)')
plt.show()
# Insight: Longer durations are clearly associated with positive responses.

# 2. Duration vs Pdays (Hue: y)
sns.scatterplot(data=df, x='duration', y='pdays', hue='y', palette='plasma')
plt.title('Duration vs Pdays (Hue: Response - y)')
plt.show()
# Insight: Clients who were previously contacted and had long durations tend 
#to respond positively.

# 3. Campaign vs Duration (Hue: y)
sns.scatterplot(data=df, x='campaign', y='duration', hue='y', palette='viridis')
plt.title('Campaign vs Duration (Hue: Response - y)')
plt.show()
# Insight: Fewer campaigns + longer calls = better success rate.

# 4. Balance vs Duration (Hue: Housing)
sns.scatterplot(data=df, x='balance', y='duration', hue='housing', palette='Set1')
plt.title('Balance vs Duration (Hue: Housing Loan)')
plt.show()
# Insight: Clients without housing loans show more variation in call duration and 
#balance.

# 5. Previous vs Pdays (Hue: poutsuccess)
sns.scatterplot(data=df, x='previous', y='pdays', hue='poutsuccess', palette='Set2')
plt.title('Previous vs Pdays (Hue: Previous Outcome: Success)')
plt.show()

