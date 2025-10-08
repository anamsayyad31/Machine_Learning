# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 20:13:12 2025

@author: anams
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

#load dataset
df = pd.read_excel('C:/5-python_crisp-ml(q)/understanding_data_assignments_datasets/EastWestAirlines.xlsx')

df.dtypes
#all cols are numerical(int)

#shape, size
print(df.shape) #rows : 3999 ,cols : 12
print(df.size) #rows x cols : 47988

df.describe() #of num cols only
'''
Balance: normal dist.; range = 4020
Qual_miles: strongly right skewed distribution; range = 1,704,838
cc1_miles: heavily right skewed distribution; range = 11,148
cc2_miles: slightly right skewed discrete distribution; range = 4
cc3_miles: slightly right skewed discrete distribution; range = 4
Bonus_miles: strongly right skewed distribution; range = 263,685
Bonus_trans: right skewed distribution; range = 83
Flight_miles_12mo: strongly right skewed distribution; range = 30,817
Flight_trans_12: strongly right skewed distribution; range = 53
Days_since_enroll: slightly left skewed distribution; range = 8,294
Award?: binary distribution (0 or 1); range = 1
'''

#1st moment : mean (central tendency)
mean_values = df.mean(numeric_only=True)
print('\nMean (1st moment):\n', mean_values)
'''
Mean (1st moment):
 ID#                   2014.819455
Balance              73601.327582
Qual_miles             144.114529
cc1_miles                2.059515
cc2_miles                1.014504
cc3_miles                1.012253
Bonus_miles          17144.846212
Bonus_trans             11.601900
Flight_miles_12mo      460.055764
Flight_trans_12          1.373593
Days_since_enroll     4118.559390
Award?                   0.370343
dtype: float64

inference:
1. Balance has the highest average (~73,601), indicating customers typically 
maintain large account balances—possibly linked to loyalty or premium tiers.

2. Bonus_miles (~17,145) and Days_since_enroll (~4,118) suggest long-term 
engagement and frequent promotional rewards. Customers appear to be retained over 
several years.

3. Flight_miles_12mo (~460) and Flight_trans_12 (~1.37) show that most customers 
fly infrequently. Travel activity is likely concentrated among a smaller segment.

4. Award? mean (~0.37) implies only 37% of customers receive awards, pointing to 
selective reward distribution or tiered loyalty structures.

5. cc1_miles, cc2_miles, and cc3_miles all average around 1–2, indicating minimal 
credit card mile accumulation—possibly due to limited usage or niche benefit 
programs.

6. Qual_miles (~144) reinforces that most customers aren’t frequent flyers, 
aligning with low flight activity metrics.
'''
var_values = df.var(numeric_only=True)
print('\nVariance (2nd moment): \n', var_values)
'''
Variance (2nd moment): 
 ID#                  1.347374e+06
Balance              1.015573e+10
Qual_miles           5.985557e+05
cc1_miles            1.895907e+00
cc2_miles            2.180060e-02
cc3_miles            3.811896e-02
Bonus_miles          5.832692e+08
Bonus_trans          9.223317e+01
Flight_miles_12mo    1.960586e+06
Flight_trans_12      1.438816e+01
Days_since_enroll    4.264781e+06
Award?               2.332473e-01
dtype: float64

inference:
variance(how widely features are spread) must be high/medium
  High Variance → Balance, Bonus_miles, Flight_miles_12mo  
  Moderate Variance → Days_since_enroll, Qual_miles, Bonus_trans, Flight_trans_12  
  Low Variance → cc1_miles, cc2_miles, cc3_miles, Award?, ID#

Inference:
1. All columns are usable for modeling.  
2. High variance features may need scaling.  
3. Balance is the most varied → diverse account sizes.  
4. Low variance in cc2_miles, cc3_miles → uniform usage.  
5. Moderate spread in tenure & engagement features.
'''
std_values = df.std(numeric_only=True)
print('\nSD (2nd moment): \n', std_values)
#higher SD low peakedness 
'''
SD (2nd moment): 
 ID#                    1160.764358
Balance              100775.664958
Qual_miles              773.663804
cc1_miles                 1.376919
cc2_miles                 0.147650
cc3_miles                 0.195241
Bonus_miles           24150.967826
Bonus_trans               9.603810
Flight_miles_12mo      1400.209171
Flight_trans_12           3.793172
Days_since_enroll      2065.134540
Award?                    0.482957
dtype: float64

inferences:
1. Higher SD → Lower Peakedness (Platykurtic) :  
Balance, Bonus_miles, Days_since_enroll

2. Lower SD → Higher Peakedness (Leptokurtic) :  
cc1_miles, cc2_miles, cc3_miles, Flight_trans_12, Award?

3. Moderate SD → Mesokurtic :  
ID#, Qual_miles, Bonus_trans, Flight_miles_12mo
'''
#skewness(symmetry)
skew_values = df.skew(numeric_only=True)
print('\nSkewness (3rd moment): \n', skew_values)
'''
Skewness (3rd moment):

ID#                  -0.003343   symmetric  
Balance               5.004187   right skew  
Qual_miles            7.512395   strong right skew  
cc1_miles             0.857569   right skew  
cc2_miles            11.210459   very strong right skew  
cc3_miles            17.195532   extreme right skew  
Bonus_miles           2.842093   right skew  
Bonus_trans           1.157362   right skew  
Flight_miles_12mo     7.451666   strong right skew  
Flight_trans_12       5.490461   right skew  
Days_since_enroll     0.120174   slight right skew  
Award?                0.537200   right skew
'''
#4th moment : kurtosis(peakedness)
kurt_values = df.kurtosis(numeric_only=True)
print('\nKurtosis (4th moment): \n', kurt_values)

'''
Kurtosis (4th moment):

ID#                  -1.199648   Platykurtic = flatter than normal  
Balance              44.157932   Leptokurtic = highly peaked  
Qual_miles           67.689351   Leptokurtic = very peaked  
cc1_miles            -0.748508   Platykurtic = slightly flat  
cc2_miles           133.786489   Leptokurtic = extremely peaked  
cc3_miles           308.654728   Leptokurtic = ultra peaked  
Bonus_miles          13.630489   Leptokurtic = peaked  
Bonus_trans           2.745737   Mesokurtic = close to normal  
Flight_miles_12mo    94.761019   Leptokurtic = very peaked  
Flight_trans_12      42.978152   Leptokurtic = highly peaked  
Days_since_enroll    -0.967505   Platykurtic = flatter than normal  
Award?               -1.712272   Platykurtic = very flat
'''
df.hist(figsize=(10,8), color='skyblue', edgecolor='black')    
plt.suptitle('hist of num features')
plt.tight_layout()
plt.show()

'''
1. Balance  
Dist: Normal distribution  
Range: ~4,020  
Peak: Around 60,000–80,000  
Kurtosis: Leptokurtic (44.15)  
Insight: Account balances are tightly clustered with a sharp peak. High kurtosis 
and variance suggest strong central tendency. May need scaling.

2. Qual_miles  
Dist: Strongly right-skewed  
Range: ~1,704,838  
Peak: Below 1,000  
Kurtosis: Very Leptokurtic (67.69)  
Insight: Most customers have very low qualification miles, with a few extreme 
outliers. Strong candidate for transformation.

3. cc1_miles  
Dist: Heavily right-skewed  
Range: ~11,148  
Peak: Around 0–2  
Kurtosis: Slightly Platykurtic (−0.75)  
Insight: Credit card 1 miles are minimal for most users. Distribution is flat with 
low spread. May not contribute much to modeling.

4. cc2_miles  
Dist: Slightly right-skewed (discrete)  
Range: 0–4  
Peak: At 0  
Kurtosis: Extremely Leptokurtic (133.79)  
Insight: Most customers don’t use credit card 2. Sharp peak and minimal spread 
suggest binary-like behavior.

5. cc3_miles  
Dist: Slightly right-skewed (discrete)  
Range: 0–4  
Peak: At 0  
Kurtosis: Ultra Leptokurtic (308.65)  
Insight: Similar to cc2_miles, usage is rare. Distribution is extremely peaked 
and may be treated as categorical.

6. Bonus_miles  
Dist: Strongly right-skewed  
Range: ~263,685  
Peak: Below 10,000  
Kurtosis: Leptokurtic (13.63)  
Insight: Promotional miles vary widely. High kurtosis and skew suggest outliers. 
Scaling and transformation recommended.

7. Bonus_trans  
Dist: Right-skewed  
Range: 0–83  
Peak: Around 5–15  
Kurtosis: Mesokurtic (2.74)  
Insight: Bonus transactions are moderately spread. Distribution is close to normal 
and suitable for modeling.

8. Flight_miles_12mo  
Dist: Strongly right-skewed  
Range: ~30,817  
Peak: Below 1,000  
Kurtosis: Very Leptokurtic (94.76)  
Insight: Most customers fly very little. High kurtosis and skew suggest a few 
frequent flyers dominate the distribution.

9. Flight_trans_12  
Dist: Strongly right-skewed  
Range: 0–53  
Peak: Around 1–3  
Kurtosis: Highly Leptokurtic (42.98)  
Insight: Flight frequency is low for most users. Sharp peak and long tail indicate 
rare travel behavior.

10. Days_since_enroll  
Dist: Slightly left-skewed  
Range: ~8,294  
Peak: Around 3,000–5,000  
Kurtosis: Platykurtic (−0.97)  
Insight: Enrollment duration is fairly balanced. Flat distribution suggests 
consistent customer retention.

11. Award?  
Dist: Binary  
Range: 0–1  
Peak: At 0  
Kurtosis: Very Platykurtic (−1.71)  
Insight: Only ~37% of customers receive awards. Flat binary distribution reflects 
selective reward structure.
'''
# Set plot style
sns.set(style='whitegrid')

# 1. Bonus_miles vs Balance (Hue: Award?)
sns.scatterplot(data=df, x='Bonus_miles', y='Balance', hue='Award?', palette='coolwarm')
plt.title('Bonus_miles vs Balance (Hue: Award?)')
plt.show()
# Moderate positive correlation

# 2. Flight_miles_12mo vs Bonus_miles (Hue: Award?)
sns.scatterplot(data=df, x='Flight_miles_12mo', y='Bonus_miles', hue='Award?', palette='viridis')
plt.title('Flight_miles_12mo vs Bonus_miles (Hue: Award?)')
plt.show()
# Strong positive correlation

# 3. Days_since_enroll vs Balance (Hue: Bonus_trans)
sns.scatterplot(data=df, x='Days_since_enroll', y='Balance', hue='Bonus_trans', palette='magma')
plt.title('Days_since_enroll vs Balance (Hue: Bonus_trans)')
plt.show()
# Weak positive correlation

# 4. Qual_miles vs Flight_miles_12mo (Hue: Award?)
sns.scatterplot(data=df, x='Qual_miles', y='Flight_miles_12mo', hue='Award?', palette='plasma')
plt.title('Qual_miles vs Flight_miles_12mo (Hue: Award?)')
plt.show()
# Strong positive correlation

# 5. Flight_trans_12 vs Bonus_trans (Hue: Award?)
sns.scatterplot(data=df, x='Flight_trans_12', y='Bonus_trans', hue='Award?', palette='cubehelix')
plt.title('Flight_trans_12 vs Bonus_trans (Hue: Award?)')
plt.show()
# Strong positive correlation
'''
1. Bonus_miles vs Balance (Hue: Award?)
Low Bonus_miles (<10,000):
Balance varies but mostly stays below 100,000.
Award hue is mostly 0 (no award).

High Bonus_miles (>30,000):
Balance often exceeds 150,000.
Award hue intensifies (more 1s).

Insight: Customers with more bonus miles tend to have higher balances. Award 
recipients are more common among high-mile earners. Moderate positive correlation.

2. Flight_miles_12mo vs Bonus_miles (Hue: Award?)
Flight_miles_12mo <5,000:
Bonus_miles mostly <20,000.
Award hue is mixed.

Flight_miles_12mo >10,000:
Bonus_miles often >50,000.
Award hue shifts toward 1.

Insight: Frequent flyers earn significantly more bonus miles. Strong positive 
correlation. Award status aligns with high travel activity.

3. Days_since_enroll vs Balance (Hue: Bonus_trans)
Enrollment <2,000 days:
Balance mostly <100,000.
Bonus_trans hue is light (few transactions).

Enrollment >5,000 days:
Balance varies widely, often >150,000.
Bonus_trans hue intensifies.

Insight: Longer-tenured customers tend to have higher balances and more bonus 
transactions. Weak positive correlation.

4. Qual_miles vs Flight_miles_12mo (Hue: Award?)
Qual_miles <1,000:
Flight_miles_12mo mostly <5,000.
Award hue is mixed.

Qual_miles >5,000:
Flight_miles_12mo often >10,000.
Award hue leans toward 1.

Insight: Qualification miles closely track actual flight miles. Strong positive 
correlation. Award status is more common among high-mile travelers.

5. Flight_trans_12 vs Bonus_trans (Hue: Award?)
Flight_trans_12 <5:
Bonus_trans mostly <10.
Award hue is mixed.

Flight_trans_12 >10:
Bonus_trans often >20.
Award hue intensifies.

Insight: More flight transactions lead to more bonus transactions. Strong positive 
correlation. Award recipients are more active travelers.
'''
