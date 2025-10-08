# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 17:34:12 2025

@author: anams
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

#load dataset
df = pd.read_csv('C:/5-python_crisp-ml(q)/clustering_datasets/AutoInsurance.csv')

#shape, size
print(df.shape) #(9134, 24)
print(df.size) #219216

df.describe() #of num cols only
'''
customer_lifetime_value : dist = right skewed dist. ,range = 81427.37 ;  
income                  : dist = right skewed dist. ,range = 99981.00 ;  
monthly_premium_auto    : dist = right skewed dist. ,range = 237.00 ;  
months_since_last_claim : dist = right skewed dist. ,range = 35.00 ;  
months_since_policy_inception : dist = slight right skewed dist. ,range = 99.00 ;  
number_of_open_complaints : dist = multimodal categorical dist. ,range = 5.00 ;  
number_of_policies      : dist = right skewed dist. ,range = 8.00 ;  
total_claim_amount      : dist = right skewed dist. ,range = 2893.14 ; 
'''

#1st moment : mean (central tendency)
mean_values = df.mean(numeric_only=True)
print('\nMean (1st moment):\n', mean_values)
'''
Mean (1st moment):
Customer Lifetime Value       8004.94  
Income                        37657.38  
Monthly Premium Auto            93.22  
Months Since Last Claim         15.10  
Months Since Policy Inception   48.06  
Number of Open Complaints        0.38  
Number of Policies               2.97  
Total Claim Amount             434.09

inference:
1. Average Customer Lifetime Value is around 8005, indicating moderately valuable long-term customers.

2. Mean income ≈ 37.6K suggests a mid-income customer base, relevant for affordability and pricing strategies.

3. Monthly Premium Auto ≈ 93 implies customers are paying moderate premiums, useful for tiered product design.

4. Months Since Last Claim ≈ 15 shows relatively low recent claim activity, hinting at a stable risk profile.

5. Policy Inception ≈ 48 months suggests long-standing customers, ideal for loyalty and retention programs.

6. Open Complaints ≈ 0.38 indicates low complaint frequency, suggesting decent service satisfaction.

7. Number of Policies ≈ 3 implies multi-policy holders, offering cross-sell opportunities.

8. Total Claim Amount ≈ 434 reflects moderate claim volume, useful for risk segmentation and pricing.
'''

var_values = df.var(numeric_only=True)
print('\nVariance (2nd moment): \n', var_values)
'''
Variance (2nd moment):  
Customer Lifetime Value       47210200.000000  
Income                        922938600.000000  
Monthly Premium Auto             1183.908000  
Months Since Last Claim           101.470500  
Months Since Policy Inception     778.744300  
Number of Open Complaints           0.828798  
Number of Policies                  5.712969  
Total Claim Amount              84390.300000

inference:
variance (how widely features are spread) must be high/medium  
  High Variance Features – Income, Customer Lifetime Value, Total Claim Amount  
  Moderate Variance Features – Monthly Premium Auto, Months Since Policy Inception  
  Low Variance Features – Number of Open Complaints, Number of Policies, Months Since Last Claim  

1. Very high variance in Income and Customer Lifetime Value suggests large differences in customer spending and value, requiring normalization or log transformation before modeling.

2. Moderate variance in Monthly Premium Auto and Policy Inception duration indicates a healthy spread, useful for tiered segmentation and retention analysis.

3. Low variance in Open Complaints and Number of Policies implies limited diversity, but they may still hold predictive power in churn or satisfaction models.

4. All features show measurable spread and are suitable for clustering or regression, though high-variance features should be scaled to avoid dominance in distance-based models.
'''
std_values = df.std(numeric_only=True)
print('\nSD (2nd moment): \n', std_values)
#higher SD low peakedness 
'''
SD (2nd moment):  
Customer Lifetime Value       6870.967608  
Income                        30379.904734  
Monthly Premium Auto            34.407967  
Months Since Last Claim         10.073257  
Months Since Policy Inception   27.905991  
Number of Open Complaints        0.910384  
Number of Policies               2.390182  
Total Claim Amount             290.500092

inference:
1. Higher SD → Lower Peakedness (Platykurtic):  
   Income, Customer Lifetime Value, Total Claim Amount  
   → Indicates wide spread and flatter distributions; normalization or log transformation may help.

2. Lower SD → Higher Peakedness (Leptokurtic):  
   Number of Open Complaints, Number of Policies  
   → Narrow spread with sharp peaks; may reflect limited variability or binary-like behavior.

3. Moderate SD → Mesokurtic:  
   Monthly Premium Auto, Months Since Last Claim, Months Since Policy Inception  
   → Balanced spread; distributions are close to normal and suitable for standard scaling. 
'''
#skewness(symmetry)
skew_values = df.skew(numeric_only=True)
print('\nSkewness (3rd moment): \n', skew_values)
#all features have low skewness (close to 0:between −0.5 and +0.5), 
#indicating near-normal dist
'''
Skewness (3rd moment):  
Customer Lifetime Value       3.032280   very strong rt skew  
Income                        0.286887   slight rt skew  
Monthly Premium Auto          2.123546   strong rt skew  
Months Since Last Claim       0.278586   slight rt skew  
Months Since Policy Inception 0.040165   near symmetric  
Number of Open Complaints     2.783263   very strong rt skew  
Number of Policies            1.253333   moderate rt skew  
Total Claim Amount            1.714966   strong rt skew
'''
#4th moment : kurtosis(peakedness)
kurt_values = df.kurtosis(numeric_only=True)
print('\nKurtosis (4th moment): \n', kurt_values)

'''
Kurtosis (4th moment):  
Customer Lifetime Value       13.823533   Leptokurtic = extremely peaked  
Income                        -1.094326   Platykurtic = very flat  
Monthly Premium Auto           6.193605   Leptokurtic = sharply peaked  
Months Since Last Claim       -1.073668   Platykurtic = very flat  
Months Since Policy Inception -1.133046   Platykurtic = very flat  
Number of Open Complaints      7.749308   Leptokurtic = sharply peaked  
Number of Policies             0.363157   Mesokurtic = close to normal  
Total Claim Amount             5.979401   Leptokurtic = sharply peaked
'''

