# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 17:28:35 2025

@author: anams
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

#load dataset
df = pd.read_excel('C:/5-python_crisp-ml(q)/clustering_datasets/Telco_customer_churn.xlsx')

#shape, size
print(df.shape) #(7043, 30)
print(df.size) #211290

df.describe() #of num cols only
'''
Number of Referrals       : dist = right skewed dist. ,range = 11.0 ;  
Tenure in Months          : dist = slight right skewed dist. ,range = 71.0 ;  
Avg Monthly Long Distance Charges : dist = slight right skewed dist. ,range = 49.99 ;  
Avg Monthly GB Download   : dist = slight right skewed dist. ,range = 85.0 ;  
Monthly Charge            : dist = slight right skewed dist. ,range = 100.5 ;  
Total Charges             : dist = slight right skewed dist. ,range = 8666.0 ;  
Total Refunds             : dist = highly right skewed dist. ,range = 49.79 ;  
Total Extra Data Charges  : dist = highly right skewed dist. ,range = 150.0 ;  
Total Long Distance Charges : dist = slight right skewed dist. ,range = 3564.72 ;  
Total Revenue             : dist = slight right skewed dist. ,range = 11957.98 ; 
'''

#1st moment : mean (central tendency)
mean_values = df.mean(numeric_only=True)
print('\nMean (1st moment):\n', mean_values)
'''
Mean (1st moment):
Number of Referrals               1.951867
Tenure in Months                32.386767
Avg Monthly Long Distance Charges 22.958954
Avg Monthly GB Download         20.515405
Monthly Charge                  64.761692
Total Charges                 2280.381264
Total Refunds                    1.962182
Total Extra Data Charges         6.860713
Total Long Distance Charges    749.099262
Total Revenue                 3034.379056
dtype: float64

inference:
1. Average tenure is around 32 months, indicating customers typically stay for nearly 3 years.

2. Mean number of referrals ≈ 1.95 suggests most customers refer at least one person, showing moderate referral engagement.

3. Average monthly long distance charges ≈ ₹23 and GB download ≈ 20.5 indicate moderate usage of telecom services.

4. Monthly charge averages around ₹65, pointing to mid-tier pricing plans being common.

5. Total charges ≈ ₹2280 and total revenue ≈ ₹3034 suggest that customers contribute significantly over time, with revenue exceeding direct charges due to add-ons.

6. Mean total refunds ≈ ₹1.96 and extra data charges ≈ ₹6.86 imply occasional refunds and moderate overage usage.

7. Total long distance charges ≈ ₹749 show that long-distance calling remains a notable revenue stream.

8. Overall, the dataset reflects a mix of mid-duration, mid-spend customers with moderate service usage and occasional add-on charges.
'''


var_values = df.var(numeric_only=True)
print('\nVariance (2nd moment): \n', var_values)
'''
Variance (2nd moment): 
Number of Referrals               9.007197
Tenure in Months               602.312800
Avg Monthly Long Distance Charges 238.644200
Avg Monthly GB Download         416.933100
Monthly Charge                  905.410900
Total Charges                5135755.000000
Total Refunds                    62.451310
Total Extra Data Charges        630.259900
Total Long Distance Charges   716833.200000
Total Revenue                8209397.000000
dtype: float64

inference:
variance (how widely features are spread) must be high/medium  
  High Variance Features - Total Revenue, Total Charges, Total Long Distance Charges  
  Moderate Variance Features - Monthly Charge, Avg Monthly GB Download, Tenure in Months  
  Low Variance Features - Number of Referrals, Total Refunds, Avg Monthly Long Distance Charges, Total Extra Data Charges  

1. High variance in Total Revenue and Total Charges indicates large differences in customer value and spending behavior — crucial for revenue segmentation.

2. Moderate variance in Monthly Charge and Tenure suggests a balanced spread of pricing plans and customer retention durations — useful for stratified retention modeling.

3. Low variance in features like Number of Referrals and Total Refunds implies limited diversity in referral behavior and refund activity — may need transformation or binning.

4. All features show measurable spread and are viable for clustering or predictive modeling, but high-variance features may require scaling to avoid dominance in distance-based algorithms.
'''
std_values = df.std(numeric_only=True)
print('\nSD (2nd moment): \n', std_values)
#higher SD low peakedness 
'''
SD (2nd moment): 
Number of Referrals               3.001199
Tenure in Months                24.542061
Avg Monthly Long Distance Charges 15.448113
Avg Monthly GB Download         20.418940
Monthly Charge                  30.090047
Total Charges                 2266.220462
Total Refunds                    7.902614
Total Extra Data Charges        25.104978
Total Long Distance Charges    846.660055
Total Revenue                 2865.204542
dtype: float64

inference:
1. Higher SD → Lower Peakedness (Platykurtic):  
   Total Revenue, Total Charges, Total Long Distance Charges  

2. Lower SD → Higher Peakedness (Leptokurtic):  
   Number of Referrals, Total Refunds  

3. Moderate SD → Mesokurtic:  
   Tenure in Months, Avg Monthly GB Download, Monthly Charge, Total Extra Data Charges, Avg Monthly Long Distance Charges 
'''
#skewness(symmetry)
skew_values = df.skew(numeric_only=True)
print('\nSkewness (3rd moment): \n', skew_values)
#all features have low skewness (close to 0:between −0.5 and +0.5), 
#indicating near-normal dist
'''
Skewness (3rd moment):  
Count                          0.000000   symmetric  
Number of Referrals            1.446060   strong rt skew  
Tenure in Months               0.240543   slight rt skew  
Avg Monthly Long Distance Charges  0.049176   near symmetric  
Avg Monthly GB Download        1.216584   strong rt skew  
Monthly Charge                -0.220524   slight lt skew  
Total Charges                  0.963791   moderate rt skew  
Total Refunds                  4.328517   very strong rt skew  
Total Extra Data Charges       4.091209   very strong rt skew  
Total Long Distance Charges    1.238282   strong rt skew  
Total Revenue                  0.919410   moderate rt skew
'''
#4th moment : kurtosis(peakedness)
kurt_values = df.kurtosis(numeric_only=True)
print('\nKurtosis (4th moment): \n', kurt_values)

'''
Kurtosis (4th moment):  
Count                          0.000000   Mesokurtic = close to normal  
Number of Referrals            0.721964   Mesokurtic = close to normal  
Tenure in Months              -1.387052   Platykurtic = very flat  
Avg Monthly Long Distance Charges  -1.254654   Platykurtic = very flat  
Avg Monthly GB Download        0.881502   Mesokurtic = close to normal  
Monthly Charge                -1.257260   Platykurtic = very flat  
Total Charges                 -0.227693   Platykurtic = slightly flat  
Total Refunds                 18.350658   Leptokurtic = extremely peaked  
Total Extra Data Charges      16.458874   Leptokurtic = extremely peaked  
Total Long Distance Charges    0.644092   Mesokurtic = close to normal  
Total Revenue                 -0.203457   Platykurtic = slightly flat 
'''
