# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 17:18:32 2025

@author: anams
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

#load dataset
df = pd.read_csv('C:/5-python_crisp-ml(q)/understanding_data_assignments_datasets/mtcars.csv')

df.dtypes
#mpg, disp, drat, wt, qsec column are float datatype
#all of the columns are numerical cols

#shape, size
print(df.shape) #rows : 32 ,cols : 11
print(df.size) #rows x cols : 352

df.describe() #of num cols only
'''
mpg : dist = Slight right skewed dist. ,range = 19.6  
cyl : dist = Discrete, left skewed dist. ,range = 4  
disp : dist = Right skewed dist. ,range = 254.9  
hp : dist = Right skewed dist. ,range = 128  
drat : dist = Slight left skewed dist. ,range = 1.16  
wt : dist = Slight right skewed dist. ,range = 2.10  
qsec : dist = Nearly symmetric dist. ,range = 4.4  
vs : dist = Binary dist. ,range = 1  
am : dist = Binary dist. ,range = 1  
gear : dist = Discrete, slightly right skewed dist. ,range = 1  
carb : dist = Discrete, right skewed dist. ,range = 3  
'''

#1st moment : mean (central tendency)
mean_values = df.mean(numeric_only=True)
print('\nMean (1st moment):\n', mean_values)
'''
Mean (1st moment):
 mpg      20.090625
cyl       6.187500
disp    230.721875
hp      146.687500
drat      3.596563
wt        3.217250
qsec     17.848750
vs        0.437500
am        0.406250
gear      3.687500
carb      2.812500
dtype: float64


inference:

1. disp and hp have the highest average values, indicating larger engine 
displacement and horsepower across the cars.

2. mpg (20.09) suggests moderate fuel efficiency overall, with some cars likely 
optimized for performance over economy.

3. cyl mean of 6.18 shows most cars have 6 cylinders, leaning toward mid-range 
engine configurations.

4. wt (3.22) and qsec (17.85) reflect average vehicle weight and acceleration 
time — suggesting balanced performance.

5. vs and am means below 0.5 imply most cars have V-shaped engines and automatic 
transmissions.

6. gear and carb averages (3.69 and 2.81) suggest most cars use 3–4 gears and 2–4 
carburetors, typical of 1970s designs.
'''


var_values = df.var(numeric_only=True)
print('\nVariance (2nd moment): \n', var_values)
'''
Variance (2nd moment): 
 mpg        36.324103
cyl         3.189516
disp    15360.799829
hp       4700.866935
drat        0.285881
wt          0.957379
qsec        3.193166
vs          0.254032
am          0.248992
gear        0.544355
carb        2.608871
dtype: float64

inference:

variance (how widely features are spread) must be high/medium

High Variance Features – disp, hp  
Moderate Variance Features – mpg, cyl, qsec, carb  
Low Variance Features – drat, wt, vs, am, gear  

1. disp and hp show very high variance, indicating wide spread in engine size 
and horsepower across cars — useful for performance modeling.

2. mpg and qsec have moderate variance, suggesting diversity in fuel efficiency 
and acceleration.

3. vs and am have low variance, reflecting limited variation in engine shape and 
transmission type — mostly binary.

4. All features are usable for modeling, but high variance features like disp and 
hp may require scaling.

5. gear and carb show moderate spread, indicating some diversity in drivetrain 
and fuel systems.
'''
std_values = df.std(numeric_only=True)
print('\nSD (2nd moment): \n', std_values)
#higher SD means low peakedness 
'''
SD (2nd moment): 
 mpg       6.026948
cyl       1.785922
disp    123.938694
hp       68.562868
drat      0.534679
wt        0.978457
qsec      1.786943
vs        0.504016
am        0.498991
gear      0.737804
carb      1.615200
dtype: float64

inference:

1. Higher SD → Lower Peakedness (Platykurtic):  
   disp, hp — wide spread in engine specs, flatter distributions  

2. Lower SD → Higher Peakedness (Leptokurtic):  
   drat, vs, am — tightly clustered values, sharper peaks  

3. Moderate SD → Mesokurtic:  
   mpg, cyl, qsec, wt, carb — balanced spread, typical bell-shaped curves  

4. disp and hp show the highest SD, indicating large variability in engine 
displacement and horsepower — likely influenced by car type.

5. vs and am have very low SD, confirming binary nature and minimal spread.

6. Most features fall in the moderate SD range, making them suitable for modeling 
without extreme outlier concerns.
'''
#skewness(symmetry)
skew_values = df.skew(numeric_only=True)
print('\nSkewness (3rd moment): \n', skew_values)
'''
Skewness (3rd moment):  
mpg     0.672377   → right skew  
cyl    -0.192261   → slight left skew  
disp    0.420233   → slight right skew  
hp      0.799407   → right skew  
drat    0.292780   → slight right skew  
wt      0.465916   → slight right skew  
qsec    0.406347   → slight right skew  
vs      0.264542   → slight right skew  
am      0.400809   → slight right skew  
gear    0.582309   → right skew  
carb    1.157091   → strong right skew  
'''
#4th moment : kurtosis(peakedness)
kurt_values = df.kurtosis(numeric_only=True)
print('\nKurtosis (4th moment): \n', kurt_values)

'''
mpg    -0.022    Mesokurtic = close to normal  
cyl    -1.763    Platykurtic = very flat  
disp   -1.068    Platykurtic = very flat  
hp      0.275    Mesokurtic = close to normal  
drat   -0.450    Platykurtic = slightly flat  
wt      0.417    Mesokurtic = moderately peaked  
qsec    0.865    Leptokurtic = more peaked than normal  
vs     -2.063    Platykurtic = extremely flat  
am     -1.967    Platykurtic = extremely flat  
gear   -0.895    Platykurtic = very flat  
carb    2.020    Leptokurtic = highly peaked  
'''
df.hist(figsize=(10,8), color='skyblue', edgecolor='black')    
plt.suptitle('hist of num features')
plt.tight_layout()
plt.show()

'''
1. mpg  
Dist: Slightly right-skewed  
Range: 10.4 – 30.0  
Peak: Around 15–22  
Kurtosis: Mesokurtic (−0.022)  
Insight: Fuel efficiency varies moderately across cars. Distribution is close to 
normal with a slight lean toward lower mpg. Suitable for modeling without 
transformation.

2. cyl  
Dist: Slightly left-skewed  
Range: 4 – 8  
Peak: Around 4–6  
Kurtosis: Platykurtic (−1.763)  
Insight: Cylinder count clusters around 4 and 6. Flat distribution suggests 
limited diversity. May benefit from encoding if used in classification.

3. disp  
Dist: Right-skewed  
Range: 71.1 – 326.0  
Peak: Around 100–150  
Kurtosis: Platykurtic (−1.068)  
Insight: Engine displacement shows wide variation. Flat distribution suggests 
diverse engine sizes. Strong candidate for scaling and transformation.

4. hp  
Dist: Right-skewed  
Range: 52 – 180  
Peak: Around 100–150  
Kurtosis: Mesokurtic (0.275)  
Insight: Horsepower is moderately spread with a slight skew. Distribution is 
close to normal, making it model-friendly with minimal preprocessing.

5. drat  
Dist: Slightly right-skewed  
Range: 2.76 – 3.92  
Peak: Around 3.0–3.8  
Kurtosis: Platykurtic (−0.450)  
Insight: Rear axle ratio is fairly consistent. Slight skew and flatness suggest 
mild variation. Suitable for modeling without transformation.

6. wt  
Dist: Slightly right-skewed  
Range: 1.51 – 3.62  
Peak: Around 2.5–3.5  
Kurtosis: Mesokurtic (0.417)  
Insight: Vehicle weight is fairly balanced. Distribution is moderately peaked and 
suitable for modeling without transformation.

7. qsec  
Dist: Slightly right-skewed  
Range: 14.5 – 18.9  
Peak: Around 17–18  
Kurtosis: Leptokurtic (0.865)  
Insight: Acceleration time is tightly clustered with a sharp peak. Indicates 
consistent performance across cars. No transformation needed.

8. vs  
Dist: Slightly right-skewed (Binary)  
Range: 0 – 1  
Peak: At 0  
Kurtosis: Platykurtic (−2.063)  
Insight: Most cars have V-shaped engines. Extremely flat distribution due to 
binary nature. Suitable for classification tasks.

9. am  
Dist: Slightly right-skewed (Binary)  
Range: 0 – 1  
Peak: At 0  
Kurtosis: Platykurtic (−1.967)  
Insight: Majority of cars have automatic transmission. Flat distribution reflects 
binary encoding. Useful for categorical modeling.

10. gear  
Dist: Right-skewed  
Range: 3 – 5  
Peak: At 3 and 4  
Kurtosis: Platykurtic (−0.895)  
Insight: Gear count is concentrated around 3 and 4. Flat distribution suggests 
limited spread. May benefit from categorical encoding.

11. carb  
Dist: Strong right-skewed  
Range: 1 – 8  
Peak: Around 2–4  
Kurtosis: Leptokurtic (2.020)  
Insight: Carburetor count varies widely with a sharp peak. High kurtosis 
indicates outliers. Consider transformation or binning.
'''
# Set plot style
sns.set(style='whitegrid')

sns.scatterplot(data=df, x='wt', y='mpg', hue='hp', palette='coolwarm')
plt.title('Weight vs MPG (Hue: Horsepower)')
plt.show()
#Strong negative correlation


sns.scatterplot(data=df, x='disp', y='hp', hue='mpg', palette='viridis')
plt.title('Displacement vs Horsepower (Hue: MPG)')
plt.show()
#Moderate positive correlation

 
sns.scatterplot(data=df, x='qsec', y='wt', hue='am', palette='magma')
plt.title('Quarter Mile Time vs Weight (Hue: Transmission)')
plt.show()
#Weak negative correlation


sns.scatterplot(data=df, x='drat', y='mpg', hue='gear', palette='plasma')
plt.title('Rear Axle Ratio vs MPG (Hue: Gear)')
plt.show()
#Slight positive correlation


sns.scatterplot(data=df, x='hp', y='mpg', hue='cyl', palette='cubehelix')
plt.title('Horsepower vs MPG (Hue: Cylinders)')
plt.show()
#Strong negative correlation

'''
1. wt vs mpg (Hue: hp)
Low wt (<3.0):

mpg values are high (25–35).

hp hue is lighter, indicating lower horsepower.

High wt (>4.0):

mpg drops below 15.

hp hue intensifies, showing higher horsepower.

Insight: Heavier cars tend to have lower fuel efficiency and higher horsepower. 
Suggests a trade-off between performance and economy.

2. disp vs hp (Hue: mpg)
Low disp (<200):

hp ranges from 50–150.

mpg hue is brighter, indicating better fuel economy.

High disp (>300):

hp exceeds 200.

mpg hue darkens, showing poor mileage.

Insight: Larger engine displacement correlates with higher horsepower but reduced 
mpg. Indicates performance-focused builds sacrifice efficiency.

3. qsec vs wt (Hue: am)
qsec >18:

wt is moderate (2.5–3.5).

am hue varies, showing mix of manual and automatic.

qsec <17:

wt increases (>3.5).

am hue leans toward automatic.

Insight: Heavier cars tend to have quicker quarter-mile times and are often 
automatic. Transmission type may influence acceleration dynamics.

4. drat vs mpg (Hue: gear)
drat <3.5:

mpg mostly <20.

gear hue varies, no strong pattern.

drat >4.0:

mpg exceeds 25.

gear hue shifts toward higher gears (4 or 5).

Insight: Higher rear axle ratios are associated with better fuel efficiency and 
more gears. Suggests tuning for highway performance.

5. hp vs mpg (Hue: cyl)
hp <100:

mpg is high (>25).

cyl hue is light, indicating fewer cylinders (4).

hp >150:

mpg drops below 15.

cyl hue darkens, showing 6 or 8 cylinders.

Insight: Cars with more cylinders and horsepower consume more fuel. Strong 
inverse relationship between power and efficiency.
'''
sns.countplot(data=df, x='am', palette='pastel')
plt.title('Transmission Type Distribution')
plt.xticks([0, 1], ['Automatic', 'Manual'])
plt.show()
#inference : manual cars are 2/3rd of automatic cars

sns.countplot(data=df, x='vs', palette='coolwarm')
plt.title('Engine Shape Distribution')
plt.xticks([0, 1], ['V-shaped', 'Straight'])
plt.show()
#inference : straight are 3/4th of v-shaped
