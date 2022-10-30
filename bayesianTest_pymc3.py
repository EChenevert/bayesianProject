# Follow a tutorial using tutorial from
# https://towardsdatascience.com/bayesian-linear-regression-in-python-using-machine-learning-to-predict-student-grades-part-2-b72059a8ac7e
import numpy as np
import pandas as pd

# Load
df = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\bayes2year\CRMS_dfi.csv", encoding="unicode escape")\
    .set_index('Unnamed: 0')
# df = pd.read_csv(r"D:\\Etienne\\fall2022\\CRMS_data\\biomassTimeSeries_CRMS.csv", encoding="unicode escape")\
#     .set_index('Simple site').drop(['index', 'Community', 'Unnamed: 0', 'level_1',
#                                     'Verified Pin Height (mm)', 'Shape_Area', 'Shape_Length',
#                                     'LengthScale (perimeter/area)', '90%thUpper_flooding (ft)',
#                                     '10%thLower_flooding (ft)', 'std_deviation_avg_flooding (ft)',
#                                     'Soil Porewater Specific Conductance (uS/cm)',
#                                     'Soil Porewater Temperature (Â°C)'], axis=1)
# df['Live/Dead Biomass'] = df['Belowground Live Biomass (g/m2)']/df['Belowground Dead Biomass (g/m2)']
# df = df.drop(['Belowground Live Biomass (g/m2)', 'Belowground Dead Biomass (g/m2)'], axis=1)
outcome = 'Accretion Rate (mm/yr)'
# Do some work on the design matrix: all assume identity for basis functions except log relationship for distance
df['log_distance_to_river_m'] = np.log(df['distance_to_river_m'])
df = df.drop(['distance_to_river_m'], axis=1)
# define target array (should be related to accretion)
t = np.asarray(df[outcome])
# df = df.drop(outcome, axis=1)

# Exploratory analysis
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure()
sns.histplot(t, stat='probability')
plt.title('Accretion Rate (mm/yr): Probability Plot')
plt.show()

# KDE for different marsh types
plt.figure()
sns.kdeplot(df[outcome], hue=df['Community'])
plt.title('KDE Density Plot of Accretion by Marsh Type')
plt.show()

# Investigate Accretion Percentiles
from scipy.stats import percentileofscore
df['Percentile'] = df[outcome].apply(lambda x: percentileofscore(df[outcome], x))
plt.figure()
plt.plot(df[outcome], df['Percentile'], 'o')
plt.xlabel('Accretion Rate (mm/yr)'); plt.ylabel('Percentile'); plt.title('Accretion Percentile (Cumulative)')
plt.show()
print('Min 50th Percentile Score: ', np.min(df.loc[df['Percentile'] > 50, outcome]))  # ~12 mm/yr
print('Min 90th Percentile Score: ', np.min(df.loc[df['Percentile'] > 90, outcome]))  # ~20 mm/yr



