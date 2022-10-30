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
from scipy.stats import percentileofscore, stats

df['Percentile'] = df[outcome].apply(lambda x: percentileofscore(df[outcome], x))
plt.figure()
plt.plot(df[outcome], df['Percentile'], 'o')
plt.xlabel('Accretion Rate (mm/yr)'); plt.ylabel('Percentile'); plt.title('Accretion Percentile (Cumulative)')
plt.show()
print('Min 50th Percentile Score: ', np.min(df.loc[df['Percentile'] > 50, outcome]))  # ~12 mm/yr
print('Min 90th Percentile Score: ', np.min(df.loc[df['Percentile'] > 90, outcome]))  # ~20 mm/yr

# Do the cool pairplot

# Implement the Bayesian Linear Regression
# Define Marsh Datasets
marshDic = {}
fresh = df[df['Community'] == 'Freshwater']
saline = df[df['Community'] == 'Saline']
intermediate = df[df['Community'] == 'Intermediate']
Brackish = df[df['Community'] == 'Brackish']

RSLR = df['RSLR (mm/yr)']
dfwhole = df.drop(['Community', 'RSLR (mm/yr)', 'Percentile'], axis=1)
df_vars = list(dfwhole.columns.values)

# rename variables
dfwhole = dfwhole.rename(columns={'Accretion Rate (mm/yr)': 'AccretionRate',
                                  'avg_flooding (ft)': 'flood_depth',
                                  '90%thUpper_flooding (ft)': 'Upperflood_depth',
                                  '10%thLower_flooding (ft)': 'Lowerflood_depth',
                                  'std_deviation_avg_flooding (ft)': 'std_flood_depth',
                                  'avg_percentflooded (%)': 'p_time_flooded',
                                  'Tide_Amp (ft)': 'Tide_Amp_ft'})


# Formula for Bayesian Linear Regression (follows R formula syntax
# PyMC3 for Bayesian Inference
import pymc3 as pm
formula = 'AccretionRate ~ ' + ' + '.join(dfwhole.columns[1:])
# Context for the model
with pm.Model() as normal_model:
    # The prior for the model parameters will be a normal distribution
    family = pm.glm.families.Normal()

    # Creating the model requires a formula and data (and optionally a family)
    pm.GLM.from_formula(formula, data=dfwhole, family=family)

    # Perform Markov Chain Monte Carlo sampling
    normal_trace = pm.sample(draws=2000, chains=2, tune=500)

# Investigate the posterior of the weight parameters (per feature)

# Investigate teh posterior of the weight parameters for each marsh type

# Use the MAP estimate of the weights to define weight values then predict



