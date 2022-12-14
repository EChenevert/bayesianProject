import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import bayes_funcs as bml


# 1.) Aquire dataset I want to work with
# ------------> may have to be done after exploratory analysis / feature selection process
# 2.) Normalize dataset between -1 and 1
# 3.) Use iterative program to find hyperparameters
# ------------> remember that this is done by

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
df = df.drop(['distance_to_river_m', 'Community'], axis=1)
# define target array (should be related to accretion)
t = np.asarray(df[outcome])
df = df.drop(outcome, axis=1)

df_vars = list(df.columns.values)

phi = np.asarray(df)
# # Normalize dataset between 0 and 1
# x_scaler = MinMaxScaler()
# phi = x_scaler.fit_transform(df)

RSLR = phi[:, 0]
df_vars.remove('RSLR (mm/yr)')
phi = phi[:, 1:]

# Shuffle the dataset to avoid training in only certain spatial areas
np.random.seed(42)
np.random.shuffle(phi)

import seaborn as sns
import matplotlib.pyplot as plt

# sns.pairplot(pd.DataFrame(phi, columns=df.drop(outcome, axis=1).columns.values))
# plt.show()
# Seems to be kiiiinnnddaaa normal, barely w some variables

# Find the weights with the obtainesd hyperparameters --> thru the lambda function
from sklearn.model_selection import train_test_split
MAE_map_ls = []
MAE_ml_ls = []
MSE_map_ls = []
MSE_ml_ls = []
MSE_map_winfo = []
ML_weights = []
MAP_weights = []

trainSize = []
trainFracArr = np.linspace(0.1, 0.9, 20)

for frac in trainFracArr:
    hold_mlMSE = []
    hold_mapMSE = []
    hold_mapMSE_winfo = []
    # Train test split
    # X_train, X_test, y_train, y_test = train_test_split(phi, t, train_size=frac)  # 0 cuz 0 corresponds to the RSLR var

    X_train = phi[:int(len(phi)*frac) - 1, :]
    X_test = phi[int(len(phi)*frac):, :]
    y_train = t[:int(len(phi)*frac) - 1]
    y_test = t[int(len(phi) * frac):]

    B, a, eff_lambda, itr = bml.iterative_prog(X_train, y_train)  # std of 0.5 cuz i normalize variables between 0 and 1

    var_weights_map = bml.leastSquares(eff_lambda, X_train, y_train)
    map_MSE = bml.returnMSE(X_test, var_weights_map, y_test)
    map_MAE = bml.returnMAE(X_test, var_weights_map, y_test)
    # hold_mapMSE.append(map_MSE)

    var_weights_ml = bml.leastSquares(0, X_train, y_train)  # recall that ml is when lambda is 0
    ml_MSE = bml.returnMSE(X_test, var_weights_ml, y_test)
    ml_MAE = bml.returnMAE(X_test, var_weights_ml, y_test)
    # hold_mlMSE.append(ml_MSE)
    # Append train size for plotting
    trainSize.append(frac)
    MSE_ml_ls.append(ml_MSE)
    MSE_map_ls.append(map_MSE)
    MAE_ml_ls.append(ml_MAE)
    MAE_map_ls.append(map_MAE)
    ML_weights.append(var_weights_ml)
    MAP_weights.append(var_weights_map)

plt.figure()
plt.plot(trainSize, MSE_map_ls, label='MAP')
plt.plot(trainSize, MSE_ml_ls, label='MLE')
# plt.plot(trainSize, MSE_map_winfo, label='informed MAP')
# plt.ylim(0, 500)
plt.title('MSE versus Train Size')
plt.ylabel('MSE')
plt.xlabel('Train Size')
plt.ylim(0, 500)
plt.legend()
plt.show()

# Find the index of the lowest MSE for ML and MAP
idxMAP = MSE_map_ls.index(min(MSE_map_ls))
best_map_weights = MAP_weights[idxMAP]
idxML = MSE_ml_ls.index(min(MSE_ml_ls))
best_ml_weights = ML_weights[idxML]
print('Best training Fraction (MAP): ', trainSize[idxMAP], '\n',
      'MSE: ', MSE_map_ls[idxMAP], '\n',
      'MAE: ', MAE_map_ls[idxMAP], '\n',
      'RMSE: ', np.sqrt(MSE_map_ls[idxMAP]))
print('Best training Fraction (ML): ', trainSize[idxML], '\n',
      'MSE: ', MSE_ml_ls[idxML], '\n',
      'MAE: ', MAE_ml_ls[idxML], '\n',
      'RMSE: ', np.sqrt(MSE_ml_ls[idxML]))

# To me this plot seems to say that test-train splits should not be used for model evaluation because I
# seem to have a lot of variability in the MSE

# NEXT: To either extract the predicted values (should be self explanatory from the returnMSE function) or to
# collect better data in order to put in this. An iterative CV evaluation should also be created

