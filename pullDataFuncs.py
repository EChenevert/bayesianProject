import pandas as pd
import numpy as np
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy import stats


# Functions used to load data
def organized_iteryears(date_col_name, df):
    ''' This method creates a data column that indicates what year the sample
    (or data) was collected or observed'''
    datetimeArr = pd.to_datetime(df[date_col_name], format='%m/%d/%Y')
    years = datetimeArr.dt.year
    return years

def organized_itermons(date_col_name, df):
    '''This is an iterdates method for the hydro data, which is logged into
    the csv more cleaning. This increases speed
    @params:
        date_col_names = is the name of the date column
        df = is the dataframe the date column is in'''
    datetimeArr = pd.to_datetime(df[date_col_name], format='%m/%d/%Y')
    months = datetimeArr.dt.month
    return months


def add_avgAccretion(accdf):
    avg_accretion = (accdf['Accretion Measurement 1 (mm)'] + accdf['Accretion Measurement 2 (mm)'] +
                     accdf['Accretion Measurement 3 (mm)'] + accdf['Accretion Measurement 4 (mm)']) / 4
    avg_accretion = pd.DataFrame(avg_accretion, columns=['Average Accretion (mm)'], index=accdf.index.values)
    newdf = pd.concat([accdf, avg_accretion], axis=1)
    return newdf


def add_accretionRate(accdf):
    accdf['Average Accretion (mm)'] = (accdf['Accretion Measurement 1 (mm)'] + accdf['Accretion Measurement 2 (mm)'] +
                                       accdf['Accretion Measurement 3 (mm)'] +
                                       accdf['Accretion Measurement 4 (mm)']) / 4
    accdf['Sample Date (mm/dd/yyyy)'] = pd.to_datetime(accdf['Sample Date (mm/dd/yyyy)'],
                                                       format='%m/%d/%Y')

    accdf['Establishment Date (mm/dd/yyyy)'] = pd.to_datetime(accdf['Establishment Date (mm/dd/yyyy)'],
                                                              format='%m/%d/%Y')

    accdf['Delta time (days)'] = accdf['Sample Date (mm/dd/yyyy)'] - \
                                 accdf['Establishment Date (mm/dd/yyyy)']

    accdf['Delta time (days)'] = accdf['Delta time (days)'].dt.days
    accdf['Delta Time (decimal_years)'] = accdf['Delta time (days)'] / 365
    accdf['Accretion Rate (mm/yr)'] = accdf['Average Accretion (mm)'] / accdf['Delta Time (decimal_years)']

    return accdf

def add_seChangeRateOLS(byyrNsitedf):
    """
    This will calculate the SEC rate using a RANSAC Regressor. Sites with less than 5 Pin height Recordings are excluded
    :param byyrNsitedf: dataframe with pin height measurements as well as site names.
    :return: dictionary with RANSAC slope values
    """
    df = byyrNsitedf[['Simple site', 'Year (yyyy)', 'Verified Pin Height (mm)']].dropna()
    sitenames = df['Simple site'].unique()
    secDict = {'Simple site': [], 'SEC Rate RANSAC (mm)': []}
    for site in sitenames:
        dfnew = df[df['Simple site'] == site]
        if len(dfnew) >= 2:
            y = dfnew['Verified Pin Height (mm)'].to_numpy()
            x = dfnew['Year (yyyy)'].to_numpy()
            plt.plot(x, y, 'o')

            ransac = linear_model.RANSACRegressor()
            ransac.fit(x.reshape(-1, 1), y)
            m = float(ransac.estimator_.coef_)
            # hold_dic['Accretion Rate mm/yr (slope value)'].append(m)
            b = float(ransac.estimator_.intercept_)

            # plt.plot(x, m * x + b)
            # plt.title(str(site) + " : " + str(m))
            # plt.xlabel("Year (yyyy)")
            # plt.ylabel("Verified Pin Height (mm)")
            # plt.show()

            secDict['Simple site'].append(site)  # Append the name of the site to dict
            secDict['SEC Rate RANSAC (mm)'].append(m)
    return secDict


def load_data():
    '''This loads all the crms data currently in the data folder of this package'''
    soil_properties = pd.read_csv(r"D:\Etienne\summer2022_CRMS\run_experiments\CRMS_Soil_Properties\CRMS_Soil_Properties.csv", encoding='unicode escape')
    # hourly_hydro = pd.read_csv(r"C:\Users\etachen\Documents\PyCharmProjs\datasetsCRMS\main\data\CRMS_Continuous_Hydrographic.csv", encoding='unicode escape')
    monthly_hydro = pd.read_csv(r"D:\Etienne\summer2022_CRMS\run_experiments\CRMS_Discrete_Hydrographic\CRMS_Discrete_Hydrographic.csv", encoding='unicode escape')
    marsh_vegetation = pd.read_csv(r"D:\Etienne\summer2022_CRMS\run_experiments\CRMS_Marsh_Vegetation\CRMS_Marsh_Vegetation.csv", encoding='unicode escape')
    forest_vegetation = pd.read_csv(r"D:\Etienne\summer2022_CRMS\run_experiments\CRMS_Forest_Vegetation\CRMS_Forest_Vegetation.csv", encoding='unicode escape')
    accretion = pd.read_csv(r"D:\Etienne\summer2022_CRMS\run_experiments\CRMS_Accretion\CRMS_Accretion.csv", encoding='unicode escape')
    biomass = pd.read_csv(r"D:\Etienne\summer2022_CRMS\run_experiments\CRMS_Biomass\CRMS_Biomass.csv", encoding='unicode escape')
    surface_elevation = pd.read_csv(r"D:\Etienne\summer2022_CRMS\run_experiments\CRMS_Surface_Elevation\CRMS_Surface_Elevation.csv", encoding='unicode escape')

    # # I only want the first ~8 cm of the soil properties data
    # soil_properties = soil_properties[(soil_properties['Sample Depth (cm)'] == '0 to 4') |
    #                                   (soil_properties['Sample Depth (cm)'] == '4 to 8') |
    #                                   (soil_properties['Sample Depth (cm)'] == '8 to 12')]

    dfs = [
        accretion,
        soil_properties,
        # hourly_hydro,
        monthly_hydro,
        marsh_vegetation,
        forest_vegetation,
        biomass,
        surface_elevation
    ]
    # Making a common column for dtermining the site name
    for d in dfs:
        # if 'Station_ID' in dfs[d].columns:
        #     dfs[d]['Simple site'] = [i[:8] for i in dfs[d]['Station_ID']]
        if 'Station ID' in d.columns:  # For surface Elevation, soil Properties, marsh vegetation, accretion
            d['Simple site'] = [i[:8] for i in d['Station ID']]
        if 'CPRA Station ID' in d.columns:  # For Monthly hydro,
            d['Simple site'] = [i[:8] for i in d['CPRA Station ID']]

        # Setting the YEARLY dates
        # if 'calendar_year' in dfs[d].columns:
        #     dfs[d]['Year (yyyy)'] = dfs[d]['calendar_year']
        if 'Sample Date (mm/dd/yyyy)' in d.columns:  # Accretion, soil properties, surface elevation
            d['Year (yyyy)'] = organized_iteryears('Sample Date (mm/dd/yyyy)', d)
        if 'Date (mm/dd/yyyy)' in d.columns:  # Monthly Hydro,
            d['Year (yyyy)'] = organized_iteryears('Date (mm/dd/yyyy)', d)
        if 'Collection Date (mm/dd/yyyy)' in d.columns:  # Marsh Veg,
            d['Year (yyyy)'] = organized_iteryears('Collection Date (mm/dd/yyyy)', d)

        # # Set the MONTHLY dates
        # if 'calendar_year' in dfs[d].columns:
        #     dfs[d]['Month (mm)'] = 0  # this means that this data is averaged over a length of years so there is no monthly data
        if 'Sample Date (mm/dd/yyyy)' in d.columns:  # Accretion, soil properties, surface elevation
            d['Month (mm)'] = organized_itermons('Sample Date (mm/dd/yyyy)', d)
        if 'Date (mm/dd/yyyy)' in d.columns:  # Monthly Hydro,
            d['Month (mm)'] = organized_itermons('Date (mm/dd/yyyy)', d)
        if 'Collection Date (mm/dd/yyyy)' in d.columns:  # Marsh Veg,
            d['Month (mm)'] = organized_itermons('Collection Date (mm/dd/yyyy)', d)


        # Add basins: I manually put each site into a basin category, this was done from teh CRMS louisiana website map
        d['Basins'] = np.arange(len(d['Simple site']))  # this is for appending a basins variable

        if 'Accretion Measurement 1 (mm)' in d.columns:
            # dfs[d] = add_avgAccretion(dfs[d])
            d = add_accretionRate(d)

    return dfs


def combine_dataframes(dfs):
    ''' this function will take the dataframes and concatenate them (stack them) based on
    their index
    NOTE: Test this again after doing the groupby functions.
    The index and concatenation may be slighly off'''
    i = dfs[0].index.to_flat_index()
    print(i)
    dfs[0].index = i
    full_df = dfs[0]
    for j in range(1, len(dfs)):  # always make sure this is the correct range length.... its confusing me
        # full_df = pd.concat([full_df, dfs[j]], axis=1, ignore_index=False).drop_duplicates()
        idx = dfs[j].index.to_flat_index()
        dfs[j].index = idx
        full_df = pd.concat([full_df, dfs[j]], join='outer', axis=1)
        # full_df = full_df.join(dfs[j], how={'left', 'outer'})
        full_df = full_df.loc[:, ~full_df.columns.duplicated()]
    return full_df


def average_bysite(dfs):
    '''Below is a df that is craeted by averaging across all years per crms site.
    NOTE: That varaibles constructed by strings are annihilated (due to the .median()command)'''
    dfs_copy = []
    for d in dfs:
        dfs_copy.append(d.copy())
    for n in range(len(dfs_copy)):
        df = dfs_copy[n].groupby(['Simple site']).median()
        # basins = dfs_copy[n].groupby(['Simple site'])  # ['Basins'].agg(pd.Series.mode).to_frame()
        # dfs[n] = pd.concat([df, basins], axis=1)
        # weird thing i decided to do spontaneously, prob can implement better to include more categorical variables
        if 'Community' in dfs_copy[n].columns:
            community = dfs_copy[n].groupby(['Simple site'])['Community'].agg(pd.Series.mode).to_frame()
            # dfs[n] = pd.concat([df, community], axis=1)
            dfs_copy[n] = pd.concat([df, community], axis=1)

        else:
            dfs_copy[n] = df

    full_df = combine_dataframes(dfs_copy)
    return full_df


def average_byyear_bysite(dfs):
    '''This will create a dataframe that incorporates data from the 17 years and each season of collection
    It can also take byyear_bymonth,bysite'''
    dfs_copy = []
    for d in dfs:
        dfs_copy.append(d.copy())
    for n in range(len(dfs_copy)):
        # dfs[n]['Season'] = [1 if i > 4 and i <= 10 else 2 for i in dfs[n]['Month (mm)']]
        df = dfs_copy[n].groupby(['Simple site', 'Year (yyyy)']).median()  # Excludes extreme events
        # basins = dfs_copy[n].groupby(['Simple site', 'Year (yyyy)'])['Basins'].agg(pd.Series.mode).to_frame()
        # weird thing i decided to do spontaneously, prob can implement better to include more categorical variables
        if 'Community' in dfs_copy[n].columns:
            community = dfs_copy[n].groupby(['Simple site', 'Year (yyyy)'])['Community'].agg(pd.Series.mode).to_frame()
            dfs_copy[n] = pd.concat([df, community], axis=1)
        else:
            dfs_copy[n] = df
        # if 'Common Name As Currently'
    full_df = combine_dataframes(dfs_copy)
    full_df = full_df.reset_index().rename(columns={'level_0': 'Simple site', 'level_1': 'Year (yyyy)'})

    return full_df


def average_byyear_bysite_seasonal(dfs):
    '''This will create a dataframe that incorporates data from the 17 years and each season of collection
    It can also take byyear_bymonth,bysite'''
    for n in range(len(dfs)):
        dfs[n]['Season'] = [1 if i > 4 and i <= 10 else 2 for i in dfs[n]['Month (mm)']]
        df = dfs[n].groupby(['Simple site', 'Year (yyyy)', 'Season']).median()  # Excludes extreme events
        # basins = dfs[n].groupby(['Simple site', 'Year (yyyy)', 'Season'])['Basins'].agg(pd.Series.mode).to_frame()
        # weird thing i decided to do spontaneously, prob can implement better to include more categorical variables
        if 'Community' in dfs[n].columns:
            community = dfs[n].groupby(['Simple site', 'Year (yyyy)', 'Season'])['Community'].agg(pd.Series.mode).to_frame()
            dfs[n] = pd.concat([df, community], axis=1)
    full_df = combine_dataframes(dfs)
    full_df = full_df.reset_index().rename(columns={'level_0':'Simple site', 'level_1':'Year (yyyy)', 'level_2':'Season'})

    return full_df


def outlierRemoval(df, thres=3):
    df = df.dropna()  # Drop all nans
    true = False
    if 'Longitude' in df.columns.values or 'Latitude' in df.columns.values:
        print('in')
        true = True
        saveorg = df[['Longitude', 'Latitude']]
        df = df.drop(['Longitude', 'Latitude'], axis=1)
    # df = df.loc[(df != 0).all(axis=1)]  # Drop all zeros
    df = df.apply(pd.to_numeric)
    length = len(df.columns.values)
    for col in df.columns.values:
        df[col + "_z"] = stats.zscore(df[col])
    for col in df.columns.values[length:]:
        df = df[np.abs(df[col]) < thres]
    df = df.drop(df.columns.values[length:], axis=1)
    if true == True:
        df = pd.concat([df, saveorg], join='inner', axis=1)
    return df

def addbasins(org_sitename, listvars, basinStr, filename):
    df = pd.read_csv(filename, encoding='unicode_escape')[listvars]
    df['Simple site'] = [i[:8] for i in df[org_sitename]]
    df = df.drop(org_sitename, axis=1).groupby('Simple site').median()
    df['Basin'] = basinStr
    return df


def exhaustiveFeatSelec(predictors, target, max_feats):

    mlr = linear_model.LinearRegression()
    feature_selector = ExhaustiveFeatureSelector(mlr,
                                                 min_features=1,
                                                 max_features=max_feats,
                                                 scoring='neg_mean_absolute_error',
                                                 # minimizes variance, at expense of bias
                                                 cv=5)  # 10 fold cross-validation

    efsmlr = feature_selector.fit(predictors, target.values.ravel())  # these are not scaled... to reduce data leakage
    print('Best CV r2 score: %.2f' % efsmlr.best_score_)
    print('Best subset (indices):', efsmlr.best_idx_)
    print('Best subset (corresponding names):', efsmlr.best_feature_names_)
    return list(efsmlr.best_feature_names_)



