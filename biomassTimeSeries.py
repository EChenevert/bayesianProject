import pullDataFuncs as pf

# This is focused on Biomass! remeber that biomass as generally bee collected during winter for some reason ...
# or on the month that i defined as winter...
# Check this time series .. assuming independence cuz (a) only from 2016 on (b)
dfs = pf.load_data()
seasonal = pf.average_byyear_bysite_seasonal(dfs).dropna(subset=['Belowground Live Biomass (g/m2)',
                                                                 'Accretion Rate (mm/yr)'])  # I want this variable!!!!!
seasonal['Year (yyyy)'] = seasonal['Year (yyyy)'].astype(int).astype(str)
years = seasonal.groupby(['Simple site', 'Year (yyyy)']).median()

dfs2 = pf.load_data()
bysite = pf.average_bysite(dfs2)['Community']
# Clean the seasonal a bit
thresd = years.dropna(thresh=years.shape[0]*0.8, how='all', axis=1)
# thresd = seasonal
# only take the important interesting variables
did = thresd[[
    'Accretion Rate (mm/yr)', 'Belowground Live Biomass (g/m2)', 'Latitude',
    'Belowground Dead Biomass (g/m2)', 'Soil Porewater Temperature (°C)', 'Soil Porewater Specific Conductance (uS/cm)',
    'Soil Porewater Salinity (ppt)', 'Verified Pin Height (mm)'
]]

# # Create the Subsidence rate
# did['Shallow Subsidence Rate (mm/yr)'] = did['Accretion Rate (mm/yr)'] - df['Surface Elevation Change Rate (cm/y)']*10
# df['SEC Rate (mm/yr)'] = df['Surface Elevation Change Rate (cm/y)']*10
# df['SLR (mm/yr)'] = 2.0  # from jankowski
# df['Deep Subsidence Rate (mm/yr)'] = ((3.7147 * df['Latitude']) - 114.26)*-1
# df['RSLR (mm/yr)'] = df['Shallow Subsidence Rate (mm/yr)'] + df['Deep Subsidence Rate (mm/yr)'] + df['SLR (mm/yr)']


# Attach external yearly variables
import pandas as pd
# Expand on the site and year average dataset
wl = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\11966_WaterLevelRange_CalendarYearTimeSeriesAll\11966.csv",
                 encoding='unicode_escape')
# Make the site name simple: Only use the CRMS0000 - H sites tho this time ... should be more consistent
wltest = wl[wl["Station_ID"].str.contains("H") == True]
# Now instill the simple site
wltest['Simple site'] = [i[:8] for i in wltest['Station_ID']]
# Only take relevant variables and set index to simple site and year for concatenation with other df
wldf = wltest[['Tide_Amp (ft)', 'calendar_year', 'avg_flooding (ft)', '90%thUpper_flooding (ft)',
               '10%thLower_flooding (ft)', 'std_deviation_avg_flooding (ft)', 'Simple site']]
wldf['calendar_year'] = wldf['calendar_year'].astype(str)
reWL = wldf.groupby(['Simple site', 'calendar_year']).median()
ccdf = pd.concat([reWL, did], axis=1, join='inner')


# Add the percent time flooded variable:
pfl = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\11968_PercentFlooded_CalendarYearTimeSeries\11968.csv",
                  encoding="unicode_escape")
pfltest = pfl[pfl["Station_ID"].str.contains("H") == True]
pfltest['Simple site'] = [i[:8] for i in pfltest['Station_ID']]
pfldf = pfltest[['Simple site', 'Year', 'avg_percentflooded (%)']]
pfldf['Year'] = pfldf['Year'].astype(str)
rePFL = pfldf.groupby(['Simple site', 'Year']).median()
pwccdf = pd.concat([rePFL, ccdf], axis=1, join='inner')

# add the remote sensing data (YEARLY):
# NDVI
ndviTS = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\table_demo_NDVI_CRMS3.csv", encoding='unicode_escape')
ndviTS['Year'] = [i[:4] for i in ndviTS['system:index']]
newNDVIts = ndviTS.drop('system:index', axis=1)
dicNDVI = {'Year': [], 'Simple site': [], 'NDVI': []}

listSites = list(newNDVIts.columns.values)
listSites.remove('.geo')
listSites.remove('imageId')
listSites.remove('Year')
for col in listSites:
    diffdf = newNDVIts[['Year', col]]
    diffdf['Simple site'] = col
    dicNDVI['Year'] = dicNDVI['Year'] + list(diffdf['Year'])
    dicNDVI['Simple site'] = dicNDVI['Simple site'] + list(diffdf['Simple site'])
    dicNDVI['NDVI'] = dicNDVI['NDVI'] + list(diffdf[col])

dfndvi = pd.DataFrame(dicNDVI)
dfndvigb = dfndvi.groupby(['Simple site', 'Year']).median()

# TSS
tssTS = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\table_demo_TSS_CRMS.csv", encoding='unicode_escape')
tssTS['Year'] = [i[:4] for i in tssTS['system:index']]
newTSSts = tssTS.drop('system:index', axis=1)
dicTSS = {'Year': [], 'Simple site': [], 'TSS': []}

tssSites = list(newTSSts.columns.values)
tssSites.remove('.geo')
tssSites.remove('imageId')
tssSites.remove('Year')
for col in tssSites:
    diffdf = newTSSts[['Year', col]]
    diffdf['Simple site'] = col
    dicTSS['Year'] = dicTSS['Year'] + list(diffdf['Year'])
    dicTSS['Simple site'] = dicTSS['Simple site'] + list(diffdf['Simple site'])
    dicTSS['TSS'] = dicTSS['TSS'] + list(diffdf[col])

dftss = pd.DataFrame(dicTSS)
dftssgb = dftss.groupby(['Simple site', 'Year']).median()

# Combine the datasets
rsdf = pd.concat([dftssgb, dfndvigb], join='inner', axis=1)
alldf = pd.concat([rsdf, pwccdf], join='inner', axis=1)

# Conduct outlier removal here prior to the adding of strings
alldf = pf.outlierRemoval(alldf, thres=3).reset_index()

# Attach the correct basin and marsh type to each specific site.
marshComRef = bysite.reset_index()
dicMarshSite = dict(zip(marshComRef['Simple site'], marshComRef['Community']))
alldf['Community'] = [dicMarshSite[site] for site in alldf['Simple site']]

# alldf.to_csv("D:\\Etienne\\fall2022\\CRMS_data\\timeseries_CRMS.csv")

## Attach the stationary distance variables and stationary water-area-perimeter variables
allsummed = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\allSummedUp5km.csv", encoding='unicode_escape')\
    .set_index('Field1')
riverDist = pd.read_csv(r"D:\Etienne\fall2022\CRMS_data\totalDataAndRivers.csv", encoding='unicode_escape')[[
    'Field1', 'distance_to_river_m'
]].set_index('Field1')
sumDist = pd.concat([allsummed, riverDist], axis=1)
sumDist['LengthScale (perimeter/area)'] = sumDist['Shape_Length']/sumDist['Shape_Area']
sumDist = sumDist.reset_index()
# stitch the terms to the timeseries
alldf = alldf.reset_index()
for col in sumDist.columns[1:]:
    dicDists = dict(zip(sumDist['Field1'], sumDist[col]))
    alldf[col] = [dicDists[site] for site in alldf['Simple site']]

alldf.to_csv("D:\\Etienne\\fall2022\\CRMS_data\\biomassTimeSeries_CRMS.csv")
