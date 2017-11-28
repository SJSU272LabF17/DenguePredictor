import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_error

features_sj = pd.read_pickle('../data/all_sj.pkl')
features_iq = pd.read_pickle('../data/all_iq.pkl')

# print(features_sj)
# print(features_iq)

total_cases = pd.read_csv('../data/dengue_labels_train.csv')
cases_sj = total_cases[total_cases['city'] == 'sj']['total_cases']
cases_iq = total_cases[total_cases['city'] == 'iq']['total_cases']

# print(cases_sj)
# print(cases_iq)

features_sj['month'] = features_sj.index.month
features_iq['month'] = features_iq.index.month

# Creating Mean

features_sj['ndvi_mean'] = (features_sj['ndvi_ne'] + features_sj['ndvi_nw'] + features_sj['ndvi_se'] + features_sj['ndvi_sw']) / 4.0
features_iq['ndvi_mean'] = (features_iq['ndvi_ne'] + features_iq['ndvi_nw'] + features_iq['ndvi_se'] + features_iq['ndvi_sw']) / 4.0

# train test split
#sj

# choose split dates
sj_valid_split = '2003-4-20'
sj_test_split = '2008-4-27' # this will split between pre and post submission dates

# split into train, valid, test (no y)
sj_train = features_sj.loc[:sj_valid_split]
sj_Xtrain = sj_train
sj_ytrain = cases_sj[:len(sj_train)]

sj_valid = features_sj.loc[sj_valid_split : sj_test_split]
sj_Xvalid = sj_valid
sj_yvalid = cases_sj[len(sj_train):]

sj_test = features_sj.loc[sj_test_split:]
sj_Xtest = sj_test

# print(len(sj_train), len(sj_Xtrain), len(sj_ytrain))
# print(len(sj_valid), len(sj_Xvalid), len(sj_yvalid))
# print(len(sj_test), len(sj_Xtest))

#iq

# choose split dates
iq_valid_split = '2007-7-01'
iq_test_split = '2010-7-01' # this will split between pre and post submission dates

# split
iq_train = features_iq.loc[:iq_valid_split]
iq_Xtrain = iq_train
iq_ytrain = cases_iq[:len(iq_Xtrain)]

iq_valid = features_iq.loc[iq_valid_split : iq_test_split]
iq_Xvalid = iq_valid
iq_yvalid = cases_iq[len(iq_train):]

iq_test = features_iq.loc[iq_test_split:]
iq_Xtest = iq_test

# check the lengths
# print(len(iq_train), len(iq_Xtrain), len(iq_ytrain))
# print(len(iq_valid), len(iq_Xvalid), len(iq_yvalid))
# print(len(iq_test), len(iq_Xtest))


# sj monthly trend

lr_sj = LinearRegression()
X = pd.get_dummies(sj_Xtrain['month'], prefix='month')
y = sj_ytrain.values

lr_sj.fit(X, y)
monthly_trend_train = pd.Series(lr_sj.predict(X)).rolling(3, min_periods = 1).mean()
sj_residuals_train = y - monthly_trend_train


# on validation data
# note: monthly trend does not need previous weeks data, so this can use the validation set
Xtest = pd.get_dummies(sj_Xvalid['month'], prefix='month')
ytest = sj_yvalid.values
monthly_trend_valid = pd.Series(lr_sj.predict(Xtest)).rolling(3, min_periods=1).mean()
sj_residuals_test = ytest - monthly_trend_valid

# plot
plt.plot(lr_sj.predict(Xtest))
plt.plot(monthly_trend_valid)
plt.plot(ytest)
plt.show()

# print(mean_absolute_error(lr_sj.predict(Xtest), ytest))
# print(mean_absolute_error(monthly_trend_valid, ytest))

# Find the residuals of the monthly trend

plt.figure(figsize=(16, 4))
plt.plot(sj_residuals_test, label = 'residuals', linewidth = 3)
plt.plot(ytest, 'g-*', alpha = .4, label = 'true values')
plt.plot(monthly_trend_valid, alpha = .4, label = 'monthly trend')
plt.title('True Values, Monthly Trend, and Residuals of Dengue Cases in San Juan, Puerto Rico')
plt.legend()
plt.show()

# print(np.mean(sj_residuals_test))
# print(len(sj_residuals_train), len(sj_residuals_test))

# get all residuals
lr_sj_month = LinearRegression()
X_months = pd.get_dummies(features_sj['month'], prefix='month')[:936]
Xtest_months = pd.get_dummies(features_sj['month'], prefix='month')[936:]
y = cases_sj.values
lr_sj_month.fit(X_months, y)
monthly_trend = pd.Series(lr_sj_month.predict(X_months)).rolling(3, min_periods=1).mean()
sj_residuals_all = y - monthly_trend

# plot all residuals
plt.figure(figsize=(32, 4))
plt.plot(features_sj[:936].index, sj_residuals_all, label = 'residuals')
plt.xlabel('Date', fontsize = 25)
plt.ylabel('Num of Cases', fontsize=25)
plt.title('Residuals of Dengue Cases in San Juan & Puerto Rico', fontsize = 35)
plt.show()

# Histogram of Residuals
plt.figure(figsize=(10,6))
plt.hist(sj_residuals_all, bins = 50)
plt.xlabel('Difference between Actual Cases and Monthly Trend (Residuals)', fontsize=15)
plt.title('Distribution of Residuals', fontsize = 25)
plt.show()

# Plot sum of correlation between rolling mean of all features and residuals
# create features of correlation between feature and resid at each rolling mean
features_scores = pd.DataFrame()
for i in range (2, 80):
    features = sj_train.drop(['month', 'weekofyear', 'month'], axis = 1)
    features = features.rolling(i, min_periods=1).mean()
    features['resid'] = sj_residuals_train.values
    features_corr = features.corr()
    features_scores[str(i)] = features_corr['resid']

# create features of max absolute value of corr and the window
feature = []
abs_corr = []
window = []

for i in features_scores.T.drop('resid', axis = 1).columns:
    feature.append(i)
    abs_corr.append(max(abs(features_scores.T[i])))
    window.append(features_scores.T[abs(features_scores.T[i]) == max(abs(features_scores.T[i]))].index[0])

scores_features = pd.DataFrame([feature, abs_corr, window]).T
scores_features.columns = ('feature', 'abs_corr', 'window')
scores_features.sort_values('abs_corr', ascending = False)

plt.figure(figsize=(16,9))
for i in (u'ndvi_ne', u'ndvi_nw', u'ndvi_se', u'ndvi_sw', u'ndvi_mean', u'precipitation_amt_mm',
       u'reanalysis_air_temp_k', u'reanalysis_avg_temp_k',
       u'reanalysis_dew_point_temp_k', u'reanalysis_max_air_temp_k',
       u'reanalysis_min_air_temp_k', u'reanalysis_precip_amt_kg_per_m2',
       u'reanalysis_relative_humidity_percent',
       u'reanalysis_sat_precip_amt_mm',
       u'reanalysis_specific_humidity_g_per_kg', u'reanalysis_tdtr_k',
       u'station_avg_temp_c', u'station_diur_temp_rng_c',
       u'station_max_temp_c', u'station_min_temp_c', u'station_precip_mm'):
    plt.plot(features_scores.T[i])
plt.title('Correlation of Rolling Mean of Features to Residuals', fontsize = 30)
plt.xlabel('Number of Weeks back to take Rolling Mean', fontsize=20)
plt.ylabel('Strength of Correlation (Absolute Value)', fontsize=20)
plt.legend(loc=(1,0), fontsize= 15)
plt.show()

plt.figure(figsize=(16,9))
for i in (u'ndvi_ne', u'ndvi_nw', u'ndvi_se', u'ndvi_sw'):
    plt.plot(features_scores.T[i])
plt.title('Correlation of Rolling Mean of Features to Residuals', fontsize = 30)
plt.xlabel('Number of Weeks back to take Rolling Mean', fontsize=20)
plt.ylabel('Strength of Correlation (Absolute Value)', fontsize=20)
plt.legend(loc=(1,0), fontsize= 15)


plt.figure(figsize=(16,9))
for i in (
       u'reanalysis_air_temp_k', u'reanalysis_avg_temp_k',
       u'reanalysis_dew_point_temp_k', u'reanalysis_max_air_temp_k',
       u'reanalysis_min_air_temp_k',
       u'station_avg_temp_c', u'station_diur_temp_rng_c',
       u'station_max_temp_c', u'station_min_temp_c'):
    plt.plot(features_scores.T[i])
plt.title('Correlation of Rolling Mean of Features to Residuals', fontsize = 30)
plt.xlabel('Number of Weeks back to take Rolling Mean', fontsize=20)
plt.ylabel('Strength of Correlation (Absolute Value)', fontsize=20)
plt.legend(loc=(1,0), fontsize= 15)
plt.show()


plt.figure(figsize=(16,9))
for i in (u'precipitation_amt_mm', u'station_precip_mm',
       u'reanalysis_precip_amt_kg_per_m2',
       u'reanalysis_relative_humidity_percent',
       u'reanalysis_sat_precip_amt_mm',
       u'reanalysis_specific_humidity_g_per_kg', u'reanalysis_tdtr_k'):
    plt.plot(features_scores.T[i])
plt.title('Correlation of Rolling Mean of Features to Residuals', fontsize = 30)
plt.xlabel('Number of Weeks back to take Rolling Mean', fontsize=20)
plt.ylabel('Strength of Correlation (Absolute Value)', fontsize=20)
plt.legend(loc=(1,0), fontsize= 15)
plt.show()
