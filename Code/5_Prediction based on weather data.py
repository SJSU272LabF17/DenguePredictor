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
#sj_predicts = pd.read_csv('../data/sj_predicts.csv')

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


# print(mean_absolute_error(lr_sj.predict(Xtest), ytest))
# print(mean_absolute_error(monthly_trend_valid, ytest))

# Find the residuals of the monthly trend

plt.figure(figsize=(16, 4))
plt.plot(sj_residuals_test, label = 'residuals', linewidth = 3)
plt.plot(ytest, 'g-*', alpha = .4, label = 'true values')
plt.plot(monthly_trend_valid, alpha = .4, label = 'monthly trend')
plt.title('True Values, Monthly Trend, and Residuals of Dengue Cases in San Juan, Puerto Rico')
plt.legend()


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


# Histogram of Residuals
plt.figure(figsize=(10,6))
plt.hist(sj_residuals_all, bins = 50)
plt.xlabel('Difference between Actual Cases and Monthly Trend (Residuals)', fontsize=15)
plt.title('Distribution of Residuals', fontsize = 25)


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


# print(features_sj["reanalysis_sat_precip_amt_mm"])#should return only reanalysis_sat_precip_amt_mm column of featurres_sj but also returning week_start_date

# for i in range(1, 80):
#     plt.plot(float(features_sj['reanalysis_sat_precip_amt_mm'].rolling(i, min_periods=1)).mean(), alpha = .25)   #converting week_start_date+reanalysis_sat_precip_amt_mm to float here(which shouldn't)
#     plt.title('Rolling Mean of reanalysis_sat_precip_amt_mm - ' + str(i) + ' Week Window')
#     #plt.savefig('../plots/Rolling Mean of reanalysis_sat_precip_amt_mm -' + str(i) + '.png')
# for i in range(1, 80):
#     plt.plot(features_sj(['reanalysis_sat_precip_amt_mm']).rolling(i, min_periods = 1).mean(), alpha = .25)
#     plt.title('Rolling Mean of reanalysis_sat_precip_amt_mm - ' + str(i) + ' Week Window')
#     plt.show()
    #plt.savefig('../plots/rolling' + str(i) + '.png')



# Plot sum of correlation between rolling std of all features and residuals
# create df of correlation between feature and resid at each rolling mean
df_scores = pd.DataFrame()
for i in range (2, 60):
    feature1 = sj_train.drop(['month', 'weekofyear', 'month'], axis = 1)
    feature1 = feature1.rolling(i, min_periods=1).std()
    feature1['resid'] = sj_residuals_train.values
    features_corr = feature1.corr()
    features_scores[str(i)] = features_corr['resid']





# create features of max absoute vaule of corr and the window
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



plt.figure(figsize=(10,6))
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
plt.title('Correlation of Rolling STD of Features to Residuals', fontsize = 15)
plt.xlabel('Number of Weeks back to take STD Mean', fontsize=10)
plt.ylabel('Strength of Correlation (Absolute Value)', fontsize=10)
plt.legend(loc=(1,0), fontsize= 4)
#plt.show()


# Plot sum of correlation between rolling std of all features and residuals
# create df of correlation between feature and resid at each rolling mean
features_scores = pd.DataFrame()
for i in range (2, 60):
    feature = sj_train.drop(['month', 'weekofyear', 'month'], axis = 1)
    feature = feature.shift(i)[i:]
    feature['resid'] = sj_residuals_train.values[i:]
    features_corr = feature.corr()
    features_scores[str(i)] = features_corr['resid']



# create df of max absoute vaule of corr and the window
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
# print(scores_features)


plt.figure(figsize=(10,6))
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
plt.title('Correlation of Weeks shifted back of Features to Residuals', fontsize = 15)
plt.xlabel('Number of Weeks shift back', fontsize=10)
plt.ylabel('Strength of Correlation (Absolute Value)', fontsize=10)
plt.legend(loc=(1,0), fontsize= 4)
#plt.show()




# Exponentially Weighted Mean
# Plot sum of correlation between rolling std of all features and residuals
# create df of correlation between feature and resid at each rolling mean
features_scores = pd.DataFrame()
for i in range (2, 60):
    feature = sj_train.drop(['month', 'weekofyear', 'month'], axis = 1)
    feature = feature.ewm(span = i).mean()
    feature['resid'] = sj_residuals_train.values
    features_corr = feature.corr()
    features_scores[str(i)] = features_corr['resid']

# create df of max absoute vaule of corr and the window
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

#print(scores_features)

plt.figure(figsize=(10,6))
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
plt.title('Correlation of Exponentially Weighted Mean of Features to Residuals', fontsize = 15)
plt.xlabel('Number of Weeks back to take Exponentially Weighted Mean', fontsize=10)
plt.ylabel('Strength of Correlation (Absolute Value)', fontsize=10)
plt.legend(loc=(1,0), fontsize= 4)
#plt.show()



# munge data

# max roll back set to 59

# rolling means df
Xtrain_means1 = features_sj['station_avg_temp_c'].rolling(window = 52).mean()[60:675]
Xtrain_means2 = features_sj['ndvi_ne'].rolling(window = 53).mean()[60:675]
Xtrain_means3 = features_sj['ndvi_sw'].rolling(window = 2).mean()[60:675]
Xtrain_means4 = features_sj['reanalysis_relative_humidity_percent'].rolling(window = 2).mean()[60:675]

# exponentially weighted means
Xtrain_xmeans1 = features_sj['station_avg_temp_c'].ewm(span = 53).mean()[60:675]

# rolling stds df
Xtrain_std1 = features_sj['reanalysis_avg_temp_k'].rolling(window = 2).std()[60:675]

# combine all dfs
Xtrain = pd.concat([Xtrain_means1, Xtrain_means2, Xtrain_means3, Xtrain_means4, Xtrain_std1], axis = 1)
ytrain = sj_residuals_train[60:]

print(len(Xtrain), len(ytrain))

# rolling means df
Xvalid_means1 = features_sj['station_avg_temp_c'].rolling(window = 53).mean()[675:936]
Xvalid_means2 = features_sj['ndvi_ne'].rolling(window = 53).mean()[675:936]
Xvalid_means3 = features_sj['ndvi_sw'].rolling(window = 2).mean()[675:936]
Xvalid_means4 = features_sj['reanalysis_relative_humidity_percent'].rolling(window = 2).mean()[675:936]

# exponentially weighted means
Xvalid_xmeans1 = features_sj['station_avg_temp_c'].ewm(span = 53).mean()[675:936]

# rolling stds df
Xvalid_std1 = features_sj['reanalysis_avg_temp_k'].rolling(window = 2).std()[675:936]

# combine all dfs
Xvalid = pd.concat([Xvalid_means1, Xvalid_means2, Xvalid_means3, Xvalid_means4, Xvalid_std1], axis = 1)[60:]
yvalid = sj_residuals_test[60:]

print(len(Xvalid), len(yvalid))

# fit on model, predict
lr_sj_residual = LinearRegression()
lr_sj_residual.fit(Xtrain, ytrain)

sj_valid_preds = lr_sj_residual.predict(Xvalid)

print(mean_absolute_error(sj_valid_preds, yvalid))
print(lr_sj_residual.score(Xvalid, yvalid))
plt.plot(yvalid.values)
plt.plot(sj_valid_preds)
#plt.show()

#Varun check this part till commented code ends
#Test grid searched rollback terms on actual cases
#Add the monthly trend and the predicted residuals to get predicted cases
# preds_features = pd.DataFrame({'pred_resid':sj_valid_preds,
#                          'real_resid':yvalid,
#                          'monthly_trend':sj_predicts[60:],
#                          'actual_cases':ytest[60:]})
# preds_features['pred_cases'] = preds_features['monthly_trend'] + preds_features['pred_resid']
# preds_features['pred_cases'] = preds_features['pred_cases'].apply(lambda x: int(x)).apply(lambda x: 0 if x < 0 else x)
# print ('predicted and actual MAE:')
# print (mean_absolute_error(preds_features['pred_cases'], preds_features['actual_cases']))
# print('monthly trend and actual MAE:')
# print (mean_absolute_error(preds_features['monthly_trend'], preds_features['actual_cases']))
# preds_features[['monthly_trend', 'actual_cases', 'pred_cases']].plot(figsize=(10,6))




#IQ
#Do the same as I did for SJ above, but to see which IQ features explain the variance in the residuals.
print("Iquitos prediction starts from here")
print(len(pd.get_dummies(iq_Xtrain['month'], prefix='month')), len(iq_ytrain.values))
print(len(pd.get_dummies(iq_Xvalid['month'], prefix='month')), len(iq_yvalid.values))


# iq monthly trend

lr_iq = LinearRegression()
X = pd.get_dummies(iq_Xtrain['month'], prefix='month')
y = iq_ytrain.values

lr_iq.fit(X, y)
monthly_trend_train = pd.Series(lr_iq.predict(X)).rolling(9, min_periods = 1).mean()
iq_residuals_train = y - monthly_trend_train


# on validation data
# note: monthly trend does not need previous weeks data, so this can use the validation set
Xtest = pd.get_dummies(iq_Xvalid['month'], prefix='month')
ytest = iq_yvalid.values
monthly_trend_valid = pd.Series(lr_iq.predict(Xtest)).rolling(9, min_periods=1).mean()
iq_residuals_test = ytest - monthly_trend_valid

# plot
plt.plot(lr_iq.predict(Xtest))
plt.plot(monthly_trend_valid)
plt.plot(ytest)
plt.title('Monthly Trend of Dengue Cases in Iquitos, Peru')
plt.legend()
#plt.show()

print(mean_absolute_error(lr_iq.predict(Xtest), ytest))
print(mean_absolute_error(monthly_trend_valid, ytest))


# Find the residuals of the monthly trend

plt.figure(figsize=(10, 6))
plt.plot(iq_residuals_test, label = 'residuals', linewidth = 3)
plt.plot(ytest, 'g-*', alpha = .4, label = 'true values')
plt.plot(monthly_trend_valid, alpha = .4, label = 'monthly trend')
plt.title('True Values, Monthly Trend, and Residuals of Dengue Cases in Iquitos, Peru')
plt.legend()
#plt.show()
print(np.mean(iq_residuals_test))

## PREDICT THE BLUE!!!
# Note: the monthly predictions for iq are pretty bad over the validation set...




# See how far back we should roll in Iquitos weather
# Rolling Mean


# Plot sum of correlation between rolling mean of all features and residuals
mean_score = []
for i in range (2, 60):
    feature = iq_train.drop(['month', 'weekofyear'], axis = 1)
    feature = feature.rolling(i).mean()[i-1:]
    feature['resid'] = iq_residuals_train.values[i-1:]

    mean_score.append(abs(feature.corr()['resid']).sum()-1)
plt.plot(mean_score)
plt.title('sum of correlation between rolling mean of all features and residuals')
plt.legend()
print("Rolling Mean")
#plt.show()


# Plot sum of correlation between rolling mean of all features and residuals
# create feature of correlation between feature and resid at each rolling mean
features_scores = pd.DataFrame()
for i in range (2, 80):
    feature = iq_train.drop(['month', 'weekofyear', 'month'], axis = 1)
    feature = feature.rolling(i, min_periods=1).mean()
    feature['resid'] = iq_residuals_train.values
    features_corr = feature.corr()
    features_scores[str(i)] = features_corr['resid']

# create df of max absolute value of corr and the window
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
#print(scores_features)


plt.figure(figsize=(10,6))
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
plt.title('Correlation of Rolling Mean of Features to Residuals', fontsize = 15)
plt.xlabel('Number of Weeks back to take Rolling Mean', fontsize=10)
plt.ylabel('Strength of Correlation', fontsize=10)
plt.legend(loc=(1,0), fontsize= 4)
#plt.show()


#Rolling Standard Deviations
# Plot sum of correlation between rolling mean of all features and residuals
# create df of correlation between feature and resid at each rolling mean
df_scores = pd.DataFrame()
for i in range (2, 80):
    feature = iq_train.drop(['month', 'weekofyear', 'month'], axis = 1)
    feature = feature.rolling(i, min_periods=1).std()
    feature['resid'] = iq_residuals_train.values
    features_corr = feature.corr()
    features_scores[str(i)] = features_corr['resid']

# create df of max absolute value of corr and the window
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
#print(scores_features)

plt.figure(figsize=(10,6))
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
plt.title('Correlation of Rolling Mean of Features to Residuals', fontsize = 15)
plt.xlabel('Number of Weeks back to take Rolling Mean', fontsize=10)
plt.ylabel('Strength of Correlation', fontsize=10)
plt.legend(loc=(1,0), fontsize= 4)
#plt.show()

#predict the residuals based on rolling weather features
features_iq.columns
len(Xtrain)
len(ytrain)

# rolling mean of feature to predict residuals

# munge data

# max roll back set to 59

# set up training data
# rolling means df
Xtrain_means1 = features_iq['ndvi_se'].rolling(window = 45).mean()[60:364]
Xtrain_means2 = features_iq['reanalysis_min_air_temp_k'].rolling(window = 5).mean()[60:364]
Xtrain_means3 = features_iq['reanalysis_relative_humidity_percent'].rolling(window = 14).mean()[60:364]
Xtrain_means4 = features_iq['station_precip_mm'].rolling(window = 6).mean()[60:364]

# rolling stds df
Xtrain_std1 = features_iq['station_precip_mm'].rolling(window = 31).std()[60:364]

# weeks shifted back
Xtrain_shift1 = features_iq['station_min_temp_c'].shift(2)[60:364]
Xtrain_shift2 = features_iq['station_precip_mm'].shift(1)[60:364]


# combine all dfs
Xtrain = pd.concat([Xtrain_means1, Xtrain_std1], axis = 1)
ytrain = iq_residuals_train[60:]

print(len(Xtrain), len(ytrain))

# set up validation data
# rolling means df
Xvalid_means1 = features_iq['ndvi_se'].rolling(window = 45).mean()[364:520]
Xvalid_means2 = features_iq['reanalysis_min_air_temp_k'].rolling(window = 5).mean()[364:520]
Xvalid_means3 = features_iq['reanalysis_relative_humidity_percent'].rolling(window = 14).mean()[364:520]
Xvalid_means4 = features_iq['station_precip_mm'].rolling(window = 6).mean()[364:520]

# rolling stds df
Xvalid_std1 = features_iq['station_precip_mm'].rolling(window = 31).std()[364:520]

# weeks shifted back
Xvalid_shift1 = features_iq['station_min_temp_c'].shift(2)[364:520]
Xvalid_shift2 = features_iq['station_precip_mm'].shift(1)[364:520]

# combine all dfs
Xvalid = pd.concat([Xvalid_means1, Xvalid_std1], axis = 1)[60:]
yvalid = iq_residuals_test[60:]

print(len(Xvalid), len(yvalid))

# model it!

lr_iq_resids = LinearRegression()
lr_iq_resids.fit(Xtrain, ytrain)

iq_valid_preds = lr_iq_resids.predict(Xvalid)

# plot iq residual predictions

plt.plot(yvalid.values, alpha = .75)
plt.plot(iq_valid_preds)
plt.title("Iq residual prediction based on weather features", fontsize = 15)
plt.legend()
#plt.show()
#print(lr_iq_resids.score(Xvalid, yvalid))
#print(mean_absolute_error(iq_valid_preds, yvalid))

# USE TEST DATA, MAKE CSV OF SUBMISSIONS
# Need to use full dataset (train + test) because values depend on previous weather data

# transform sj data

month_dums = pd.get_dummies(features_sj['month'], prefix='month')
temp_roll_means = pd.DataFrame(features_sj[['station_avg_temp_c']].rolling(window = 55).mean())
# temp_roll_std = pd.DataFrame(features_sj[['station_avg_temp_c', 'precipitation_amt_mm']].rolling(window = 8).mean())

# combine into test set
Xtest = pd.concat([month_dums, temp_roll_means], axis = 1)

# train model
lr_sj_full = LinearRegression()
lr_sj_full.fit(Xtest[60:-260], cases_sj.values[60:])

# predicts
sj_full_preds = lr_sj_full.predict(Xtest[55:])
sj_submit_preds = sj_full_preds[-260:]


# transform iq data
month_dums = pd.get_dummies(features_iq['month'], prefix='month')
temp_roll_means = pd.DataFrame(features_iq[['ndvi_nw']].rolling(window = 68).mean())
# temp_roll_std = pd.DataFrame(df_iq[['station_avg_temp_c', 'precipitation_amt_mm']].rolling(window = 8).mean())

# combine into test
Xtest = pd.concat([month_dums, temp_roll_means], axis = 1)

# train
lr_iq_full = LinearRegression()
lr_iq_full.fit(Xtest[68:-156], cases_iq.values[68:])

# predicts
iq_full_preds = lr_iq_full.predict(Xtest[68:])
iq_submit_preds = iq_full_preds[-156:]

plt.plot(sj_submit_preds)
plt.title("SJ submit prediction")
#plt.show()
plt.plot(iq_submit_preds)
plt.title("IQ submit prediction")
#plt.show()

total_predictions = list(sj_submit_preds) + list(iq_submit_preds)

print(total_predictions)

#make CSV
# submission
submission_format = pd.read_csv('../data/submission_format.csv')

submission_format['total_cases'] = total_predictions
submission_format['total_cases'] = submission_format['total_cases'].apply(lambda x: int(x))

# Save to CSV, use current date
submission_format.to_csv('../data/dengue_submission_9_12_17v1.csv', index=False)

#
# Make predictions v2
# Predict the residuals, then add them to the monthly trend
# San Juan


# get monthly trend of whole test df
lr_sj_month = LinearRegression()
X_months = pd.get_dummies(features_sj['month'], prefix='month')[:936]
Xtest_months = pd.get_dummies(features_sj['month'], prefix='month')[936:]
y = cases_sj.values

lr_sj_month.fit(X_months, y)
monthly_trend = pd.Series(lr_sj_month.predict(X_months)).rolling(3, min_periods=1).mean()
sj_residuals_all = y - monthly_trend

# create test df of rolling weather stats
# rolling means df
Xtrain_means1 = features_sj['station_avg_temp_c'].rolling(window = 53).mean()[60:936]
Xtrain_means2 = features_sj['ndvi_ne'].rolling(window = 53).mean()[60:936]
Xtrain_means3 = features_sj['ndvi_sw'].rolling(window = 2).mean()[60:936]
Xtrain_means4 = features_sj['reanalysis_relative_humidity_percent'].rolling(window = 2).mean()[60:936]

# exponentially weighted means
Xtrain_xmeans1 = features_sj.ewm(span = 30).mean()[60:936]

# rolling stds df
Xtrain_std1 = features_sj['reanalysis_avg_temp_k'].rolling(window = 2).std()[60:936]

# combine all dfs
Xtrain = pd.concat([Xtrain_means1], axis = 1)
ytrain = sj_residuals_all[60:]


# create test df on rolling weather stats
# rolling means df
Xtest_means1 = features_sj['station_avg_temp_c'].rolling(window = 53).mean()[936:]
Xtest_means2 = features_sj['ndvi_ne'].rolling(window = 53).mean()[936:]
Xtest_means3 = features_sj['ndvi_sw'].rolling(window = 2).mean()[936:]
Xtest_means4 = features_sj['reanalysis_relative_humidity_percent'].rolling(window = 2).mean()[936:]

# exponentially weighted means
Xtest_xmeans1 = features_sj.ewm(span = 30).mean()[936:]

# rolling stds df
Xtest_std1 = features_sj['reanalysis_avg_temp_k'].rolling(window = 2).std()[936:]

# combine all dfs
Xtest_weather = pd.concat([Xtest_means1], axis = 1)

# fit on model
lr_sj_resid = LinearRegression()
lr_sj_resid.fit(Xtrain, ytrain)

# make predictions on monthly data and residual data
sj_monthly_predictions = pd.Series(lr_sj_month.predict(Xtest_months)).rolling(3, min_periods=1).mean()
sj_resid_predictions = lr_sj_resid.predict(Xtest_weather)
sj_cases_predictions = pd.Series(sj_resid_predictions + sj_monthly_predictions).rolling(1, min_periods=1).mean()
sj_cases_predictions = sj_cases_predictions.apply(lambda x: 1 if x < 1 else int(x))



# plt.plot(sj_resid_preds)
plt.plot(sj_monthly_predictions)
plt.plot(sj_cases_predictions)
plt.title("SJ Predictions")
#plt.show()

#Iquitos
print("Iquitos cases length: "+str(len(cases_iq)))

print(features_iq.ewm(span=5).mean().head())

print(features_iq.ewm(span=2).mean().head())



# get monthly trend of whole test df
lr_iq_month = LinearRegression()
X_months = pd.get_dummies(features_iq['month'], prefix='month')[:520]
Xtest_months = pd.get_dummies(features_iq['month'], prefix='month')[520:]
y = cases_iq.values

lr_iq_month.fit(X_months, y)
monthly_trend = pd.Series(lr_iq_month.predict(X_months)).rolling(8, min_periods=1).mean()
iq_residuals_all = y - monthly_trend

# create test df of rolling weather stats
# rolling means df
Xtrain_means1 = features_iq['ndvi_nw'].rolling(window = 53).mean()[60:520]
Xtrain_means2 = features_iq['station_avg_temp_c'].rolling(window = 53).mean()[60:520]
Xtrain_means3 = features_iq['reanalysis_relative_humidity_percent'].rolling(window = 14).mean()[60:520]
Xtrain_means4 = features_iq['station_precip_mm'].rolling(window = 6).mean()[60:520]

# rolling stds features
Xtrain_std1 = features_iq['station_precip_mm'].rolling(window = 31).std()[60:520]

# weeks shifted back
Xtrain_shift1 = features_iq['station_min_temp_c'].shift(2)[60:520]
Xtrain_shift2 = features_iq['station_precip_mm'].shift(4)[60:520]


# combine all features
Xtrain = pd.concat([Xtrain_means2], axis = 1)
ytrain = iq_residuals_all[60:]


# create test df on rolling weather stats
# rolling means features
Xtest_means1 = features_iq['ndvi_nw'].rolling(window = 53).mean()[520:]
Xtest_means2 = features_iq['station_avg_temp_c'].rolling(window = 53).mean()[520:]
Xtest_means3 = features_iq['reanalysis_relative_humidity_percent'].rolling(window = 14).mean()[520:]
Xtest_means4 = features_iq['station_precip_mm'].rolling(window = 6).mean()[520:]

# rolling stds features
Xtest_std1 = features_iq['station_precip_mm'].rolling(window = 31).std()[520:]

# weeks shifted back
Xtest_shift1 = features_iq['station_min_temp_c'].shift(2)[520:]
Xtest_shift2 = features_iq['station_precip_mm'].shift(4)[520:]


# combine all features
Xtest_weather = pd.concat([Xtest_means2], axis = 1)

# fit on model
lr_iq_resid = LinearRegression()
lr_iq_resid.fit(Xtrain, ytrain)

# make predictions on monthly data and residual data
iq_monthly_predictions = pd.Series(lr_iq_month.predict(Xtest_months)).rolling(8, min_periods=1).mean()
iq_resid_predictions = lr_iq_resid.predict(Xtest_weather)
iq_cases_predictions = pd.Series(iq_monthly_predictions + iq_resid_predictions)
iq_cases_predictions = iq_cases_predictions.apply(lambda x: 0 if x < 1 else int(x))



plt.plot(iq_resid_predictions)
plt.plot(iq_monthly_predictions)
plt.plot(iq_cases_predictions)
plt.title("Iquitos prediction")
plt.show()

#make CSV
total_predictions = list(sj_cases_predictions) + list(iq_cases_predictions)


# submission
submission_format = pd.read_csv('../data/submission_format.csv')

submission_format['total_cases'] = total_predictions
submission_format['total_cases'] = submission_format['total_cases'].apply(lambda x: int(x))

# Save to CSV, use current date
submission_format.to_csv('../data/dengue_submission_11_28_17v1.csv', index=False)

print(submission_format.head())