import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
plt.style.use('bmh')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_error


# read in csv
features_iq = pd.read_pickle('../data/features_iq.pkl')
features_sj = pd.read_pickle('../data/features_sj.pkl')
#print(features_sj)

# create month variable in the data
features_sj['month'] = features_sj.index.month
features_iq['month'] = features_iq.index.month

# train test split
#sj
sj_train = features_sj .loc[:'2003-04-20']
# print(sj_train)
sj_Xtrain = sj_train.drop('total_cases',axis = 1)
sj_ytrain = sj_train[['total_cases']]

sj_test = features_sj.loc['2003-04-20':]
sj_Xtest = sj_test.drop('total_cases', axis = 1)
sj_ytest = sj_test[['total_cases']]

#iq
iq_train = features_iq.loc[:'2006-06-20']
iq_Xtrain = iq_train.drop('total_cases', axis = 1)
iq_ytrain = iq_train[['total_cases']]

iq_test = features_iq.loc['2006-06-20':]
iq_Xtest = iq_test.drop('total_cases', axis = 1)
iq_ytest = iq_test[['total_cases']]






# Set up test data to fit into model

# read in test data
test = pd.read_csv('../data/dengue_features_test.csv', parse_dates=[3])
submission_format = pd.read_csv('../data/submission_format.csv')

#split into cities features
test_sj = test[test['city'] == 'sj']
test_iq = test[test['city'] == 'iq']

# print(test_iq)
#
#calculate mean of the both cities data
test_iq_mean=test_iq.mean()
test_sj_mean=test_sj.mean()

#  Missing values filled by mean value
test_sj = test_sj.fillna(test_sj_mean)
test_iq = test_iq.fillna(test_iq_mean)
#
# set index to the dates
test_sj.set_index('week_start_date', drop = True, inplace = True)
test_iq.set_index('week_start_date', drop = True, inplace = True)




# drop non-numerical values
test_sj.drop(['city', 'year', 'weekofyear'], axis = 1, inplace = True)
test_iq.drop(['city', 'year', 'weekofyear'], axis = 1, inplace = True)

#print(test_sj)

# create month variable
test_sj['month'] = test_sj.index.month
test_iq['month'] = test_iq.index.month

#print(test_sj)



plt.figure(figsize=(20,8))
plt.plot(features_sj['total_cases'])
plt.title("San Juan monthly prediction")
plt.xlabel('Date', fontsize = 24)
plt.ylabel('Number of Cases', fontsize = 24)
#plt.show()

plt.figure(figsize=(20,8))
plt.plot(features_iq['total_cases'])
plt.title("Iquitos monthly prediction")
plt.xlabel('Date', fontsize = 24)
plt.ylabel('Number of Cases', fontsize = 24)
#plt.show()


# sj month by month only
lr_sj_months = LinearRegression()
month_dums = pd.get_dummies(features_sj['month'], prefix = 'month')
X = month_dums
y = features_sj['total_cases']
lr_sj_months.fit(X, y)
#print(lr_sj_months)


#predict test values
Xtest = pd.get_dummies(features_sj['month'], prefix='month')
sj_predicts = lr_sj_months.predict(Xtest)
#print (mean_absolute_error(sj_predicts, y))

# plot predictions
plt.figure(figsize=(20, 8))
plt.plot(features_sj['total_cases'], alpha = .75, label = 'Actual Values')
plt.plot(features_sj['total_cases'].index, sj_predicts, linewidth = 5, label = 'Predicted Values')
plt.legend(fontsize = 24)
plt.title('San Juan: Monthly Trend of Dengue Fever Cases')
plt.ylabel('Number of Cases', fontsize = 24)
plt.xlabel('Date', fontsize = 24)
#plt.show()


# plot with rolling mean of monthly trend
plt.figure(figsize=(20, 8))
plt.plot(features_sj['total_cases'], alpha = .75, label = 'Actual Values')
plt.plot(features_sj['total_cases'].index, sj_predicts, linewidth = 5, label = 'Predicted Values')
plt.plot(features_sj['total_cases'].index, pd.DataFrame(sj_predicts).rolling(3, min_periods = 1).mean(),
         linewidth = 5, label = 'Rolling Predicted Values')
plt.legend(fontsize = 24)
plt.title('San Juan: Monthly Trend of Dengue Fever Cases,plot with rolling mean of monthly trend')
plt.ylabel('Number of Cases', fontsize = 24)
plt.xlabel('Date', fontsize = 24)
plt.show()

# iq month by month only
lr_iq_months = LinearRegression()
month_dums = pd.get_dummies(features_iq['month'], prefix = 'month')
X = month_dums
y = features_iq['total_cases']
lr_iq_months.fit(X, y)
# iq_predicts = lr.predict(X)
# plt.plot(iq_predicts)
# plt.plot(y)
# print mean_absolute_error(iq_predicts, y)

# #predict test values
Xtest = pd.get_dummies(features_iq['month'], prefix='month')
iq_predicts = lr_iq_months.predict(Xtest)

# plot predictions
plt.figure(figsize=(20, 8))
plt.plot(features_iq['total_cases'], alpha = .75, label = 'Actual Values')
plt.plot(features_iq['total_cases'].index, iq_predicts, linewidth = 4, label = 'Predicted Values')
plt.legend(fontsize = 24)
plt.title('Iquitos: Monthly Trend of Dengue Fever Cases', fontsize = 24)
plt.ylabel('Number of Cases', fontsize = 24)
plt.xlabel('Date', fontsize = 24)
plt.show()

# plot predictions
plt.figure(figsize=(16, 3))
plt.plot(features_iq['total_cases'], alpha = .75, label = 'Actual Values')
plt.plot(features_iq['total_cases'].index, iq_predicts, label = 'Predicted Values')
plt.plot(features_iq['total_cases'].index, pd.DataFrame(iq_predicts).rolling(3, min_periods = 1).mean(),
         linewidth = 2, label = 'Rolling Predicted Values')
plt.legend()
plt.title('Iquitos: Monthly Trend of Dengue Fever Cases')
plt.ylabel('Number of Cases')
plt.xlabel('Date')
plt.show()
# submission
submission_format = pd.read_csv('../data/submission_format.csv')
submission_format.head()
#print(submission_format)


# predict, then take rolling mean (window 3), then round to integer

# san juan
sj_test_predicts = pd.Series(lr_sj_months.predict(pd.get_dummies(test_sj['month'])))
# change, or turn this off, to stop the rolling mean
sj_test_predicts = sj_test_predicts.rolling(3, min_periods=1).mean()
sj_test_predicts = sj_test_predicts.apply(lambda x: int(x))
#print(sj_test_predicts)

# iquitos
iq_test_predicts = pd.Series(lr_iq_months.predict(pd.get_dummies(test_iq['month'])))
# change, or turn this off, to stop the rolling mean
iq_test_predicts = iq_test_predicts.rolling(3, min_periods=1).mean()
iq_test_predicts = iq_test_predicts.apply(lambda x: int(x))
#print(iq_test_predicts)


total_preds = pd.concat([sj_test_predicts, iq_test_predicts], axis = 0)
total_preds.reset_index(drop = True, inplace=True)


print(len(total_preds))
print(len(submission_format['total_cases']))

# monthly_preds = sj_predicts.append(iq_predicts).reset_index(drop = True)


submission_format['total_cases'] = total_preds
print(submission_format)
print("completed")









