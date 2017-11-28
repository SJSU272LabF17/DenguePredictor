import pandas as pd
# read in features, parse the date column to date object

features = pd.read_csv('../data/dengue_features_train.csv', parse_dates=[3])
target = pd.read_csv('../data/dengue_labels_train.csv')

# add total cases into df
features['total_cases'] = target['total_cases']

# create new datasets

features_sj = features[features['city'] == 'sj']
features_iq = features[features['city'] == 'iq']

# ## FrontFill to Impute into Missing values
#
# features_sj = features_sj.fillna(method = 'ffill')
# features_iq = features_iq.fillna(method = 'ffill')


features_iq_mean = features_iq.mean()
features_sj_mean = features_sj.mean()

# Cleaning the data
features_sj = features_sj.fillna(features_sj_mean)
features_iq = features_iq.fillna(features_iq_mean)


# set index to the dates
features_sj.set_index('week_start_date', drop = True, inplace = True)
features_iq.set_index('week_start_date', drop = True, inplace = True)

#drop non-numerical values
features_sj.drop(['city', 'year'], axis = 1, inplace = True)
features_iq.drop(['city', 'year'], axis = 1, inplace = True)

#print(features_iq)


# save to pickle file
features_iq.to_pickle('../data/features_iq.pkl')
features_sj.to_pickle('../data/features_sj.pkl')



#All training features set in one features

train = pd.read_csv('../data/dengue_features_train.csv', parse_dates=[3])
#print(train)

test = pd.read_csv('../data/dengue_features_test.csv', parse_dates=[3])
#, parse_dates=[3]

print(test)

full_features = pd.concat([train, test], axis = 0)
print(len(train))
print(len(test))
print(len(full_features))


all_sj = full_features[full_features['city'] == 'sj']
all_iq = full_features[full_features['city'] == 'iq']

#calculating mean values to fill in the missing data

all_iq_mean=all_iq.mean()
all_sj_mean=all_sj.mean()

all_iq=all_iq.fillna(all_iq_mean)
all_sj=all_sj.fillna(all_sj_mean)

#set index to date
all_sj.set_index('week_start_date', drop = True, inplace = True)
all_iq.set_index('week_start_date', drop = True, inplace = True)

#drop non-numerical values
all_sj.drop(['city', 'year'], axis = 1, inplace = True)
all_iq.drop(['city', 'year'], axis = 1, inplace = True)

# save to pickle file
all_iq.to_pickle('../data/all_iq.pkl')
all_sj.to_pickle('../data/all_sj.pkl')