import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

features = pd.read_csv('../data/dengue_features_train.csv', parse_dates=[3])
# print(features)

features_iq = features[features['city'] == 'iq']
features_sj = features[features['city'] == 'sj']

# print(features_sj)
# print(features_iq)

fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(features_iq.isnull().reset_index(drop=True),ax=ax, cbar = False, yticklabels = 50)

plt.ylabel("Row number", size = 22)
plt.xlabel("Feature name", size = 22)
plt.title("Iquitos Missing Data", size = 32)
# plt.show()

fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(features_sj.isnull(),ax=ax, cbar = False, yticklabels = 50)

plt.ylabel("Row number", size = 22)
plt.xlabel("Feature name", size = 22)
plt.title("San Juan Missing Data", size = 32)
# plt.show()

features_iq_mean = features_iq.mean()
features_sj_mean = features_sj.mean()

# Cleaning the data
features_sj = features_sj.fillna(features_sj_mean)
features_iq = features_iq.fillna(features_iq_mean)

# print(features_sj)
# print(features_iq)

# drop non-numerical values
features_sj.drop(['city', 'year'], axis = 1, inplace = True)
features_iq.drop(['city', 'year'], axis = 1, inplace = True)

# Saving the cleaned data
features_sj.to_csv("../data/features_sj.csv")
features_iq.to_csv("../data/features_iq.csv")

