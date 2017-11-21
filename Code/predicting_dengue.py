import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot


feature_sj = pd.read_csv("../data/features_sj.csv")
feature_iq = pd.read_csv("../data/features_iq.csv")

target = pd.read_csv("../data/dengue_labels_train.csv")

feature_sj['total_cases'] = target['total_cases']
feature_iq['total_cases'] = target['total_cases']

# A summary of cases
'''
print(feature_sj.describe())
print(feature_iq.describe())
print(target.describe())
'''

# Plotting heatmaps for visualization
# The temprature variables correlate well in San Juan but not in Iquitos. Therefor we would go towards
# creating different models for these two cities as temprature is the main criteria we would consider
'''
sns.heatmap(feature_sj.corr())
sns.heatmap(feature_iq.corr())
pyplot.show()
'''

# Doing an ananlysis of how these cases change over a period of time

feature_sj.set_index('week_start_date', drop = True, inplace = True)
feature_iq.set_index('week_start_date', drop = True, inplace = True)

# Measuring NVDI(Normalized Difference Vegetation Index)
feature_sj['nvdi'] = feature_sj[feature_sj.columns[3:7]].mean(axis = 1)
feature_iq['nvdi'] = feature_iq[feature_iq.columns[3:7]].mean(axis = 1)

# A plot of week by week average for San Juan


# A plot for week by week cases for San Juan
fig, ax = pyplot.subplots(2, 1, figsize=(22, 12))

# Top plot(NVDI level)
pyplot.subplot(211)
for i in feature_sj.columns[3:7]:
    feature_sj[i].plot(alpha=.3)
feature_sj['nvdi'].plot(alpha=1, c='k', linewidth=1)
pyplot.title('NDVI Level in San Juan, Puerto Rico', size=25)
pyplot.xlabel('Year', size=20)
pyplot.ylabel('NDVI', size=20)

# Bottom plot (Mean)
pyplot.subplot(212)
for i in feature_iq.columns[3:7]:
    feature_sj.groupby('weekofyear')[i].mean().plot(alpha=.3)
feature_sj.groupby('weekofyear')['nvdi'].mean().plot(alpha=1, c='k', linewidth=5)
pyplot.title('Mean NDVI in San Juan, Puerto Rico', size=25)
pyplot.xlabel('Week of Year', size=20)
pyplot.ylabel('NDVI', size=20)
pyplot.legend(loc='best')

pyplot.tight_layout(pad=3)
# pyplot.show()

# feature_iq['year'] = feature_iq.index.year

# A plot for week by week cases for Iquitos
fig, ax = pyplot.subplots(2, 1, figsize=(22, 12))

# Top plot(NVDI level)
pyplot.subplot(211)
for i in feature_iq.columns[3:7]:
    feature_iq[i].plot(alpha = .3)
feature_iq['nvdi'].plot(alpha = 1, c = 'k', linewidth = 1)
pyplot.title('NDVI Level over Time in Iquitos, Peru', size = 25)
pyplot.xlabel('Year', size = 20)
pyplot.ylabel('NDVI', size = 20)

# Bottom plot (Mean)
pyplot.subplot(212)
for i in feature_iq.columns[3:7]:
    feature_iq.groupby('weekofyear')[i].mean().plot(alpha = .3)
feature_iq.groupby('weekofyear')['nvdi'].mean().plot(alpha = 1, c = 'k', linewidth = 5)
pyplot.title('Mean NDVI Level in Iquitos, Peru', size = 25)
pyplot.xlabel('Week of Year', size = 20)
pyplot.ylabel('NDVI', size = 20)
pyplot.legend(loc = 'best')


pyplot.tight_layout(pad=3)
pyplot.show()