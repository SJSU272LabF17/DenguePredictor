import pandas as pd
import numpy as np

features = pd.read_csv('/home/varunu28/Desktop/predicting-dengue/datasets/dengue_features_train.csv')
# print(features)

labels = pd.read_csv('/home/varunu28/Desktop/predicting-dengue/datasets/dengue_labels_train.csv')
# print(labels)

features_iq = features[features['city'] == 'iq']
features_sj = features[features['city'] == 'sj']

# print(features_sj)
# print(features_iq)

features_iq_mean = features_iq.mean()
features_sj_mean = features_sj.mean()

# Cleaning the data
features_sj = features_sj.fillna(features_sj_mean)
features_iq = features_iq.fillna(features_iq_mean)

# print(features_sj)
# print(features_iq)

# Saving the cleaned data
features_sj.to_csv("/home/varunu28/PycharmProjects/272Project/testdata/features_sj.csv")
features_iq.to_csv("/home/varunu28/PycharmProjects/272Project/testdata/features_iq.csv")

