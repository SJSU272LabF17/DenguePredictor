import matplotlib.pyplot as plt
import pandas as pd

target = pd.read_csv('../data/dengue_labels_train.csv')
#print(target)
feature_iq = pd.read_csv('../data/features_iq.csv')
#print(feature_iq)

feature_sj = pd.read_csv('../data/features_sj.csv')
#print(feature_sj)

features = target.set_index(['year', 'weekofyear'])

for i in ['iq', 'sj']:
    data = features[features['city'] == i]
    data.plot(figsize = (15,5))
    plt.title(str(i))
    plt.xlabel("Year, Week of Year")
    plt.ylabel("Number of Cases")

plt.show()