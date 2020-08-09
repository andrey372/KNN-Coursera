# Import libraries
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
import wget
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/Cust_Segmentation.csv'
df_origin = pd.read_csv(url)
df = df_origin.drop('Address', axis = 1)

#preprocc
X = df.values[:,1:]
X = np.nan_to_num(X)
scaler = StandardScaler().fit_transform(X)

#model
kmeans = KMeans(init="k-means++", n_clusters=3, n_init=12)
kmeans.fit(X)
labels = kmeans.labels_

df['clusters'] = labels
print(df.groupby('clusters').mean())

plt.scatter(X[:,0], X[:,3], c = labels, alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()
