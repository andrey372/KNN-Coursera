# Import libraries
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from kneed import KneeLocator

df_origin = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/Cust_Segmentation.csv')
df = df_origin.drop(['Address'], axis=1)

X = df.values[:, 1:]
X = np.nan_to_num(X)
scaled_X = StandardScaler().fit_transform(X)

#kneeLocator

sse = []
for k in range(1,11):
    kmeans2 = KMeans(n_clusters=k , init='k-means++', n_init=8).fit(scaled_X)
    sse.append(kmeans2.inertia_)

kl = KneeLocator(range(1,11), sse, curve = 'convex', direction='decreasing')


print(kl.elbow)

#modeling
kmeans = KMeans(init='k-means++', n_clusters=4, n_init=10).fit(scaled_X)

df['Cluster_labels'] = kmeans.labels_
print(df.head())