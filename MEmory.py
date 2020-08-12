#Generate the synthetic data and labels
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples = , centers = , cluster_std = )

#Elbow method / KneeLocator
from kneed import KneeLocator
sse = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k , init='k-means++', n_init=8).fit(scaled_X)
    sse.append(kmeans.inertia_)

kl = KneeLocator(range(1,11), sse, curve = 'convex', direction='decreasing')
kl.elbow

#Modelimg K-Means
from sklearn.cluster import KMeans
kmeans = KMeans(init='k-means', n_clusters= , n_init= ).fit(X)
kmeans.labels_


#pd
df.apply(pd.to_numeric, errors='coerce')