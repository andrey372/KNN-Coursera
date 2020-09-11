# Import libraries
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from kneed import KneeLocator
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import load_digits
import scipy.cluster.hierarchy as shc
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
import seaborn as sns


#blobs

X, y  = make_blobs(n_samples=500, centers=[[5, 5], [0, 0], [1, 5],[5, -1]], cluster_std=0.9)

#dendogram
#dendogram = shc.dendrogram(shc.linkage(X, method='ward'))
#plt.show()

#model
X_scaled = StandardScaler().fit_transform(X)
db = DBSCAN(eps = 0.3, min_samples=10).fit(X_scaled)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True


print('twst')
