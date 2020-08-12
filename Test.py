# Import libraries
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering


#blobs

X, y  = make_blobs(n_samples=500, centers=5, cluster_std=0.9)

#dendogram
dendogram = shc.dendrogram(shc.linkage(X, method='ward'))
plt.show()

#model
cluster = AgglomerativeClustering(n_clusters=4, linkage='ward').fit(X)
colors = cluster.labels_

plt.scatter(X[:,0], X[:,1], c = X[:,0])
plt.show()

agglomerativew
agglomerative
agglomerative