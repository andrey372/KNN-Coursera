# Import libraries
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs

X, y = make_blobs(n_samples=500, cluster_std=0.9,centers=4)
plt.scatter(X[:,0], X[:,1], marker = 'o')
plt.show()

#model
agglomCluster = AgglomerativeClustering(n_clusters=4, linkage='average')
agglomCluster.fit(X, y)