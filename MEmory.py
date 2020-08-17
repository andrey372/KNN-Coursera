#Generate the synthetic data and labels
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples = , centers = , cluster_std = )


#KMEANS MODELING
        #Elbow method / KneeLocator
        from kneed import KneeLocator
        sse = []
        for k in range(1,11):
            kmeans = KMeans(n_clusters=k , init='k-means++', n_init=8).fit(scaled_X)
            sse.append(kmeans.inertia_)

        kl = KneeLocator(range(1,11), sse, curve = 'convex', direction='decreasing')
        kl.elbow

        #silhuete_coef
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=8),fit(X_scaled)
            score = silhouette_score(scaled_features, kmeans.labels_)
            ilhouette_coefficients.append(score)

        #Modelimg K-Means
        from sklearn.cluster import KMeans
        kmeans = KMeans(init='k-means', n_clusters= , n_init= ).fit(X)
        kmeans.labels_

#AGGLOMERATIVE CLUSTERING
        #dendogram
        dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
        plt.show()

        #Agglomerative model
        from sklearn.cluster import AgglomerativeClustering
        cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward').fit(X_scaled)


#pd
df.apply(pd.to_numeric, errors='coerce')

#sns labeleb scattering
facet = sns.lmplot(data=df, x='X', y='y', hue='clusters',
                   fit_reg=False)
