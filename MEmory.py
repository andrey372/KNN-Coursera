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

#DBSCSN CLUSTERING
        # Load data in X
        db = DBSCAN(eps=0.3, min_samples=10).fit(X)

        # Identifying which points make up our “core points”
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels))
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = 'k'

            class_member_mask = (labels == k)

            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                     markeredgecolor='k',
                     markersize=6)

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                     markeredgecolor='k',
                     markersize=6)

        plt.title('number of clusters: %d' % n_clusters_)
        plt.show()


#pd
df.apply(pd.to_numeric, errors='coerce')

#sns labeleb scattering
facet = sns.lmplot(data=df, x='X', y='y', hue='clusters',
                   fit_reg=False)
