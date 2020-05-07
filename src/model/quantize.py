# -*- coding: utf-8 -*-


import numpy as np
from sklearn import cluster
from sklearn.decomposition import PCA


def quantizer(samples, n_cluster=2, slices=100, n_components=2):
    pca = PCA(n_components=n_components, svd_solver='full', random_state=42, whiten=False)
    pca.fit(samples)
    print "Compressed ratio: {}".format(sum(pca.explained_variance_ratio_))
    new_samples = pca.transform(samples)

    k_means = cluster.KMeans(n_clusters=n_cluster, n_init=4, random_state=42)
    k_means.fit(new_samples)
    centres = k_means.cluster_centers_.squeeze()
    print "Centres:"
    for centre in centres:
        print centre

    slot = (centres[1] - centres[0]) / slices
    # print slot

    print "Class count: {}".format(sum(k_means.labels_))

    return centres[0], slot


if __name__ == '__main__':

    data = [
        [1, 2, 3, 5],
        [2, 8, 7, 1.],
        [3, 5, 7, 9],
        [23, 4, 1, 0]
    ]

    base, slt = quantizer(data)

    # for i in data:
    #     print i > (base + slt * 50)