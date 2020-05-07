# -*- coding: utf-8 -*-

"""
@Time   : 2020-05-04 16:35
@Author : Kyrie.Z
@File   : quantizer.py
"""

from tqdm import tqdm
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


class Quantizer(object):
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.code = 'hashcode'
        self.clfs = {}

    @staticmethod
    def series_reshape(se: pd.Series):
        return se.values.reshape(len(se), 1)

    def fit(self, dataframe):
        for name in tqdm(dataframe.columns):
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
            input_se = self.series_reshape(dataframe[name])
            kmeans.fit(input_se)
            self.clfs[name] = kmeans
        return self.clfs

    def transform(self, dataframe):
        for col in dataframe.columns:
            dataframe[col] = self.clfs[col].predict(self.series_reshape(dataframe[col])).astype(str)
        dataframe[self.code] = dataframe.apply(''.join, axis=1)

        return dataframe


if __name__ == '__main__':
    nm = ['a', 'b', 'c', 'd']

    df = pd.DataFrame(
        np.random.randint(0, 10, size=(10, 4)),
        columns=nm
    )
    df['e'] = df['d']

    print(df)

    vq = Quantizer(n_clusters=2)
    vq.fit(df)
    res = vq.transform(df)

    print(res)
