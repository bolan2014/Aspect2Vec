# -*- coding: utf-8 -*-

"""
@Time   : 2020-04-26 16:26
@Author : Kyrie.Z
@File   : extra_graph_from_aspect.py
"""

from tqdm import tqdm
from collections import Counter
from sklearn import preprocessing

from src.utils.config import Configuration


class GraphBuilder:
    def __init__(self, cf: Configuration, delimiter: str='|'):
        self.cf = cf
        self.delimiter = delimiter
        self.item_aspect = self.get_aspect_from_file()
        self.aspects = self.flatten()
        self.encoder = self.build_encoder()
        # self.counter = Counter()

    def get_aspect_from_file(self):
        import pandas as pd
        df = pd.read_csv(self.cf.cleaned_data_file, sep='\t')

        # TODO: Do some filtering and constraint about the dataframe, and get their aspects
        aspect_series = df['aspects']
        self.cf.logging.info('get item aspects from file')
        item_aspect = []
        for aspect_line in tqdm(aspect_series):
            # TODO: consider whether need (val -> key) map or not
            item_aspect.append(
                aspect_line.split(self.delimiter)
            )
        self.cf.logging.info('item size: {}'.format(len(item_aspect)))
        return item_aspect

    def flatten(self):
        total_aspects = [aspect for aspects in self.item_aspect for aspect in aspects]
        # example: ['inseam: 28', 'size type: regular', 'rise: mid-rise', 'wash: colored']
        self.cf.logging.info('total aspect size: {}'.format(len(total_aspects)))
        return total_aspects

    def build_encoder(self):
        encoder = preprocessing.LabelEncoder()
        self.cf.logging.info('build node encoder ...')
        encoder.fit(self.aspects)
        return encoder

    def make_edges(self):
        # the frequency is denoted as edge weight
        pass


if __name__ == '__main__':
    config = Configuration('../', suffix='ebay-mlc', file_name='validation_set.tsv')

    kg_builder = GraphBuilder(cf=config)

    # cities = ["paris", "paris", "tokyo", "amsterdam"]
    # node_encoder = NodeEncoder(cities)
    # print(node_encoder.encoder.transform(cities))
