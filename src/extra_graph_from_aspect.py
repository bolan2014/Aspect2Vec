# -*- coding: utf-8 -*-

"""
@Time   : 2020-04-26 16:26
@Author : Kyrie.Z
@File   : extra_graph_from_aspect.py
"""

import pickle
from tqdm import tqdm
from collections import Counter
from sklearn import preprocessing

from src.utils.config import Configuration


class GraphExtractor:
    def __init__(self, cf: Configuration, delimiter: str = '|', dump_file: bool = False):
        self.cf = cf
        self.logging = cf.logging
        self.delimiter = delimiter
        self.dump_file = dump_file
        self.item_aspect = self.get_aspect_from_file()
        self.aspects = self.flatten()
        self.aspect_id_mapping, self.id_aspect_mapping = self.build_encoder()
        self.edges = self.make_edges()

    def get_aspect_from_file(self):
        import pandas as pd
        df = pd.read_csv(self.cf.cleaned_data_file, sep='\t')

        # TODO: Do some filtering and constraint about the dataframe, and get their aspects
        aspect_series = df['aspects']
        self.logging.info('get item aspects from file: {}'.format(self.cf.cleaned_data_file))
        item_aspect = []
        for aspect_line in tqdm(aspect_series):
            # TODO: consider whether need (val -> key) map or not
            item_aspect.append(
                aspect_line.split(self.delimiter)
            )
        self.logging.info('item size: {}'.format(len(item_aspect)))
        return item_aspect

    def flatten(self):
        total_aspects = [aspect for aspects in self.item_aspect for aspect in aspects]
        # example: ['inseam: 28', 'size type: regular', 'rise: mid-rise', 'wash: colored']
        self.logging.info('total aspect size: {}'.format(len(total_aspects)))
        return total_aspects

    def build_encoder(self):
        """
        :return: two mappings
        """
        encoder = preprocessing.LabelEncoder()
        self.logging.info('build node encoder ...')
        encoder.fit(self.aspects)

        classes, ids = encoder.classes_, encoder.transform(encoder.classes_)
        id_aspect_mapping = dict(zip(ids, classes))
        aspect_id_mapping = dict(zip(classes, ids))
        try:
            assert len(id_aspect_mapping) == len(aspect_id_mapping)
        except AssertionError:
            self.logging.debug('mapping size not equal: {} vs {}'.format(len(id_aspect_mapping), len(aspect_id_mapping)))
        self.logging.info('unique aspect size: {}'.format(len(id_aspect_mapping)))
        return aspect_id_mapping, id_aspect_mapping

    def make_pairs(self, aspects: list):
        """
        make aspect pairs with order
        :param aspects:
        :return: aspect pairs from the same item
        """
        pairs = []
        for i in range(len(aspects)):
            for j in range(i+1, len(aspects)):
                a_i, a_j = self.aspect_id_mapping[aspects[i]], self.aspect_id_mapping[aspects[j]]
                if aspects[i] < aspects[j]:
                    pairs.append((a_i, a_j))
                else:
                    pairs.append((a_j, a_i))
        return pairs

    def make_edges(self):
        """
        the frequency is denoted as edge weight
        :return: tuple of numbers
        """
        edges = []
        for aspects in tqdm(self.item_aspect):
            if len(aspects) > 1:
                edges.extend(self.make_pairs(aspects=aspects))
        return edges

    def dump_to_file(self):
        self.logging.info('total edges size: {}'.format(len(self.edges)))
        edges_with_weight = dict(Counter(self.edges))
        self.logging.info('unique edges size: {}'.format(len(edges_with_weight)))

        if self.dump_file is True:
            self.logging.info('dump to file ...')

            # dump total edges
            with open(self.cf.edge_file, 'w') as handler:
                for edge in self.edges:
                    handler.write('{} {}\n'.format(edge[0], edge[1]))

            # dump unique edges with weight
            with open(self.cf.weighted_edge_file, 'w') as handler:
                for edge, weight in edges_with_weight.items():
                    handler.write('{} {} {}\n'.format(edge[0], edge[1], weight))

            # dump aspect to id mapping
            with open(self.cf.aspect_to_id_cache, 'wb') as handler:
                pickle.dump(self.aspect_id_mapping, handler, protocol=pickle.HIGHEST_PROTOCOL)

            # dump id to aspect mapping
            with open(self.cf.id_to_aspect_cache, 'wb') as handler:
                pickle.dump(self.id_aspect_mapping, handler, protocol=pickle.HIGHEST_PROTOCOL)

            self.logging.info('Done')


if __name__ == '__main__':
    config = Configuration('../', suffix='ebay-mlc', file_name='validation_set.tsv')

    kg_extractor = GraphExtractor(cf=config, dump_file=True)
    kg_extractor.dump_to_file()
