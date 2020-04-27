# -*- coding: utf-8 -*-

"""
@Time   : 2020-04-26 14:54
@Author : Kyrie.Z
@File   : config.py
"""

import os


class Configuration(object):
    def __init__(self, base_path, suffix):
        self.data_prefix = os.path.join(base_path, 'data', suffix)
        self.cache_prefix = os.path.join(base_path, 'cache', suffix)

        self.edge_file = os.path.join(self.data_prefix, 'edges.txt')
        self.edge_cache = os.path.join(self.cache_prefix, 'edges.pkl')
        self.edge_label_file = os.path.join(self.data_prefix, 'edge_labels.txt')
        self.edge_label_cache = os.path.join(self.cache_prefix, 'edge_labels.pkl')
        self.mlc_origin_data = os.path.join(self.data_prefix, 'validation_set.tsv')
        self.flipkart_origin_data = os.path.join(self.data_prefix, 'flipkart_com-ecommerce_sample.csv')
        self.mlc_data_file = os.path.join(self.data_prefix, 'mlc_dataset.tsv')
        self.flipkart_data_file = os.path.join(self.data_prefix, 'fk_dataset.csv')


if __name__ == '__main__':
    config = Configuration('../../', suffix='ebay-mlc')
    print(config.edge_file)
    assert config.edge_file == '../../data/ebay-mlc/edges.txt'
