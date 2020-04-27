# -*- coding: utf-8 -*-

"""
@Time   : 2020-04-26 14:54
@Author : Kyrie.Z
@File   : config.py
"""

import os
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class Configuration(object):
    def __init__(self, base_path, suffix, file_name=None):
        self.logging = logging
        self.data_prefix = os.path.join(base_path, 'data', suffix)
        self.cache_prefix = os.path.join(base_path, 'cache', suffix)

        self.edge_file = os.path.join(self.data_prefix, 'edges.txt')
        self.edge_cache = os.path.join(self.cache_prefix, 'edges.pkl')
        self.edge_label_file = os.path.join(self.data_prefix, 'edge_labels.txt')
        self.edge_label_cache = os.path.join(self.cache_prefix, 'edge_labels.pkl')
        self.origin_file = os.path.join(self.data_prefix, file_name)
        self.cleaned_data_file = os.path.join(self.data_prefix, '{}_dataset.csv'.format(suffix))


if __name__ == '__main__':
    config = Configuration('../../', suffix='ebay-mlc', file_name='validation_set.tsv')
    config.logging.info(config.edge_file)
    assert config.edge_file == '../../data/ebay-mlc/edges.txt'
