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
    def __init__(self, base_path: str, suffix: str, file_name='', embed_dim=100, max_len=30, cols=None, vec_prefix='aspect_vector'):
        self.logging = logging
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.cols = cols
        self.vec_prefix = vec_prefix
        self.data_prefix = os.path.join(base_path, 'data', suffix)
        self.cache_prefix = os.path.join(base_path, 'cache', suffix)

        self.edge_file = os.path.join(self.data_prefix, 'edges.txt')
        self.weighted_edge_file = os.path.join(self.data_prefix, 'weighted_edges.txt')
        self.id_to_aspect_cache = os.path.join(self.cache_prefix, 'id_to_aspect.pkl')
        self.aspect_to_id_cache = os.path.join(self.cache_prefix, 'aspect_to_id.pkl')
        self.aspect_by_item_cache = os.path.join(self.cache_prefix, 'aspect_by_item.pkl')  # list of lists
        self.aspect_sequence_file = os.path.join(self.data_prefix, 'aspect_corpus.txt')
        self.aspect_id_corpus = os.path.join(self.data_prefix, 'aspect_id_corpus.txt')
        self.aspect_vector_file = os.path.join(self.data_prefix, f'{self.vec_prefix}.vec')
        self.aspect_vector_cache = os.path.join(self.cache_prefix, f'{self.vec_prefix}.pkl')
        self.tokenizer_cache = os.path.join(self.cache_prefix, 'tokenizer.pkl')
        self.id_tokenizer_cache = os.path.join(self.cache_prefix, 'id_tokenizer.pkl')

        self.graph_cache = os.path.join(self.cache_prefix, 'graph.pkl')
        self.edge_label_file = os.path.join(self.data_prefix, 'edge_labels.txt')
        self.edge_label_cache = os.path.join(self.cache_prefix, 'edge_labels.pkl')
        self.origin_file = os.path.join(self.data_prefix, file_name)
        self.cleaned_data_file = os.path.join(self.data_prefix, '{}_dataset.tsv'.format(suffix))
        self.pairset_data_file = os.path.join(self.data_prefix, '{}_pairset.tsv'.format(suffix))

        self.train_file = os.path.join(self.data_prefix, 'train.tsv')
        self.valid_file = os.path.join(self.data_prefix, 'valid.tsv')
        self.train_id_file = os.path.join(self.data_prefix, 'train_id.tsv')
        self.valid_id_file = os.path.join(self.data_prefix, 'valid_id.tsv')
        self.model_file = os.path.join(self.cache_prefix, f'{self.vec_prefix}_siamese_lstm.bin')


class Columns:
    def __init__(self):
        self.uniq_id_1 = 'uniq_id_1'
        self.uniq_id_2 = 'uniq_id_2'
        self.aspects_1 = 'aspects_1'
        self.aspects_2 = 'aspects_2'
        self.category = 'category'
        self.label = 'label'

        self.all = [self.uniq_id_1, self.uniq_id_2, self.aspects_1, self.aspects_2, self.label]


if __name__ == '__main__':
    config = Configuration('../../', suffix='ebay-mlc', file_name='validation_set.tsv')
    config.logging.info(config.edge_file)
    assert config.edge_file == '../../data/ebay-mlc/edges.txt'
