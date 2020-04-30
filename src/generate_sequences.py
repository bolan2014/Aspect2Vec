# -*- coding: utf-8 -*-

"""
@Time   : 2020-04-10 16:43
@Author : Kyrie.Z
@File   : generate_sequences.py
"""

from src.utils.config import Configuration
from src.utils.initializer import GraphInitializer
from src.utils.dedicated_walker import Walker

config = Configuration(base_path='../', suffix='ebay-mlc', file_name='validation_set.tsv')

init = GraphInitializer(cf=config)

# print(init.node_labels)

walker = Walker(
    G=init.graph,
    node_labels=init.node_labels,
    seq_labels=[2, 3, 4, 5, 6, 7, 8, 9],
    alpha=1.,
    beta=1.
)

start_node = 1

trvs = walker.traverse(start_node)
print(' '.join(map(lambda x: init.node_index[x], trvs)))
