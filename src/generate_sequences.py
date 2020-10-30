# -*- coding: utf-8 -*-

"""
@Time   : 2020-04-10 16:43
@Author : Kyrie.Z
@File   : generate_sequences.py
"""

from src.utils.config import Configuration
from src.graph_embedding.initializer import GraphInitializer
from src.graph_embedding.dedicated_walker import Walker

config = Configuration(base_path='../', suffix='flipkart', file_name='flipkart_com-ecommerce_sample.csv')

init = GraphInitializer(config=config)

# print(init.node_labels)

walker = Walker(
    G=init.graph,
    config=config,
    node_labels=init.node_labels,
    seq_length= 8,
    alpha=1.,
    beta=1.
)

seqs = walker.traverse()
print(len(seqs))
print(seqs[:10])
for i in seqs[:10]:
    print('|'.join(map(lambda x: init.node_index[x], i)))
