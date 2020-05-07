# -*- coding: utf-8 -*-

"""
@Time   : 2020-04-10 16:43
@Author : Kyrie.Z
@File   : generate_sequences.py
"""

from tqdm import tqdm
from src.utils.config import Configuration
from src.utils.initializer import GraphInitializer
from src.utils.dedicated_walker import Walker
from src.utils.aspect2vec_trainer import Trainer

# config = Configuration(base_path='../', suffix='ebay-mlc', file_name='validation_set.tsv')
config = Configuration(base_path='../', suffix='flipkart', file_name='flipkart_com-ecommerce_sample.csv')

init = GraphInitializer(cf=config)

# print(init.node_labels)

walker = Walker(
    G=init.graph,
    node_labels=init.node_labels,
    seq_labels=[2, 3, 4, 5, 6, 7, 8, 9],
    alpha=1.,
    beta=1.
)

# generate sequence for each start node
start_nodes = list(walker.G.nodes)

# for start_node in tqdm(start_nodes):
#     trvs = walker.traverse(start_node)
#     print(' '.join(map(lambda x: init.node_index[x], trvs)))


with open(config.aspect_sequence_file, 'w', encoding='utf-8') as handler:
    for aspects in init.aspect_by_item:
        handler.write('{}\n'.format(' '.join(aspects)))

trainer = Trainer(input_file=config.aspect_sequence_file, model='cbow')

trainer.dump_to_file(config.aspect_vector_file)

# avg.loss:  0.886780
