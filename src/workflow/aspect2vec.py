# -*- coding: utf-8 -*-

"""
@Time ： 2020/10/30 1:49 PM
@Author ： Yuchen Guo
@File ： aspect2vec.py

"""

from src.utils.config import Configuration
from src.preprocess.data_cleaner import clean_and_dump_dataset
from src.preprocess.pairset_generator import PairsetGenerator
from src.graph_embedding.graph_extractor import GraphExtractor
from src.graph_embedding.initializer import GraphInitializer
from src.graph_embedding.dedicated_walker import Walker


if __name__ == '__main__':
    cf = Configuration('../../', suffix='flipkart', file_name='flipkart_com-ecommerce_sample.csv')
    # cf = Configuration('../../', suffix='ebay', file_name='human_labeled_set.tsv')

    df = clean_and_dump_dataset(config=cf, dump_flag=True)

    pair_generator = PairsetGenerator(df, category=["clothing", "jewellery", "footwear", "mobiles & accessories"],
                                      negative_ratio=5)
    pairset = pair_generator.generate_pairset(postive_restrict=None, output_file=cf.pairset_data_file)

    kg_extractor = GraphExtractor(config=cf, dump_file=True)
    kg_extractor.dump_to_file()

    init = GraphInitializer(config=cf)
    walker = Walker(
        G=init.graph,
        config=cf,
        node_labels=init.node_labels,
        seq_length=8,
        alpha=1.,
        beta=1.
    )
    seqs = walker.traverse(dump_flag=True)




