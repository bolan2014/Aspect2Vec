# -*- coding: utf-8 -*-

"""
@Time   : 2020-04-02 16:32
@Author : Kyrie.Z
@File   : initializer.py
"""


import os
import pickle
import networkx as nx
from collections import Counter

from src.utils.config import Configuration


class GraphInitializer(object):
    def __init__(self, config: Configuration):
        self.cf = config
        self.logging = config.logging
        self.graph = self.build_graph()
        self.node_labels = self.get_node_label()
        self.node_index = self.get_node_name()
        self.aspect_by_item = self.get_aspect_by_item()

    def build_graph(self):
        """
        convert the edge list to what the `nx` can take in
        example of edge line: 1 2\n
        :return:
        """
        if os.path.exists(self.cf.graph_cache):
            self.logging.info("load graph from cache file ...")
            G = nx.read_gpickle(path=self.cf.graph_cache)
            self.logging.info("load {} nodes and {} edges from data".format(len(G.nodes), len(G.edges())))
            return G
        else:
            with open(self.cf.edge_file) as handler:
                edge_lines = handler.readlines()
                edge_lines = list(map(lambda x: x.strip(), edge_lines))
                edge_counter = Counter(edge_lines)
                weighted_edges = []
                for edge, weight in edge_counter.items():
                    # freshness between 2 nodes is initialized to 1
                    weighted_edges.append("{} {} 1".format(edge, weight))
                G = nx.parse_edgelist(weighted_edges, nodetype=int, data=(('weight', float), ('freshness', float)))
                self.logging.info("load {} nodes and {} edges from data".format(len(G.nodes), len(G.edges())))
                # save to cache file
                nx.write_gpickle(G, path=self.cf.graph_cache)

                return G

    def get_node_label(self):
        """
        get node type -> aspect key, label map -> {"upc": '1', "mpn": '2', "ean": '3', "brand": '4', "type": '5',
        "power": '6', "voltage": '7', "model": '8', "color": '9'}
        :return:
        """
        with open(self.cf.edge_label_cache, "rb") as handler:
            labels = pickle.load(handler)
            self.logging.info("load {} node labels from cache file".format(len(labels)))
            return labels

    def get_node_name(self):
        """
        get value from idx -> aspect name
        :return:
        """
        with open(self.cf.id_to_aspect_cache, "rb") as handler:
            names = pickle.load(handler)
            self.logging.info("load {} node names from cache file".format(len(names)))
            return names

    def get_aspect_by_item(self):
        """
        [[aspect1, aspect2], [], ...]
        :return:
        """
        with open(self.cf.aspect_by_item_cache, "rb") as handler:
            aspect_by_item = pickle.load(handler)
            self.logging.info("load aspect of {} items from cache file".format(len(aspect_by_item)))
            return aspect_by_item


if __name__ == '__main__':
    CF = Configuration(base_path='../../', suffix='ebay', file_name='')
    initializer = GraphInitializer(CF)
