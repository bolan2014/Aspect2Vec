# -*- coding: utf-8 -*-

"""
@Time   : 2020-04-02 16:32
@Author : Kyrie.Z
@File   : initializer.py
"""


import os
import pickle
import logging
import networkx as nx
from collections import Counter

from src.utils.config import Configuration

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class Initializer(object):
    def __init__(self, config: Configuration):
        self.config = config
        self.graph = self.build_graph()
        self.node_labels = self.get_node_label()
        self.node_index = self.get_node_index()

    def build_graph(self):
        """
        convert the edgelist to what the `nx` can take in
        example of edge line: 1 2\n
        :return:
        """
        if os.path.exists(self.config.edge_cache):
            logging.info("load graph from cache file ...")
            G = nx.read_gpickle(path=self.config.edge_cache)
            logging.info("load {} nodes and {} edges from data".format(len(G.nodes), len(G.edges())))
            return G
        else:
            with open(self.config.edge_file) as handler:
                edge_lines = handler.readlines()
                edge_lines = list(map(lambda x: x.strip(), edge_lines))
                edge_counter = Counter(edge_lines)
                weighted_edges = []
                for edge, weight in edge_counter.items():
                    # freshness between 2 nodes is initialized to 1
                    weighted_edges.append("{} {} 1".format(edge, weight))
                G = nx.parse_edgelist(weighted_edges, nodetype=str, data=(('weight', float), ('freshness', float)))
                logging.info("load {} nodes and {} edges from data".format(len(G.nodes), len(G.edges())))
                # save to cache file
                nx.write_gpickle(G, path=self.config.edge_cache)

                return G

    def get_node_label(self):
        """
        get node type -> aspect key, label map -> {"upc": '1', "mpn": '2', "ean": '3', "brand": '4', "type": '5',
        "power": '6', "voltage": '7', "model": '8', "color": '9'}
        :return:
        """
        if os.path.exists(self.config.edge_label_cache):
            with open(self.config.edge_label_cache, "rb") as handler:
                labels = pickle.load(handler)
                logging.info("load {} node labels from cache file".format(len(labels)))
                return labels
        else:
            with open(self.config.edge_label_file) as handler:
                labels = {}
                for line in handler:
                    key, val = line.strip().split(" ")
                    labels[key] = val
                logging.info("load {} node labels from file".format(len(labels)))

            with open(self.config.edge_label_cache, "wb") as handler:
                pickle.dump(labels, handler, protocol=pickle.HIGHEST_PROTOCOL)
            return labels

    def get_node_index(self):
        """
        get value from idx -> aspect name
        :return:
        """
        with open(self.config.node_index_cache, "rb") as handler:
            names = pickle.load(handler)
            logging.info("load {} node names from cache file".format(len(names)))
            return names


if __name__ == '__main__':
    CF = Configuration(base_dir='../../')
    initializer = Initializer(CF)

    assert len(initializer.graph.nodes) == 22840
    assert len(initializer.graph.edges) == 80667
    assert len(initializer.node_labels) == 22841
    assert len(initializer.node_index) == 22841
    assert initializer.node_index.inverse[22000] == 'sd20017'
