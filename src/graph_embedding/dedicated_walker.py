# -*- coding: utf-8 -*-

"""
@Time   : 2020-04-02 16:31
@Author : Kyrie.Z
@File   : dedicated_walker.py
"""

import math
import random
import pickle
from tqdm import tqdm


class Walker(object):
    def __init__(self, G, config, node_labels: dict, seq_length, alpha, beta):
        self.G = G
        self.node_labels = node_labels
        self.seq_length = seq_length
        self.alpha = alpha
        self.beta = beta
        self.cf = config

    def traverse(self, n=100000, dump_flag=True):
        """
        generate a traversal sequence from each node
        :param n:
        :param dump_flag:
        :param node:
        :return:
        """
        sequences = []
        self.cf.logging.info("generating walk sequences...")
        cnt = 0
        for node in tqdm(self.G):
            if cnt > n:
                break
            seq = self.walk(start_node=node)
            sequences.append(seq)
            cnt += 1

        if dump_flag:
            # dump walk sequences
            with open(self.cf.sequences_file, 'w') as handler:
                for s in sequences:
                    s = list(map(str, s))
                    handler.write(' '.join(s)+'\n')
                    # pickle.dump(sequences, handler, protocol=pickle.HIGHEST_PROTOCOL)
        return sequences

    def walk(self, start_node):
        sequence = [start_node]
        cur_node = start_node
        tabu = [self.node_labels[cur_node]]
        # TODO: find other terminate conditions
        while len(sequence) != self.seq_length:
            next_node = self.next_step(cur_node, tabu)
            if next_node is None:  # cannot find a proper next node
                break
            tabu.append(self.node_labels[next_node])
            sequence.append(next_node)
            cur_node = next_node
        self.update_freshness(sequence)
        return sequence

    def next_step(self, cur_node, tabu):
        """
        the method to choose a next node
        :param cur_node:
        :param tabu:
        :return:
        """
        neighbors = self.G[cur_node]  # one-hop neighbors
        probs = self.calc_probs(neighbors, tabu)
        if probs is not None:  # cannot find a proper next node
            next_node = self.roulette_choose(probs)
            return next_node
        elif len(neighbors) > 0:
            return random.choice(list(neighbors))
        return

    def calc_probs(self, neighbors_info, forbidden_labels):
        """
        calculate the probabilities of moving to other allowed nodes
        :param neighbors_info:
        :param forbidden_labels:
        :return:
        """
        probs = {}
        for neighbor in neighbors_info:
            if self.node_labels[neighbor] in forbidden_labels:
                continue
            info = neighbors_info[neighbor]
            probs[neighbor] = self.calc_attractiveness(info)
        if len(probs) == 0:
            return
        # normalize
        total_attract = sum(probs.values())  # allowed neighbors
        for neighbor, prob in probs.items():
            probs[neighbor] = prob / total_attract
        assert abs(sum(probs.values()) - 1) < 0.1
        return probs

    def calc_attractiveness(self, info: dict):
        """
        calculate attractiveness of a neighbor node
        :param info: contain `weight` and `freshness`
        :return:
        """
        tau, eta = info['freshness'], info['weight']
        return math.pow(tau, self.alpha) * math.pow(eta, self.beta)

    @staticmethod
    def roulette_choose(probs: dict):
        pick = random.uniform(0, 1.)
        current = 0.
        for neighbor, attract in probs.items():
            current += attract
            if current > pick:
                return neighbor
            else:
                continue

    # TODO: implement the update logic at the end of traverse
    def update_freshness(self, seqs):
        if len(seqs) < 2:
            return
        # calculate total distance
        total_dis = 0
        for i in range(len(seqs) - 1):
            total_dis += 1 / self.G[seqs[i]][seqs[i+1]]['weight']
        rou = total_dis / (1 + total_dis)

        for j in range(len(seqs) - 1):
            freshness = self.G[seqs[j]][seqs[j+1]]['freshness']
            self.G[seqs[j]][seqs[j+1]]['freshness'] = freshness * rou


if __name__ == '__main__':
    pass
