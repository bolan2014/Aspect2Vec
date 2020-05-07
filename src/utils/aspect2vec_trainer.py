# -*- coding: utf-8 -*-

"""
@Time   : 2020-05-04 09:54
@Author : Kyrie.Z
@File   : aspect2vec_trainer.py
"""

import fasttext


class Trainer:
    def __init__(self, input_file, model='cbow', dim=100, epoch=50, minCount=1):
        self.model = fasttext.train_unsupervised(
            input_file,
            model=model,
            dim=dim,
            epoch=epoch,
            minCount=minCount
        )

    def dump_to_file(self, file_name):
        words = self.model.get_words()
        with open(file_name, "w", encoding='utf-8') as handler:
            handler.write("{} {}\n".format(len(words), self.model.get_dimension()))
            for w in words:
                v = self.model.get_word_vector(w)
                vstr = w
                for vi in v:
                    vstr += " " + str(vi)
                handler.write("{}\n".format(vstr))

