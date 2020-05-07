# -*- coding: utf-8 -*-

"""
@Time   : 2020-05-04 10:28
@Author : Kyrie.Z
@File   : train_siamese_network.py
"""
import os
import codecs
import pickle
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint

from src.model.network import SiameseNetworkDistance
from src.model.metrics import DistanceMetirc, contrastive_loss, distance_acc

from src.utils.config import *


def make_tokenizer(file_name, cache_file):
    if not os.path.exists(cache_file):
        tokenizer = Tokenizer(char_level=False)
        corpus = []
        with open(file_name, encoding='utf-8') as reader:
            for line in reader:
                corpus.append(line.strip())
        tokenizer.fit_on_texts(corpus)
        with open(cache_file, "wb") as writer:
            pickle.dump(tokenizer, writer, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        tokenizer = pickle.load(open(cache_file, "rb"))
    return tokenizer


def padding(text, tokenizer, sent_len):
    sequence = tokenizer.texts_to_sequences(text)
    return pad_sequences(sequence, maxlen=sent_len, padding='pre')


def make_embedding_matrix(tokenizer, vec_dim, embed_file, cache_file):
    if not os.path.exists(cache_file):
        embedding_index = {}
        with codecs.open(embed_file, encoding='utf-8') as reader:
            next(reader)
            for line in reader:
                values = line.strip().split(" ")
                word = values[0]
                coefs = np.asarray(values[1:], dtype="float32")
                embedding_index[word] = coefs
        word_index = tokenizer.word_index
        embedding_matrix = np.zeros((len(word_index)+1, vec_dim))
        for word, index in word_index.items():
            vector = embedding_index.get(word)
            if vector is not None:
                embedding_matrix[index] = vector
        np.save(cache_file, embedding_matrix)
    else:
        embedding_matrix = np.load(cache_file)
    return embedding_matrix


def build_df_from_file(file_name, name: Columns):
    dataframe = pd.read_csv(file_name, sep='\t')
    return dataframe[name.all]


def build_input_df(file_name:str, cfg: Configuration, tknizer):
    name = cfg.cols

    dataframe = build_df_from_file(file_name, name)

    print(dataframe[name.label].value_counts())
    # print(dataframe[name.title1].isna().sum())
    # print(dataframe[name.title2].isna().sum())

    y = dataframe.pop(name.label)
    input_df = [
        padding(dataframe[name.aspects_1], tknizer, cfg.max_len),
        padding(dataframe[name.aspects_2], tknizer, cfg.max_len)
    ]
    return input_df, y


if __name__ == '__main__':
    columns = Columns()
    # tc = Configuration(base_path='../', suffix='flipkart', cols=columns)
    tc = Configuration(base_path='../', suffix='flipkart', cols=columns, embed_dim=128, vec_prefix='sdne')

    aspect_tokenizer = make_tokenizer(tc.aspect_id_corpus, tc.id_tokenizer_cache)
    embeddings = make_embedding_matrix(aspect_tokenizer, tc.embed_dim, tc.aspect_vector_file, tc.aspect_vector_cache)

    print("Training samples:")
    train_input, train_y = build_input_df(tc.train_id_file, tc, aspect_tokenizer)
    print("Validation samples:")
    valid_input, valid_y = build_input_df(tc.valid_id_file, tc, aspect_tokenizer)

    model = SiameseNetworkDistance(
        vocab_size=len(aspect_tokenizer.word_index) + 1,
        embed_dim=tc.embed_dim,
        embed_matrix=embeddings,
        title_max_len=tc.max_len,
        verbose=True
    )

    metric = DistanceMetirc()
    earlyStop = EarlyStopping(monitor="val_loss", patience=2, verbose=0)
    checkPoint = ModelCheckpoint(tc.model_file, monitor="val_loss", save_best_only=True, verbose=0)

    model.compile(loss=contrastive_loss, optimizer='nadam', metrics=[distance_acc])

    model.fit(train_input, train_y, epochs=8, batch_size=1280, callbacks=[metric, checkPoint],
              validation_data=(valid_input, valid_y))

