# -*- coding: utf-8 -*-

"""
@Time   : 2020-05-04 16:33
@Author : Kyrie.Z
@File   : vector_quantization.py
"""

import pandas as pd
from keras.models import Model
from scipy.spatial import distance

from src.model.metrics import get_metric_sk
from src.model.network import siamese_network, SiameseNetworkDistance
from src.train_siamese_network import make_tokenizer, make_embedding_matrix, padding, build_df_from_file

from src.utils.quantizer import Quantizer
from src.utils.config import *


def hash_classifier(left_vectors, right_vectors):
    assert len(left_vectors) == len(right_vectors)
    answer = []
    for i in range(len(left_vectors)):
        answer.append(distance.hamming(left_vectors[i], right_vectors[i]))
    return answer


def hamming2(left_vectors, right_vectors):
    """Calculate the Hamming distance between two bit strings"""
    assert len(left_vectors) == len(right_vectors)
    answer = []
    for i in range(len(left_vectors)):
        answer.append(sum(c1 != c2 for c1, c2 in zip(left_vectors[i], right_vectors[i])))
    return answer


def build_separate_input(file_name, name: Columns, tknizer, cfg):
    dataframe = build_df_from_file(file_name, name)
    print(dataframe[name.label].value_counts())

    y = dataframe.pop(name.label)

    left_p = [
        padding(dataframe[name.aspects_1], tknizer, cfg.max_len)
    ]
    right_p = [
        padding(dataframe[name.aspects_2], tknizer, cfg.max_len)
    ]

    return left_p, right_p, y


if __name__ == '__main__':
    nm = Columns()
    tc = Configuration(base_path='../', suffix='flipkart', embed_dim=128, vec_prefix='sdne')

    tokenizer = make_tokenizer(tc.aspect_id_corpus, tc.id_tokenizer_cache)
    embeddings = make_embedding_matrix(tokenizer, tc.embed_dim, tc.aspect_vector_file, tc.aspect_vector_cache)

    model = SiameseNetworkDistance(
        vocab_size=len(tokenizer.word_index) + 1,
        embed_dim=tc.embed_dim,
        embed_matrix=embeddings,
        title_max_len=tc.max_len,
        verbose=True
    )
    model.load_weights(tc.model_file)

    print("Build model input ...")
    input_left, input_right, label = build_separate_input(tc.valid_id_file, nm,tokenizer, tc)

    layer_nm = "left_output"
    inter_layer_model = Model(inputs=model.input[:1],
                              outputs=model.get_layer(layer_nm).output)

    left_output = inter_layer_model.predict(input_left, batch_size=256, verbose=1)
    right_output = inter_layer_model.predict(input_right, batch_size=256, verbose=1)

    df_left = pd.DataFrame(left_output)
    df_right = pd.DataFrame(right_output)

    # print(df_left)
    # print(df_right)

    total_title = pd.concat([df_left, df_right]).drop_duplicates()
    print("Total title: {}".format(total_title.shape[0]))

    # for exact match
    vq = Quantizer(n_clusters=2)
    vq.fit(total_title)

    df_left = vq.transform(df_left)
    # print(df_left['hashcode'])
    df_right = vq.transform(df_right)
    # print(df_right['hashcode'])

    preds = []

    for i in range(len(df_left)):
        if df_left[vq.code][i] == df_right[vq.code][i]:
            preds.append(1)
        else:
            preds.append(0)

    metrics = get_metric_sk(label, preds)
    print("metrics: acc-{:.4f}, p-{:.4f}, r-{:.4f}, f1-{:.4f}".format(metrics[3], metrics[0], metrics[1], metrics[2]))

# aspect2vec: metrics: acc-0.9608, p-0.9512, r-0.7959, f1-0.8667
# deepwalk:   metrics: acc-0.9151, p-0.9623, r-0.4889, f1-0.6484
# sdne:       metrics: acc-0.9274, p-0.8312, r-0.6863, f1-0.7518
# node2vec:   metrics: acc-0.9204, p-0.9694, r-0.5192, f1-0.6762
# line:       metrics: acc-0.9472, p-0.9786, r-0.6855, f1-0.8062
# struct2vec: metrics: acc-0.9107, p-0.9688, r-0.4570, f1-0.6210
