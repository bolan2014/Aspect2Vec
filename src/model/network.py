# -*- coding: utf-8 -*-


from keras.layers import Dense
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers import Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import SpatialDropout1D
from keras.layers.merge import concatenate
from keras.layers import Lambda
from keras.models import Model
from keras.layers import BatchNormalization
from keras import regularizers

from model.metrics import euclidean_distance, dist_output_shape, cosine_distance
from model.attention import AttentionWithContext, DoNothing


def siamese_network(vocab_size, embed_dim, embed_matrix, title_max_len, image_embed_dim=1024, price_dim=1, output_dim=64, verbose=True):

    def create_base_network():
        title_input = Input(shape=(title_max_len,))
        title_embedding = Embedding(vocab_size, embed_dim, weights=[embed_matrix], trainable=False)(title_input)
        title_part = Bidirectional(LSTM(50, return_sequences=True))(title_embedding)
        title_part = AttentionWithContext()(title_part)

        image_input = Input(shape=(image_embed_dim,))
        image_part = Dense(64, activation='linear', kernel_initializer='uniform')(image_input)
        image_part = BatchNormalization()(image_part)
        image_part = Dropout(0.2)(image_part)
        image_part = Dense(32, activation='sigmoid', kernel_initializer='glorot_uniform')(image_part)

        price_input = Input(shape=(price_dim,))
        price_part = Dense(1, activation='linear', kernel_initializer='uniform')(price_input)
        price_part = BatchNormalization()(price_part)
        price_part = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')(price_part)

        merged = concatenate([title_part, image_part, price_part])
        merged = Dense(output_dim, kernel_initializer='glorot_uniform', activation='softsign', kernel_regularizer=regularizers.l1(0.01))(merged)
        return Model(
            inputs=[title_input, image_input, price_input],
            outputs=merged
        )

    base_network = create_base_network()

    # if verbose:
    #     base_network.summary()

    title_input_left = Input(shape=(title_max_len,))
    image_input_left = Input(shape=(image_embed_dim,))
    price_input_left = Input(shape=(price_dim,))
    left_part = base_network([title_input_left, image_input_left, price_input_left])
    left_part = DoNothing()(left_part)

    title_input_right = Input(shape=(title_max_len,))
    image_input_right = Input(shape=(image_embed_dim,))
    price_input_right = Input(shape=(price_dim,))
    right_part = base_network([title_input_right, image_input_right, price_input_right])
    right_part = DoNothing()(right_part)

    preds = Lambda(euclidean_distance, output_shape=dist_output_shape)([left_part, right_part])

    model = Model(
        inputs=[title_input_left, image_input_left, price_input_left, title_input_right, image_input_right, price_input_right],
        outputs=preds
    )

    if verbose:
        model.summary()
    return model


def SiameseNetworkDistance(vocab_size, embed_dim, embed_matrix, title_max_len, verbose=False):
    def create_base_network():
        title_input = Input(shape=(title_max_len,))
        title_embedding = Embedding(vocab_size, embed_dim, weights=[embed_matrix], trainable=False)(title_input)
        title_bilstm = Bidirectional(LSTM(32, return_sequences=True))(title_embedding)
        title_attention = AttentionWithContext()(title_bilstm)
        # title_part = Dense(32, kernel_initializer='glorot_uniform', activation='sigmoid', kernel_regularizer=regularizers.l1(0.01))(title_attention)
        return Model(
            inputs=[title_input],
            outputs=title_attention
        )
    base_network = create_base_network()

    title_input_left = Input(shape=(title_max_len,))
    title_left = base_network([title_input_left])
    title_left = DoNothing(name="left_output")(title_left)

    title_input_right = Input(shape=(title_max_len,))
    title_right = base_network([title_input_right])
    title_right = DoNothing(name="right_output")(title_right)

    preds = Lambda(euclidean_distance, output_shape=dist_output_shape)([title_left, title_right])

    model = Model(
        inputs=[title_input_left, title_input_right],
        outputs=preds
    )
    if verbose:
        model.summary()

    return model


def siamese_mlp(input_dim=1024, verbose=1):

    def create_base_network():
        model_input = Input(shape=(input_dim,))
        linear_layer = Dense(64, activation='linear', kernel_initializer='uniform')(model_input)
        batch_norm = BatchNormalization()(linear_layer)
        dropout = Dropout(0.2)(batch_norm)
        dense = Dense(32, activation='sigmoid', kernel_initializer='glorot_uniform')(dropout)
        return Model(
            inputs=model_input,
            outputs=dense
        )

    base_network = create_base_network()

    input_left = Input(shape=(input_dim,))
    left_part = base_network(input_left)
    input_right = Input(shape=(input_dim,))
    right_part = base_network(input_right)


    # price_input_left = Input(shape=(1,))
    # price_left = Dense(1, kernel_initializer='uniform', activation='linear')(price_input_left)

    # price_input_right = Input(shape=(1,))
    # price_right = Dense(1, kernel_initializer='uniform', activation='linear')(price_input_right)

    # merged_left = concatenate([image_left, price_left])
    # merged_right = concatenate([image_right, price_right])
    # merged = concatenate([image_left, image_right], axis=-1)

    preds = Lambda(euclidean_distance, output_shape=dist_output_shape)([left_part, right_part])
    # preds = Dense(1, activation='sigmoid')(merged)

    model = Model(
        inputs=[input_left, input_right],
        outputs=preds
    )
    if verbose:
        model.summary()

    return model

