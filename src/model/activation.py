# -*- coding: utf-8 -*-


from keras import backend as K


def round_through(x):
    rounded = K.round(x)
    return x + K.stop_gradient(rounded - x)


def clip_through(x, min_val, max_val):
    clipped = K.clip(x, min_val, max_val)
    clipped_through = x + K.stop_gradient(clipped - x)
    return clipped_through


def _hard_sigmoid(x):
    x = (0.5 * x) + 0.5
    return K.clip(x, 0, 1)


def binary_sigmoid(x):
    return round_through(_hard_sigmoid(x))


def binary_tanh(x):
    x = 2 * binary_sigmoid(x) - 1
    return x


def quantized_tanh(W, nb=16):
    non_sign_bits = nb-1
    m = pow(2,non_sign_bits)
    Wq = K.clip(round_through(W*m),-m,m-1)/m
    return Wq