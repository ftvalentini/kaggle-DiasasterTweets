import numpy as np
import matplotlib.pyplot as plt

from neural.core import helpers

from keras.models import Sequential
from keras.layers import Embedding, Dense, CuDNNGRU, LeakyReLU
from keras.optimizers import Adam, SGD
from keras.regularizers import l2

def build_gru(tokenizer
              ,optimizer='adam'
              ,learn_rate=0.001, l2_strength=0.001, decay_strength=0.0, momentum=0.9
              ,embedding_dim=128
              ):
    """
    Build and compile NN with GRU for classification
    -- if vocab_size = None --> use all words
    -- pad_type = 'pre' or 'post'
    -- optimizer = 'adam' or 'sgd'
    """
    # define vocab_size
    vocab_size = len(tokenizer.word_index) + 1
    # build NN
    relu_leak = 0.1
    if optimizer=='adam':
        optimizer = Adam(lr=learn_rate, decay=decay_strength)
    if optimizer=='sgd':
        optimizer = SGD(lr=learn_rate, decay=decay_strength, momentum=momentum
                        ,nesterov=True)
    regularizer = l2(l2_strength)
    mod = Sequential()
    mod.add(Embedding(vocab_size, embedding_dim
                        # ,input_length=seq_maxlen
                        ))
    mod.add(CuDNNGRU(512, kernel_regularizer=regularizer))
    mod.add(Dense(256, kernel_regularizer=regularizer))
    mod.add(LeakyReLU(alpha=relu_leak))
    mod.add(Dense(128, kernel_regularizer=regularizer))
    mod.add(LeakyReLU(alpha=relu_leak))
    mod.add(Dense(1, activation='sigmoid'))
    # compile
    mod.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
    return mod

# TODO: add build_bigru y otras arquitecturas
