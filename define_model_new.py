#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 18:29:37 2018

@author: simon
"""

import numpy as np

import tensorflow as tf
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dropout, Dense, TimeDistributed, Lambda, LSTM, Reshape, Permute
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.activations import linear
from trafficgraphnn.batch_graph_attention_layer import  BatchGraphAttention
from trafficgraphnn.time_distributed_multi_input import TimeDistributedMultiInput
from keras.utils.vis_utils import plot_model

from trafficgraphnn.attention_decoder import AttentionDecoder
#from trafficgraphnn.preprocess_data import reshape_for_4Dim, reshape_for_3Dim
from trafficgraphnn.reshape_layers import ReshapeForLSTM, ReshapeForOutput


### Configuration of the deep learning model
width_1gat = 128 # Output dimension of first GraphAttention layer
F_ = 128         # Output dimension of last GraphAttention layer
n_attn_heads = 5              # Number of attention heads in first GAT layer
dropout_rate = 0.5            # Dropout rate applied to the input of GAT layers
attn_dropout = 0.5            #Dropout of the adjacency matrix in the gat layer
l2_reg = 1e-4               # Regularization rate for l2
learning_rate = 0.001      # Learning rate for optimizer
n_units = 128   #number of units of the LSTM cells

def define_model(num_simulations, num_timesteps, num_lanes, num_features, A):

        X1_in = Input(batch_shape=(None, num_timesteps, num_lanes, num_features))
        A_in = Input(batch_shape=(None, num_timesteps, num_lanes, num_lanes))

        dropout1 = TimeDistributed(Dropout(dropout_rate))(X1_in)

        graph_attention_1 = TimeDistributedMultiInput(BatchGraphAttention(width_1gat,
                                           attn_heads=n_attn_heads,
                                           attn_heads_reduction='average',
                                           attn_dropout=attn_dropout,
                                           activation='linear',
                                           kernel_initializer='random_uniform',
                                           kernel_regularizer=l2(l2_reg))
                                           )([dropout1, A_in])

        dropout2 = TimeDistributed(Dropout(dropout_rate))(graph_attention_1)

        graph_attention_2 = TimeDistributedMultiInput(BatchGraphAttention(F_,
                                           attn_heads=n_attn_heads,
                                           attn_heads_reduction='average',
                                           attn_dropout=attn_dropout,
                                           activation='linear',
                                           kernel_regularizer=l2(l2_reg),
                                           kernel_initializer='random_uniform'))([dropout2, A_in])

        dropout3 = TimeDistributed(Dropout(dropout_rate))(graph_attention_2)

        dense1 = TimeDistributed(Dense(128, activation = linear,
                                       kernel_regularizer=l2(l2_reg),
                                       kernel_initializer='random_uniform'))(dropout3)

        dropout4 = TimeDistributed(Dropout(dropout_rate))(dense1)

        #make sure that the reshape is made correctly!
        encoder_inputs = ReshapeForLSTM()(dropout4)



        decoder_inputs = LSTM(n_units,
                              return_sequences=True,
                              kernel_initializer='random_uniform')(encoder_inputs)

        decoder_output = AttentionDecoder(n_units, 2, causal=True)(decoder_inputs) #Attention! 2 output features now!

        reshaped_output = ReshapeForOutput(num_lanes)(decoder_output)

        model = Model(inputs= [X1_in, A_in], outputs=reshaped_output)

        optimizer = Adam(lr=learning_rate)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error',
                      metrics=['mean_absolute_percentage_error'])
        model.summary()
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        return model
