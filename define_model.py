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
from keras.layers import Input, Dropout, Dense, TimeDistributed, Lambda, LSTM
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.activations import linear
from trafficgraphnn.batch_graph_attention_layer import  BatchGraphAttention
from keras.utils.vis_utils import plot_model

from trafficgraphnn.attention_decoder import AttentionDecoder
from trafficgraphnn.preprocess_data import reshape_for_GAT, reshape_for_LSTM


### Configuration of the deep learning model
width_1gat = 128 # Output dimension of first GraphAttention layer
F_ = 128         # Output dimension of last GraphAttention layer
n_attn_heads = 5              # Number of attention heads in first GAT layer
dropout_rate = 0.5            # Dropout rate applied to the input of GAT layers
attn_dropout = 0.5            #Dropout of the adjacency matrix in the gat layer
l2_reg = 1e-4               # Regularization rate for l2
learning_rate = 0.001      # Learning rate for optimizer
n_units = 128   #number of units of the LSTM cells



def define_model(sample_size_train, timesteps_per_sample, N, F, A):
    #define necessary functions   
        
        #define the model
        A_tf = tf.convert_to_tensor(A, dtype=np.float32)
#        num_lanes = tf.convert_to_tensor(N, dtype=np.float32) #num_lanes ha to become an Input later, rn it's hardcoded
        
        X1_in = Input(batch_shape=(sample_size_train*N, timesteps_per_sample, F))
#        num_lanes_in = Input(shape=(1,)) 

        
        shaped_X1_in = Lambda(reshape_for_GAT)(X1_in)
        
        dropout1 = TimeDistributed(Dropout(dropout_rate))(shaped_X1_in)
        
        graph_attention_1 = TimeDistributed(BatchGraphAttention(width_1gat,
                                           A_tf,
                                           attn_heads=n_attn_heads,
                                           attn_heads_reduction='average',
                                           attn_dropout=attn_dropout,
                                           activation='linear',
                                           kernel_initializer='random_uniform',
                                           kernel_regularizer=l2(l2_reg))
                                           )(dropout1)
        
        dropout2 = TimeDistributed(Dropout(dropout_rate))(graph_attention_1)
        
        graph_attention_2 = TimeDistributed(BatchGraphAttention(F_,
                                           A_tf,
                                           attn_heads=n_attn_heads,
                                           attn_heads_reduction='average',
                                           attn_dropout=attn_dropout,
                                           activation='linear',
                                           kernel_regularizer=l2(l2_reg),
                                           kernel_initializer='random_uniform'))(dropout2)
        
        dropout3 = TimeDistributed(Dropout(dropout_rate))(graph_attention_2)
        
        dense1 = TimeDistributed(Dense(128, activation = linear, 
                                       kernel_regularizer=l2(l2_reg), 
                                       kernel_initializer='random_uniform'))(dropout3)
        
        dropout4 = TimeDistributed(Dropout(dropout_rate))(dense1)
        
        #make sure that the reshape is made correctly!
        encoder_inputs = Lambda(reshape_for_LSTM)(dropout4)
        decoder_inputs = LSTM(n_units,
             batch_input_shape=(sample_size_train*N, timesteps_per_sample, F), 
             return_sequences=True, 
             kernel_initializer='random_uniform')(encoder_inputs)
        
        decoder_output = AttentionDecoder(n_units, 1)(decoder_inputs) #Attention! 5 output features now!

        model = Model(inputs= X1_in, outputs=decoder_output) 
        
        optimizer = Adam(lr=learning_rate)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error',
                      metrics=['accuracy'])
        model.summary()
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        
        model.save('models/untrained_model_complete.h5')
        return model