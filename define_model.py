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
from keras.layers.core import Reshape, Permute
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.activations import linear
from trafficgraphnn.batch_graph_attention_layer import  BatchGraphAttention
from keras.utils.vis_utils import plot_model

from trafficgraphnn.attention_decoder import AttentionDecoder

### Configuration of the deep learning model
width_1gat = 128 # Output dimension of first GraphAttention layer
F_ = 128         # Output dimension of last GraphAttention layer
n_attn_heads = 5              # Number of attention heads in first GAT layer
dropout_rate = 0.5            # Dropout rate applied to the input of GAT layers
attn_dropout = 0.5            #Dropout of the adjacency matrix in the gat layer
l2_reg = 1e-4               # Regularization rate for l2
learning_rate = 0.001      # Learning rate for optimizer
n_units = 128   #number of units of the LSTM cells



def define_model(num_samples, num_timesteps, num_lanes, num_features, A, save_model = False):
    #define necessary functions  
    
    def reshape_for_GAT(input_arr):
        input_arr_reshaped = K.reshape(input_arr, (-1, num_lanes, num_timesteps, num_features))
        output_arr = K.permute_dimensions(input_arr_reshaped, (0, 2, 1, 3))
        return output_arr
    
    def reshape_for_LSTM(input_arr):
        input_arr_permuted = K.permute_dimensions(input_arr, (0, 2, 1, 3))
        output_arr = K.reshape(input_arr_permuted, (-1, num_timesteps, input_arr.shape[3]))
        return output_arr
        
    #define the model
    #A_tf = tf.convert_to_tensor(A, dtype=np.float32)
    A = A.astype(np.float32)
#        num_lanes = tf.convert_to_tensor(N, dtype=np.float32) #num_lanes ha to become an Input later, rn it's hardcoded
    
    X1_in = Input(batch_shape=(None, num_timesteps, num_features)) #(num_lanes*num_old_samples x timesteps x num_features)
#        num_lanes_in = Input(shape=(1,)) 

    
    shaped_X1_in = Lambda(reshape_for_GAT)(X1_in)
    #reshaped_X1_in = Reshape((-1, num_lanes, num_timesteps, num_features))(X1_in)
    #permuted_X1_in = Permute((0, 2, 1, 3))(reshaped_X1_in)
    
    dropout1 = TimeDistributed(Dropout(dropout_rate))(shaped_X1_in)
    
    graph_attention_1 = TimeDistributed(BatchGraphAttention(width_1gat,
                                       A,
                                       attn_heads=n_attn_heads,
                                       attn_heads_reduction='average',
                                       attn_dropout=attn_dropout,
                                       activation='linear',
                                       kernel_regularizer=l2(l2_reg))
                                       )(dropout1)
    
    dropout2 = TimeDistributed(Dropout(dropout_rate))(graph_attention_1)
    
    graph_attention_2 = TimeDistributed(BatchGraphAttention(F_,
                                       A,
                                       attn_heads=n_attn_heads,
                                       attn_heads_reduction='average',
                                       attn_dropout=attn_dropout,
                                       activation='linear',
                                       kernel_regularizer=l2(l2_reg)))(dropout2)
    
    dropout3 = TimeDistributed(Dropout(dropout_rate))(graph_attention_2)
    
    dense1 = TimeDistributed(Dense(128, activation = linear, 
                                   kernel_regularizer=l2(l2_reg)))(dropout3)
    
    dropout4 = TimeDistributed(Dropout(dropout_rate))(dense1)
    
    #make sure that the reshape is made correctly!
    encoder_inputs = Lambda(reshape_for_LSTM)(dropout4)
    #permuted_enc_inputs = TimeDistributed(Permute((0, 2, 1, 3)))(dropout4)
    #encoder_inputs = TimeDistributed(Reshape((-1, num_timesteps, 128)))(permuted_enc_inputs)  #ATTENTION!!! Adjust feature dimesion!!
    
    
    decoder_inputs = LSTM(n_units,
         batch_input_shape=(num_samples, num_timesteps, num_features), 
         return_sequences=True)(encoder_inputs)
    
    decoder_output = AttentionDecoder(n_units, 1)(decoder_inputs) #Attention! 5 output features now!

    model = Model(inputs= X1_in, outputs=decoder_output) 
    
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
    
    if save_model:
        model.save('models/untrained_model_complete.h5')
    return model