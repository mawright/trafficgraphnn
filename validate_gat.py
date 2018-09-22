#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 10:34:55 2018

@author: simon
"""

from __future__ import division
from trafficgraphnn.genconfig import ConfigGenerator
from trafficgraphnn.sumo_network import SumoNetwork
import numpy as np
import pandas as pd
import math
import pickle
from trafficgraphnn.liumethod import LiuEtAlRunner

from trafficgraphnn.preprocess_data import PreprocessData

import tensorflow as tf
import keras.backend as K
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Input, Dropout, Dense, TimeDistributed, Reshape, Lambda, LSTM
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD, Adagrad
from keras.regularizers import l2
from keras.activations import relu, linear
from trafficgraphnn.batch_graph_attention_layer import  BatchGraphAttention
from keras.utils.vis_utils import plot_model
from trafficgraphnn.attention_decoder import AttentionDecoder
from trafficgraphnn.postprocess_predictions import store_predictions_in_df, resample_predictions

#------ Configuration of the whole simulation -------
### Configuration of the Network ###
grid_number = 3 #TODO: make num lanes adjustable
#N = 120 #number of lanes after getting subgraph
grid_length = 600 #meters
num_lanes =3

### Configuration of the Simulation ###
end_time = 1500 #seconds

period_1_2 = 0.3
period_3_4 = 0.35
period_5 = 0.4

period_test = 0.4

binomial = 2
seed = 50
fringe_factor = 1000

### Configuration of Liu estimation ###
use_started_halts = False #use startet halts as ground truth data or maxJamLengthInMeters
show_plot = False
show_infos = False

### Configuration for preprocessing the detector data
average_interval = 10  #Attention -> right now no other average interval than 1 is possible  -> bugfix necessary!
sample_size = 15     #number of steps per sample in size of average interval
interpolate_ground_truth = True #interpolate ground-truth data with np.linspace

### Configuration of the deep learning model
width_1gat = 128 # Output dimension of first GraphAttention layer
F_ = 64         # Output dimension of last GraphAttention layer
n_attn_heads = 5              # Number of attention heads in first GAT layer
dropout_rate = 0.3            # Dropout rate applied to the input of GAT layers
attn_dropout = 0.3            #Dropout of the adjacency matrix in the gat layer
l2_reg = 5e-100               # Regularization rate for l2
learning_rate = 0.001      # Learning rate for optimizer
epochs = 2000            # Number of epochs to run for
es_patience = 100             # Patience fot early stopping
n_units = 128   #number of units of the LSTM cells

#----------------------------------------------------

for train_num in range(6):
    
    ### ATTENTION: The arrangement of nodes can be shuffled between X_train_tens, X_val_tens and X_test_tens!!! Is this a problem???
    X_train_tens = np.load('train_test_data/X_train_tens' + str(train_num) + '.npy')
    X_val_tens = np.load('train_test_data/X_val_tens' + str(train_num) + '.npy') 
    Y_train_tens = np.load('train_test_data/Y_train_tens' + str(train_num) + '.npy')
    Y_val_tens = np.load('train_test_data/Y_val_tens' + str(train_num) + '.npy')   
    
    print('loaded train data from train_num',  train_num)
    
    N = 120
    A = np.eye(N,N) # A does not change 
    
    
    print('X_train_tens.shape:', X_train_tens.shape)
    print('X_val_tens.shape:', X_val_tens.shape)
    
    print('now reduce batch size')
    ### delete reduction later!!!
    # reduce batch size
    X_train_tens = X_train_tens[0:16, :, :, :]
    Y_train_tens = Y_train_tens[0:16, :, :, :]
    X_val_tens = X_val_tens[0:16, :, :, :]
    Y_val_tens = Y_val_tens[0:16, :, :, :]

    print('X_train_tens.shape:', X_train_tens.shape)
    print('Y_train_tens.shape:', Y_train_tens.shape)
    ###
    
    
    
    ### Train the deep learning model ###
    sample_size_train = int(X_train_tens.shape[0]) 
    sample_size_val = int(X_val_tens.shape[0])  
    print('sample_size_train:', sample_size_train) 
    
    timesteps_per_sample = X_train_tens.shape[1] #Number of timesteps in a sample
    N = X_train_tens.shape[2]          # Number of nodes in the graph
    F = X_train_tens.shape[3]          # Original feature dimensionality
    
    print('timesteps_per_sample', timesteps_per_sample)
    print('N', N)
    print('F', F)
    
    if train_num == 0:
        #define necessary functions
        def shape_X1(x):
            print('x.shape:', x.shape)
            return K.reshape(x, (int(x.shape[0])//N, timesteps_per_sample, N, F))
        
        def reshape_X1(x):
            return K.reshape(x, (-1, timesteps_per_sample, F_))
        
        def reshape_X2(x):
            return K.reshape(x, (-1, timesteps_per_sample, 1))
        
        def calc_X2(Y):
            sample_size = Y.shape[0]
            start_slice = np.zeros((1, N, 1))
            X2 = np.zeros((sample_size, timesteps_per_sample, N, 1))
            for sample in range(sample_size):  
                X2[sample, :, :, :] = np.concatenate([start_slice, Y[sample, :-1, :, :]], axis = 0)
            X2 = tf.Variable(X2) 
            return X2
        
        def reshape_skip_LSTM(x):
            return K.reshape(x, (-1, F_))
        
                #define the model
        A_tf = tf.convert_to_tensor(A, dtype=np.float32)
        
        X1_in = Input(batch_shape=(sample_size_train*N, timesteps_per_sample, F))
        
        shaped_X1_in = Lambda(shape_X1)(X1_in)
        
        dropout1 = TimeDistributed(Dropout(dropout_rate))(shaped_X1_in)
        
        graph_attention_1 = TimeDistributed(BatchGraphAttention(width_1gat,
                                           A_tf,
                                           attn_heads=n_attn_heads,
                                           attn_heads_reduction='average',
                                           attn_dropout=attn_dropout,
                                           activation='linear',
                                           kernel_regularizer=l2(l2_reg)),
                                           )(dropout1)
        
        dropout2 = TimeDistributed(Dropout(dropout_rate))(graph_attention_1)
        
        graph_attention_2 = TimeDistributed(BatchGraphAttention(F_,
                                           A_tf,
                                           attn_heads=n_attn_heads,
                                           attn_heads_reduction='average',
                                           attn_dropout=attn_dropout,
                                           activation='linear',
                                           kernel_regularizer=l2(l2_reg)))(dropout2)
        
        dense1 = TimeDistributed(Dense(64, activation = relu))(graph_attention_2)
        
        dropout3 = TimeDistributed(Dropout(dropout_rate))(dense1)
        
        #make sure that the reshape is made correctly!
        encoder_inputs = Lambda(reshape_X1)(dropout3)
        decoder_inputs = LSTM(n_units,
             batch_input_shape=(sample_size_train*N, timesteps_per_sample, F), 
             return_sequences=True)(encoder_inputs)
        decoder_output = AttentionDecoder(n_units, 16)(decoder_inputs) #Attention! 16 output features now!
        
        dense2 = Dense(1, activation = linear)(decoder_output)
        
        model = Model(inputs=X1_in, outputs=dense2) #try to smooth the output with dense layer

        optimizer = Adam(lr=learning_rate)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error',
                      weighted_metrics=['accuracy'])
        model.summary()
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        
    
    ##reshape X and Y tensor for the deep learning model       
    #training
    X1_train = K.reshape(X_train_tens, (sample_size_train*N, timesteps_per_sample, F))
    Y_train = K.reshape(Y_train_tens, (sample_size_train*N, timesteps_per_sample, 1))
    print('X1_train.shape:', X1_train.shape)
    print('Y_train.shape:', Y_train.shape)
    
    #validation
    X1_val = K.reshape(X_val_tens, (sample_size_val*N, timesteps_per_sample, F))
    Y_val = K.reshape(Y_val_tens, (sample_size_val*N, timesteps_per_sample, 1))
    print('X1_val.shape:', X1_val.shape)
    print('Y_val.shape:', Y_val.shape)
       
    A_tf = tf.convert_to_tensor(A, dtype=np.float32)
    validation_data = (X1_val, Y_val)
        
    model.fit(X1_train,
              Y_train,
              epochs=epochs,
              steps_per_epoch = 1,
              validation_data = validation_data,
              validation_steps = 1,
              shuffle=False,  # Shuffling data means shuffling the whole graph
             )
    print('--- model.fit number', train_num, 'finished')

### Predict results ###

X_test_tens = np.load('train_test_data/X_test_tens.npy')
Y_test_tens = np.load('train_test_data/Y_test_tens.npy')
preproccess_end_time = np.load('train_test_data/preproccess_end_time.npy')

with open("train_test_data/order_lanes_test.txt", "rb") as fp:   # Unpickling
    order_lanes_test = pickle.load(fp)

print('loaded test data from train_num')

print('loaded order of lanes:', order_lanes_test)

#reduce batch size
X_test_tens = X_train_tens[0:16, :, :, :]
Y_test_tens = Y_train_tens[0:16, :, :, :]
    
#test
sample_size_test = int(X_test_tens.shape[0])  
X1_test = K.reshape(X_test_tens, (sample_size_test*N, timesteps_per_sample, F))
Y_test = K.reshape(Y_test_tens, (sample_size_test*N, timesteps_per_sample, 1))
print('X1_test.shape:', X1_test.shape)
print('Y_test.shape:', Y_test.shape)

Y_hat = model.predict(X1_test, verbose = 1, steps = 1) 
print('Y_hat.shape:', Y_hat.shape)
prediction = K.reshape(Y_hat, (int(Y_hat.shape[0])//N, timesteps_per_sample, N, 1))
print('prediction.shape:', prediction.shape)

eval_results = model.evaluate(x= X1_test,
                              y=Y_test, 
                              steps=1)

print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))


model.save('models/model_complete.h5')
# serialize model to JSON
model_json = model.to_json()
with open("models/attn_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/attn_model_weights.h5")
print("Saved attn model to disk")


store_predictions_in_df(prediction, order_lanes_test, 200, average_interval, Aeye = True)
