#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 17:35:33 2018

@author: simon
"""

from __future__ import division
from trafficgraphnn.genconfig import ConfigGenerator
from trafficgraphnn.sumo_network import SumoNetwork
import numpy as np
import pandas as pd
import math
import pickle
import os
from trafficgraphnn.liumethod import LiuEtAlRunner

from trafficgraphnn.preprocess_data import PreprocessData

import tensorflow as tf
import keras.backend as K
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Input, Dropout, Dense, TimeDistributed, Reshape, Lambda, LSTM
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD, Adagrad
from keras.regularizers import l2
from keras.activations import linear
from trafficgraphnn.batch_graph_attention_layer import  BatchGraphAttention
from keras.utils.vis_utils import plot_model
from trafficgraphnn.attention_decoder import AttentionDecoder
from trafficgraphnn.postprocess_predictions import store_predictions_in_df, resample_predictions
from define_model import define_model


### Configuration for tarining process
train_val_pair = 2
training_mode = 'sequential'   # either sequential or shuffle 
num_training_steps = 2
epochs = 10

# total epochs = num_training_steps x train_val_pair x epochs

es_patience = 50   # number of epochs with no improvement after which training will be stopped.
es_callback = EarlyStopping(monitor='val_loss', patience=es_patience, verbose=1)


#------------- load data -------------
for pair_num in range(train_val_pair):
    X_train = np.load('train_test_data/X_train_tens' + str(pair_num) + '.npy')
    X_val = np.load('train_test_data/X_val_tens' + str(pair_num) + '.npy') 
    Y_train = np.load('train_test_data/Y_train_tens' + str(pair_num) + '.npy')
    Y_val = np.load('train_test_data/Y_val_tens' + str(pair_num) + '.npy')   
    if pair_num == 0:
        
        #create storage
        num_samples = X_train.shape[0]
        num_timesteps = X_train.shape[1]
        num_lanes = X_train.shape[2]
        num_features = X_train.shape[3]
        num_targets = Y_train.shape[3]
        
        X_train_storage = np.zeros((train_val_pair, num_samples, num_timesteps, num_lanes, num_features))
        X_val_storage = np.zeros((train_val_pair, num_samples, num_timesteps, num_lanes, num_features))
        
        Y_train_storage = np.zeros((train_val_pair, num_samples, num_timesteps, num_lanes, num_targets))
        Y_val_storage = np.zeros((train_val_pair, num_samples, num_timesteps, num_lanes, num_targets))
        
        print('X_train_storage.shape:', X_train_storage.shape) #debug
        print('Y_train_storage.shape:', Y_train_storage.shape)
        
        A = np.load('train_test_data/A0.npy') #A does not change
    
    #fill storage
    X_train_storage[pair_num, :, :, :] = X_train
    X_val_storage[pair_num, :, :, :] = X_val
    
    Y_train_storage[pair_num, :, :, :] = Y_train
    Y_val_storage[pair_num, :, :, :] = Y_val
    
# -------------- training model ---------------
model = define_model(num_samples, num_timesteps, num_lanes, num_features, A)

for training_step in range(num_training_steps):
    for pair_num in range(train_val_pair):
        if training_mode == 'sequential':
            X_train_tens = X_train_storage[pair_num, :, :, :]
            X_val_tens = X_val_storage[pair_num, :, :, :]
            
            Y_train_tens = Y_train_storage[pair_num, :, :, :]
            Y_val_tens = Y_val_storage[pair_num, :, :, :]
            
        elif training_mode == 'shuffle':
            num_tain = np.random.randint(0, train_val_pair)
            num_val = np.random.randint(0, train_val_pair)
        
            X_train_tens = X_train_storage[num_tain, :, :, :]
            X_val_tens = X_val_storage[num_val, :, :, :]
            
            Y_train_tens = Y_train_storage[num_tain, :, :, :]
            Y_val_tens = Y_val_storage[num_val, :, :, :]
            
        X_train = K.reshape(X_train_tens, (num_samples*num_lanes, num_timesteps, num_features))
        Y_train = K.reshape(Y_train_tens, (num_samples*num_lanes, num_timesteps, num_targets))
        
        #validation
        X_val = K.reshape(X_val_tens, (num_samples*num_lanes, num_timesteps, num_features))
        Y_val = K.reshape(Y_val_tens, (num_samples*num_lanes, num_timesteps, num_targets))
           
        A_tf = tf.convert_to_tensor(A, dtype=np.float32)
        validation_data = (X_val, Y_val)
            
        model.fit(X_train,
                  Y_train,
                  epochs=epochs,
                  steps_per_epoch = 1,
                  validation_data = validation_data,
                  validation_steps = 1,
                  shuffle=False,  # Shuffling data means shuffling the whole graph
                  callbacks = [es_callback]
                  )
    print('------ training step number', training_step, ' with ', train_val_pair, ' number of training samples finished ---------')
    

# --------------- predict test data -------------------
### Predict results ###

X_test_tens = np.load('train_test_data/X_test_tens.npy')
Y_test_tens = np.load('train_test_data/Y_test_tens.npy')
average_interval = np.load('train_test_data/average_interval.npy')

with open("train_test_data/order_lanes_test.txt", "rb") as fp:   # Unpickling
    order_lanes_test = pickle.load(fp)
    
#test
sample_size_test = int(X_test_tens.shape[0])  
X1_test = K.reshape(X_test_tens, (sample_size_test*num_lanes, num_timesteps, num_features))
Y_test = K.reshape(Y_test_tens, (sample_size_test*num_lanes, num_timesteps, num_targets))

Y_hat = model.predict(X1_test, verbose = 1, steps = 1) 
prediction = K.reshape(Y_hat, (int(Y_hat.shape[0])//num_lanes, num_timesteps, num_lanes, num_targets))

model.save('models/model_complete.h5')
# serialize model to JSON
model_json = model.to_json()
with open("models/attn_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/attn_model_weights.h5")
print("Saved attn model to disk")

store_predictions_in_df(prediction, order_lanes_test, 200, average_interval, alternative_prediction = False) 