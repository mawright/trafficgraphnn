#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 12:33:24 2018

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

from trafficgraphnn.preprocess_data import PreprocessData, reshape_for_LSTM, reshape_for_GAT

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
from keras.models import load_model



### Configuration for tarining process
train_val_pair = 2
epochs = 2

# total epochs = num_training_steps x train_val_pair x epochs

es_patience = 10   # number of epochs with no improvement after which training will be stopped.
es_callback = EarlyStopping(monitor='val_loss', patience=es_patience, verbose=1)

save_model_steps = 10 #saves the model after several training steps, to see progress in training
cnt_save_steps = 0
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
        
        A = np.load('train_test_data/A0.npy') #A does not change
    
    #fill storage
    X_train_storage[pair_num, :, :, :] = X_train
    X_val_storage[pair_num, :, :, :] = X_val   
    Y_train_storage[pair_num, :, :, :] = Y_train
    Y_val_storage[pair_num, :, :, :] = Y_val
    
#reshape to one big X
X_train_storage = K.reshape(X_train_storage, (train_val_pair * num_samples, num_timesteps, num_lanes, num_features))
X_val_storage = K.reshape(X_val_storage, (train_val_pair * num_samples, num_timesteps, num_lanes, num_features))    
Y_train_storage = K.reshape(Y_train_storage, (train_val_pair * num_samples, num_timesteps, num_lanes, num_targets))
Y_val_storage = K.reshape(Y_val_storage, (train_val_pair * num_samples, num_timesteps, num_lanes, num_targets))

print('X_train_storage.shape:', X_train_storage.shape) #debug
print('Y_train_storage.shape:', Y_train_storage.shape)



X_train = reshape_for_LSTM(X_train_storage)
X_val = reshape_for_LSTM(X_val_storage)
Y_train = reshape_for_LSTM(Y_train_storage)
Y_val = reshape_for_LSTM(Y_val_storage)

####reduce batch size
#X_train = X_train[0:960, :, :]
#X_val = X_val[0:960, :, :]
#Y_train = Y_train[0:960, :, :]
#Y_val = Y_val[0:960, :, :]

print('X_train.shape:', X_train.shape) #debug
print('Y_train.shape:', Y_train.shape)

train_model = define_model(train_val_pair*num_samples, num_timesteps, num_lanes, num_features, A)
validation_data = (X_val, Y_val)
    
train_model.fit(X_train,
          Y_train,
          epochs=epochs,
          steps_per_epoch = train_val_pair, #make as much steps as simulations are available batch_size = total num sampes / steps_per_epoch
          validation_data = validation_data,
          validation_steps = train_val_pair,
          shuffle=False,  # Shuffling data means shuffling the whole graph
          callbacks = [es_callback]
          )

# --------------- predict test data -------------------
### Predict results ###

X_test_tens = tf.convert_to_tensor(np.load('train_test_data/X_test_tens.npy'), dtype=np.float32)
Y_test_tens =  tf.convert_to_tensor(np.load('train_test_data/Y_test_tens.npy'), dtype=np.float32)
average_interval = np.load('train_test_data/average_interval.npy')

with open("train_test_data/order_lanes_test.txt", "rb") as fp:   # Unpickling
    order_lanes_test = pickle.load(fp)
    
X_test = reshape_for_LSTM(X_test_tens)
Y_test = reshape_for_LSTM(Y_test_tens)

#creating new model: Allows us to have diffent batch size that for training
prediction_model = define_model(num_samples, num_timesteps, num_lanes, num_features, A)
old_weights = train_model.get_weights() #copy weights from training model
prediction_model.set_weights(old_weights)

Y_hat = prediction_model.predict(X_test, verbose = 1, steps = 1) 
Y_hat = tf.convert_to_tensor(Y_hat, dtype=np.float32)
prediction = reshape_for_GAT(Y_hat)

store_predictions_in_df(prediction, order_lanes_test, 200, average_interval, alternative_prediction = False) 


#train_model.save('models/train_model_complete_final.h5')
## serialize model to JSON
#model_json = train_model.to_json()
#with open("models/train_model_final.json", "w") as json_file:
#    json_file.write(model_json)
## serialize weights to HDF5
#train_model.save_weights("models/train_model_weights_final.h5")
#print("Saved attn model to disk")
#
#prediction_model.save('models/prediction_model_complete_final.h5')
# #serialize model to JSON
#model_json = prediction_model.to_json()
#with open("models/prediction_model_final.json", "w") as json_file:
#    json_file.write(model_json)
## serialize weights to HDF5
#prediction_model.save_weights("models/prediction_model_weights_final.h5")
#print("Saved attn model to disk")
