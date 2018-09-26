#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 19:00:22 2018

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

from trafficgraphnn.preprocess_data import PreprocessData, reshape_for_3Dim, reshape_for_4Dim

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

### Configuration for training process
train_val_pair = 2
epochs = 2
simulations_per_batch = 1 #each batch has data from 'batch_size_in_simulations' simulations

num_predictions = 1

config = ConfigGenerator(net_name='test_net')
data_path = config.get_preprocessed_data_dir() + '/' #location where files from generate_data.py are stored

#data_path = 'data_storage/peri_0_4_bin_2_grid_3_len_600_lanes_3_time_1500_simu_50/train_test_data/' #custom location

es_patience = 10   # number of epochs with no improvement after which training will be stopped.
es_callback = EarlyStopping(monitor='val_loss', patience=es_patience, verbose=1)

#------------- load data -------------
for pair_num in range(train_val_pair):
    X_train = np.load(data_path +'X_train_tens_' + str(pair_num) + '.npy')
    X_val = np.load(data_path +'X_val_tens_' + str(pair_num) + '.npy') 
    Y_train = np.load(data_path +'Y_train_tens_' + str(pair_num) + '.npy')
    Y_val = np.load(data_path +'Y_val_tens_' + str(pair_num) + '.npy')   
    if pair_num == 0:
        
        #create storage
        num_samples = X_train.shape[0]
        num_timesteps = X_train.shape[1]
        num_lanes = X_train.shape[2]
        num_features = X_train.shape[3]
        num_targets = Y_train.shape[3]
        
        print('num_samples:', num_samples)
        print('num_timesteps:', num_timesteps)
        print('num_lanes:', num_lanes)
        print('num_features:', num_features)
        print('num_targets:', num_targets)
        
        X_train_storage = np.zeros((train_val_pair, num_timesteps, num_lanes, num_features))
        X_val_storage = np.zeros((train_val_pair, num_timesteps, num_lanes, num_features))
        Y_train_storage = np.zeros((train_val_pair, num_timesteps, num_lanes, num_targets))
        Y_val_storage = np.zeros((train_val_pair, num_timesteps, num_lanes, num_targets))
        
        A = np.load(data_path +'A_0.npy') #A does not change
    
    #fill storage
    X_train_storage[pair_num, :, :, :] = X_train
    X_val_storage[pair_num, :, :, :] = X_val   
    Y_train_storage[pair_num, :, :, :] = Y_train
    Y_val_storage[pair_num, :, :, :] = Y_val
    

print('X_train_storage.shape:', X_train_storage.shape) #debug
print('Y_train_storage.shape:', Y_train_storage.shape)

dataset_size = train_val_pair #total number of samples in whole dataset
batch_size_in_samples = simulations_per_batch #number of samples for one batch
num_batches = dataset_size//batch_size_in_samples

print('dataset_size:', dataset_size)
print('batch_size_in_samples:', batch_size_in_samples)
print('num_batches:', num_batches)

train_model = define_model(batch_size_in_samples, num_timesteps, num_lanes, num_features, A, save_model = True)

validation_data = (X_val_storage, Y_val_storage)
    
train_model.fit(X_train_storage,
          Y_train_storage,
          epochs=epochs,
          batch_size = 2,
#          steps_per_epoch = train_val_pair, #make as much steps as simulations are available batch_size = total num sampes / steps_per_epoch
          validation_data = validation_data,
#          validation_steps = train_val_pair,
          shuffle=False,  # Shuffling data means shuffling the whole graph
          callbacks = [es_callback]
          )

train_model.save('models/train_model_complete_final.h5')
model_yaml = train_model.to_yaml()
with open("models/model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
train_model.save_weights("models/model_weights.h5")
print("Saved model to disk")

# --------------- predict test data -------------------
### Predict results ###

curr_prediction = 0
X_test_tens = tf.convert_to_tensor(np.load(data_path +'X_test_tens_' + str(curr_prediction) +'.npy'), dtype=np.float32)
Y_test_tens =  tf.convert_to_tensor(np.load(data_path +'Y_test_tens_' + str(curr_prediction) +'.npy'), dtype=np.float32)
average_interval = np.load(data_path +'average_interval_' + str(curr_prediction) +'.npy')

with open(data_path +'order_lanes_test_' + str(curr_prediction) + '.txt', "rb") as fp:   # Unpickling
    order_lanes_test = pickle.load(fp)
    
#X_test = reshape_for_3Dim(X_test_tens)
#Y_test = reshape_for_3Dim(Y_test_tens)



#    if curr_prediction == 0: #creating model for first time
#        #creating new model: Allows us to have diffent batch size that for training
#        prediction_model = define_model(int(X_test.shape[0]), num_timesteps, num_lanes, num_features, A)
#        
#        old_weights = train_model.get_weights() #copy weights from training model
#        prediction_model.set_weights(old_weights)

#alternative prediction!!!
prediction_model = define_model(num_samples, num_timesteps, num_lanes, num_features, A)
old_weights = train_model.get_weights() #copy weights from training model
prediction_model.set_weights(old_weights)

Y_hat = prediction_model.predict(X_test_tens, verbose = 1, steps = 1) 
prediction = tf.convert_to_tensor(Y_hat, dtype=np.float32)
store_predictions_in_df(data_path, prediction, order_lanes_test, 200, average_interval, simu_num = curr_prediction, alternative_prediction = True) 


#regular prediction
Y_hat = train_model.predict(X_test_tens, verbose = 1, steps = 1) 
prediction = tf.convert_to_tensor(Y_hat, dtype=np.float32)
store_predictions_in_df(data_path, prediction, order_lanes_test, 200, average_interval, simu_num = curr_prediction, alternative_prediction = False) 


#train_model.save('models/train_model_complete_final.h5')
# serialize model to JSON
#model_json = train_model.to_json()
#with open("models/train_model_final.json", "w") as json_file:
#    json_file.write(model_json)
## serialize weights to HDF5
#train_model.save_weights("models/train_model_weights_final.h5")

##serialize model to YAML


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
