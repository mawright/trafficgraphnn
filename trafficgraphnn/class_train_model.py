#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 14:18:48 2018

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
from define_model_new import define_model
from keras.models import load_model


class TrainModel(object):
    def __init__(self,
            data_path = None
            ):

        if data_path == None:
            config = ConfigGenerator(net_name='test_net2'),
            print(config)
            self.data_path = config[0].get_preprocessed_data_dir() + '/'
        else:
            self.data_path = data_path
            #data_path = 'data_storage/peri_0_4_bin_2_grid_3_len_600_lanes_3_time_1500_simu_50/train_test_data/' #custom location
            #data_path = 'train_test_data/' #custom location

        self.A = np.load(self.data_path +'A.npy') #A does not change
        with open(self.data_path +'order_lanes.txt', "rb") as fp:   # Unpickling
            self.order_lanes = pickle.load(fp)
        self.average_interval = np.load(self.data_path +'average_interval.npy')

    def build_X_Y(self,
                  num_simulations,
                  index_start,
                  ):

        index_end = index_start + num_simulations
        for simu_num, index in zip(range(index_start, index_end), range(num_simulations)):
            X_simu = np.load(self.data_path +'X_' + str(simu_num) + '.npy')
            Y_simu = np.load(self.data_path +'Y_' + str(simu_num) + '.npy')
            if simu_num == index_start:
                #create storage
                self.num_samples = X_simu.shape[0]
                self.num_timesteps = X_simu.shape[1]
                self.num_lanes = X_simu.shape[2]
                self.num_features = X_simu.shape[3]
                self.num_targets = Y_simu.shape[3]

                print('num_samples:', self.num_samples)
                print('num_timesteps:', self.num_timesteps)
                print('num_lanes:', self.num_lanes)
                print('num_features:', self.num_features)
                print('num_targets:', self.num_targets)

                X = np.zeros((num_simulations, self.num_timesteps, self.num_lanes, self.num_features))
                Y = np.zeros((num_simulations, self.num_timesteps, self.num_lanes, self.num_targets))

            #fill big X matrix
            X[index, :, :, :] = X_simu
            Y[index, :, :, :] = Y_simu
            print('X.shape:', X.shape) #debug
            print('Y.shape:', Y.shape)
        return X, Y


    def train_model(self,
                    X_train,
                    Y_train,
                    X_val,
                    Y_val,
                    simulations_per_batch = 1,
                    epochs = 1,
                    es_patience = 10):

        num_simulations = X_train.shape[0] #total number of samples in whole dataset
        assert num_simulations == Y_train.shape[0]
        assert num_simulations == X_val.shape[0]
        assert num_simulations == Y_val.shape[0]

        num_batches = num_simulations//simulations_per_batch
        print('num_batches:', num_batches)

        self.train_model = define_model(simulations_per_batch,
                                       self.num_timesteps,
                                       self.num_lanes,
                                       self.num_features,
                                       self.A)
        
        A_stack = self.stack_A(num_simulations)       

        validation_data = ([X_val, A_stack], Y_val)
           # number of epochs with no improvement after which training will be stopped.
        self.es_callback = EarlyStopping(monitor='val_loss', patience=es_patience, verbose=1)
        self.train_model.fit([X_train, A_stack],
                  Y_train,
                  epochs=epochs,
                  batch_size = simulations_per_batch,
                  validation_data = validation_data,
                  shuffle=False,
                  callbacks = [self.es_callback]
                  )
        print('Finished training of model')


    def save_model(self, path = 'models/'):
        try:
            os.mkdir(path)
            print ("Successfully created the directory %s " % path)
        except OSError:
            print ("Creation of the directory %s failed" % path)

        self.train_model.save(path + 'train_model_complete_final.h5')
        model_yaml = self.train_model.to_yaml()
        with open(path + "model.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
        # serialize weights to HDF5
        self.train_model.save_weights(path + "model_weights.h5")
        print("Saved model to disk")


    def predict(self, X_predict, Y_ground_truth, prediction_number):

        assert X_predict.shape[0] == Y_ground_truth.shape[0]
        self.prediction_model = define_model(X_predict.shape[0],
                                        self.num_timesteps,
                                        self.num_lanes,
                                        self.num_features,
                                        self.A)
        
        A_stack = self.stack_A(1)
        train_weights = self.train_model.get_weights() #copy weights from training model
        self.prediction_model.set_weights(train_weights)
        Y_hat = self.prediction_model.predict([X_predict, A_stack], verbose = 1, steps = 1)
#        prediction = tf.convert_to_tensor(Y_hat, dtype=np.float32)

        store_predictions_in_df(self.data_path,
                                Y_hat,
                                Y_ground_truth,
                                self.order_lanes,
                                200,
                                self.average_interval,
                                simu_num = prediction_number,
                                alternative_prediction = False)

        eval_results = self.prediction_model.evaluate([X_predict, A_stack],
                                      Y_ground_truth,
                                      batch_size=1,
                                      verbose=1)
        print('Done.\n'
              'Test loss: {}\n'
              'Test mape: {}'.format(*eval_results))
        
        return Y_hat
    
    def stack_A(self, num_simulations):
        A = self.A.astype(np.float32)
        A_list = [A for _ in range(self.num_timesteps)]
        A_stack_time = np.stack(A_list, axis=0)
        print('A_stack_time.shape ', A_stack_time.shape)
        A_list2 = [A_stack_time for _ in range(num_simulations)]
        A_stack_simu = np.stack(A_list2, axis=0)
        print('A_stack_simu.shape after reshape', A_stack_simu.shape)
        return A_stack_simu

