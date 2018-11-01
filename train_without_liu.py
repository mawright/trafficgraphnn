#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 21:22:43 2018

@author: simon
"""
from numpy.random import seed
seed(3)
from tensorflow import set_random_seed
set_random_seed(44)
print('Set tf seed to 44')

from trafficgraphnn.class_train_model import TrainModel

#trainer = TrainModel(data_path = 'preprocessed_data/')
trainer = TrainModel(data_path = 'data/networks/new_data/preprocessed_data/',
                     multi_gat = True)

X_train, Y_train = trainer.build_X_Y(num_simulations = 1, index_start = 0)
X_val, Y_val = trainer.build_X_Y(num_simulations = 1, index_start = 1)

#-------- Delete liu estimation from X -------------
X_train = X_train[:, :, :, 0:-1]
X_val = X_val[:, :, :, 0:-1]

print('X_train.shape:', X_train.shape)
print('X_val.shape:', X_val.shape)

trainer.reset_num_features(X_val.shape[3])

trainer.train_model_multi_gat(X_train, Y_train, X_val, Y_val, 
                   simulations_per_batch = 1, 
                   epochs = 2,
                   es_patience = 25)

trainer.save_train_model()

pred_start_index = 0
num_predictions = 2
end_index = pred_start_index + num_predictions
for pred in range(pred_start_index, end_index):
    X_predict, Y_predict = trainer.build_X_Y(num_simulations = 1, index_start = pred)
    
    #-------- Delete liu estimation from X -------------
    X_predict = X_predict[:, :, :, 0:-1]
    
    _ = trainer.predict_on_best_model(X_predict, Y_predict, prediction_number = pred)
