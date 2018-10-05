#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:06:33 2018

@author: simon
"""

from trafficgraphnn.class_train_model import TrainModel

#trainer = TrainModel(data_path = 'preprocessed_data/')
trainer = TrainModel()

X_train, Y_train = trainer.build_X_Y(num_simulations = 1, index_start = 0)
X_val, Y_val = trainer.build_X_Y(num_simulations = 1, index_start = 1)

trainer.train_model(X_train, Y_train, X_val, Y_val, 
                   simulations_per_batch = 1, 
                   epochs = 2,
                   es_patience = 25)
trainer.save_train_model()

pred_start_index = 0
num_predictions = 2
end_index = pred_start_index + num_predictions
for pred in range(pred_start_index, end_index):
    X_predict, Y_predict = trainer.build_X_Y(num_simulations = 1, index_start = pred)
    _ = trainer.predict(X_predict, Y_predict, prediction_number = pred)
trainer.save_prediction_model()

#problem with set weights of best model to new prediction model
#solution -> get weight from every layer and reshape it idividually
prediction = trainer.predict(X_predict, Y_predict, 
                             prediction_number = 0, 
                             use_best_model = True,
                             best_model_path = 'savio ws/trained_models/20181004_1_lower_l2/models/')