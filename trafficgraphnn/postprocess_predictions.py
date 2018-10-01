#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 18:09:31 2018

@author: simon
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import keras.backend as K
from keras.losses import mean_absolute_percentage_error

def resample_predictions(predictions):
    sess = tf.InteractiveSession()
    #rearrange samples that we get the dimension [lanesxtimexprediction]
    num_samples = predictions.shape[0]
    timesteps_per_sample = predictions.shape[1]
    num_lanes = predictions.shape[2]
    
    predictions_resampled = np.zeros((num_lanes, num_samples*timesteps_per_sample))
    for lane in range(num_lanes):
        lane_slice = predictions[:, :, lane, 0]
        lane_slice = lane_slice.eval() #convert tensor to np array
        for sample in range(num_samples):
            if sample == 0:
                lane_series = lane_slice[sample, :]
                lane_series = np.reshape(lane_series, (1,timesteps_per_sample))
                #print('sample = 0!!; lane_series.shape:', lane_series.shape)
            else:
                #print('lane_series.shape in first else:', lane_series.shape)
                #print('sample != 0; lane_series_reshaped.shape:', np.reshape(lane_series, (1, lane_series.shape[0])).shape)
                #print('lane_slice[sample, :].shape:', np.reshape(lane_slice[sample, :],(1, 10)).shape)
                lane_series = np.hstack((lane_series, np.reshape(lane_slice[sample, :], (1,timesteps_per_sample))))
                #print('lane_series.shape after hstack:',lane_series.shape)
        #print('lane_series.shape before reshape:', lane_series.shape)
        lane_series = np.reshape(lane_series, (1, num_samples*timesteps_per_sample))
        predictions_resampled[lane, :] = lane_series
        
    print('predictions_resampled.shape:', predictions_resampled.shape)
    return predictions_resampled

def store_predictions_in_df(path, 
                            predictions,
                            ground_truth,
                            order_lanes, 
                            start_time, 
                            average_interval, 
                            simu_num = 0, 
                            alternative_prediction = False):
    #resampled_predictions = resample_predictions(predictions)
    #resampled_predictions = np.transpose(resampled_predictions)
    
    timesteps_per_sample = predictions.shape[1]
    num_lanes = predictions.shape[2]
    num_features = predictions.shape[3]
    
    resampled_predictions = K.eval(K.reshape(predictions, (timesteps_per_sample, num_lanes, num_features))) #reshape to (timesteps x lanes x features)
    
#    df_prediction_results = pd.DataFrame(data = resampled_predictions[:,:], 
#                                         index = range(start_time, resampled_predictions.shape[0] * average_interval + start_time, average_interval), 
#                                         columns = order_lanes)
    
    df_prediction_results = pd.DataFrame()
    for lane, index_lane in zip(order_lanes, range(len(order_lanes))):
        iterables = [[lane], ['ground-truth queue', 'prediction queue', 'ground-truth nVehSeen', 'prediction nVehSeen']]
        index = pd.MultiIndex.from_product(iterables, names=['lane', 'values'])

        df_lane = pd.DataFrame(index = range(start_time, resampled_predictions.shape[0] * average_interval + start_time, average_interval), 
                               columns = index)
        df_lane.index.name = 'timesteps'
        df_lane[lane, 'ground-truth queue'] = ground_truth[1, :, index_lane, 1]
        df_lane[lane, 'prediction queue'] = predictions[1, :, index_lane, 1]
        df_lane[lane, 'ground-truth nVehSeen'] = ground_truth[1, :, index_lane, 2]
        df_lane[lane, 'prediction nVehSeen'] = predictions[1, :, index_lane, 2]
        
        df_prediction_results = pd.concat([df_prediction_results, df_lane], axis = 1)

    if alternative_prediction == False:
        df_prediction_results.to_hdf(path + 'nn_prediction_results_' + str(simu_num) + '.h5', key = 'prediciton_results')
        print('Stored prediction results in dataframe for simulation_number', simu_num)
    
    if alternative_prediction == True:
        df_prediction_results.to_hdf(path + 'nn_prediction_results_alternative_' + str(simu_num) + '.h5', key = 'prediciton_results')
        print('Stored alternative prediction results in dataframe for simulation_number', simu_num)        
    
def plot_predictions(df_predictions_1, df_liu_results, df_predictions_2):
    list_lanes = df_predictions_1.columns
    time_predictions = df_predictions_1.index.values
    #print('time_predictions:', time_predictions)

    for lane in list_lanes:
        print('plot for lane:', lane)
        
        fig = plt.figure()
        fig.set_figheight(5)
        fig.set_figwidth(12)
            
        time_liu_results = df_liu_results.loc[:, (lane, 'phase end')]
        #print('time_liu_results:', time_liu_results)
        
        ground_truth, = plt.plot(time_liu_results[:],
                                 df_liu_results.loc[:, (lane, 'ground-truth')],
                                 c='b', label= 'Ground-truth')
        liu_estimation, = plt.plot(time_liu_results[:], 
                                   df_liu_results.loc[:, (lane, 'estimated hybrid')],
                                   c='r', label= 'Liu et al.')
        
        dl_prediction_1, = plt.plot(time_predictions, df_predictions_1.loc[:, (lane, 'prediction queue')],
                                   c='g', label= 'long time seq')
        

        #if df_predictions_Aeye == None:
        #    plt.legend(handles=[ground_truth, liu_estimation, dl_prediction], fontsize = 18)
        #else:
        dl_prediction_2, = plt.plot(time_predictions, df_predictions_2.loc[:, (lane, 'prediction queue')],
                       c='k', label= 'short time seq')        
        plt.legend(handles=[ground_truth, liu_estimation, dl_prediction_1, dl_prediction_2], fontsize = 18)
                
        plt.xticks(np.arange(0, 6000, 100))
        plt.xticks(fontsize=18)
        plt.yticks(np.arange(0, 800, 50))
        plt.yticks(fontsize=18)
        plt.xlim(time_predictions[0],time_predictions[-1])
        plt.ylim(0, 300)
        
        #TODO: implement background color by using tls data
        
        plt.xlabel('time [s]', fontsize = 18)
        plt.ylabel('queue length [m]', fontsize = 18)
        plt.show()
        
        print('MAPE for df_predictions_1')
        calc_MAPE_of_predictions(lane, df_predictions_1)
        print('MAPE for df_predictions_2')
        calc_MAPE_of_predictions(lane, df_predictions_2)

def calc_MAPE_of_predictions(lane, df_predictions):
    ground_truth_queue = df_predictions.loc[:, (lane, 'ground-truth queue')]
    prediction_queue = df_predictions.loc[:, (lane, 'prediction queue')]    
    MAPE_queue = K.eval(mean_absolute_percentage_error(ground_truth_queue, prediction_queue))
    print('MAPE queue:', MAPE_queue)
    
    ground_truth_nVehSeen = df_predictions.loc[:, (lane, 'ground-truth nVehSeen')]
    prediction_nVehSeen = df_predictions.loc[:, (lane, 'prediction nVehSeen')]    
    MAPE_nVehSeen = K.eval(mean_absolute_percentage_error(ground_truth_nVehSeen, prediction_nVehSeen))
    print('MAPE nVehSeen:', MAPE_nVehSeen)
    