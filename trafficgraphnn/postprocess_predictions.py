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

def store_predictions_in_df(predictions, order_lanes, start_time, average_interval, alternative_prediction = False):
    resampled_predictions = resample_predictions(predictions)
    resampled_predictions = np.transpose(resampled_predictions)

    df_prediction_results = pd.DataFrame(data = resampled_predictions[:,:], 
                                         index = range(start_time, resampled_predictions.shape[0] * average_interval + start_time, average_interval), 
                                         columns = order_lanes)
    if alternative_prediction == False:
        df_prediction_results.to_hdf('nn_prediction_results.h5', key = 'prediciton_results')
        print('Stored prediction results in dataframe')
    
    if alternative_prediction == True:
        df_prediction_results.to_hdf('nn_prediction_results_alternative.h5', key = 'prediciton_results')
        print('Stored prediction Aeye results in dataframe')        
    
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
        
        dl_prediction_1, = plt.plot(time_predictions, df_predictions_1.loc[:, lane],
                                   c='g', label= 'dl_prediction_1')
        

        #if df_predictions_Aeye == None:
        #    plt.legend(handles=[ground_truth, liu_estimation, dl_prediction], fontsize = 18)
        #else:
        dl_prediction_2, = plt.plot(time_predictions, df_predictions_2.loc[:, lane],
                       c='k', label= 'dl_prediction_2')        
        plt.legend(handles=[ground_truth, liu_estimation, dl_prediction_1, dl_prediction_2], fontsize = 18)
                
        plt.xticks(np.arange(0, 6000, 100))
        plt.xticks(fontsize=18)
        plt.yticks(np.arange(0, 800, 50))
        plt.yticks(fontsize=18)
        plt.xlim(time_predictions[0],time_predictions[-1])
        plt.ylim(0, 200)
        
        #TODO: implement background color by using tls data
        
        plt.xlabel('time [s]', fontsize = 18)
        plt.ylabel('queue length [m]', fontsize = 18)
        plt.show()

    