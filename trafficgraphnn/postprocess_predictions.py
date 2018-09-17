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

def store_predictions_in_df(predictions, order_lanes, df_liu_results, start_time, end_time, average_interval):
    resampled_predictions = resample_predictions(predictions)
    resampled_predictions = np.transpose(resampled_predictions)
    #print('resampled_predictions.shape (transpose):', resampled_predictions.shape)
    #print('resampled_predictions (transpose):', resampled_predictions)
    #list_columns = df_liu_results.columns.unique(level = 'lane')
    #print('list_coumns:', list_columns)
    df_prediction_results = pd.DataFrame(data = resampled_predictions[:,:], 
                                         index = range(start_time, resampled_predictions.shape[0] * average_interval + start_time, average_interval), 
                                         columns = order_lanes)
    df_prediction_results.to_hdf('nn_prediction_results.h5', key = 'prediciton_results')
    print('Stored prediction results in dataframe')
    
def plot_predictions(df_predictions, df_liu_results):
    list_lanes = df_predictions.columns
    time_predictions = df_predictions.index.values
    #print('time_predictions:', time_predictions)

    for lane in list_lanes:
        
        fig = plt.figure()
        fig.set_figheight(5)
        fig.set_figwidth(12)
            
        time_liu_results = df_liu_results.loc[:, (lane, 'phase end')]
        #print('time_liu_results:', time_liu_results)
        
        ground_truth, = plt.plot(time_liu_results[:],
                                 df_liu_results.loc[:, (lane, 'ground-truth')],
                                 c='b', label= 'ground truth')
        liu_estimation, = plt.plot(time_liu_results[:], 
                                   df_liu_results.loc[:, (lane, 'estimated hybrid')],
                                   c='r', label= 'liu')
        
        dl_prediction, = plt.plot(time_predictions, df_predictions.loc[:, lane],
                                   c='g', label= 'deep net')
        
        plt.legend(handles=[ground_truth, liu_estimation, dl_prediction], fontsize = 18)
                
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

    