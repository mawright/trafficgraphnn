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
from keras.losses import mean_absolute_percentage_error, mean_absolute_error

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
    
#    resampled_ground_truth = K.eval(K.reshape(predictions, (timesteps_per_sample, num_lanes, num_features))) #reshape to (timesteps x lanes x features) bc whe have only one simulation
#    resampled_predictions = K.eval(K.reshape(predictions, (timesteps_per_sample, num_lanes, num_features))) #reshape to (timesteps x lanes x features)
    
#    df_prediction_results = pd.DataFrame(data = resampled_predictions[:,:], 
#                                         index = range(start_time, resampled_predictions.shape[0] * average_interval + start_time, average_interval), 
#                                         columns = order_lanes)
    
    df_prediction_results = pd.DataFrame()
    for lane, index_lane in zip(order_lanes, range(len(order_lanes))):
        iterables = [[lane], ['ground-truth queue', 'prediction queue', 'ground-truth nVehSeen', 'prediction nVehSeen']]
        index = pd.MultiIndex.from_product(iterables, names=['lane', 'values'])

        df_lane = pd.DataFrame(index = range(start_time, timesteps_per_sample * average_interval + start_time, average_interval), 
                               columns = index)
        df_lane.index.name = 'timesteps'
        df_lane[lane, 'ground-truth queue'] = ground_truth[0, :, index_lane, 0]
        df_lane[lane, 'prediction queue'] = predictions[0, :, index_lane, 0]
        df_lane[lane, 'ground-truth nVehSeen'] = ground_truth[0, :, index_lane, 1]
        df_lane[lane, 'prediction nVehSeen'] = predictions[0, :, index_lane, 1]
        
        df_prediction_results = pd.concat([df_prediction_results, df_lane], axis = 1)

    if alternative_prediction == False:
        df_prediction_results.to_hdf(path + 'nn_prediction_results_' + str(simu_num) + '.h5', key = 'prediciton_results')
        print('Stored prediction results in dataframe for simulation_number', simu_num)
    
    if alternative_prediction == True:
        df_prediction_results.to_hdf(path + 'nn_prediction_results_alternative_' + str(simu_num) + '.h5', key = 'prediciton_results')
        print('Stored alternative prediction results in dataframe for simulation_number', simu_num) 
    
    return df_prediction_results
    
def plot_predictions_1_df(df_predictions_1, df_liu_results):
    #list_lanes = df_predictions_1.columns
    list_lanes = df_predictions_1.columns.unique(level = 'lane')
    print(list_lanes)
    time_predictions = df_predictions_1.index.values
    #print('time_predictions:', time_predictions)
    
    plot_lane_list = ['left1to0/1_2']

    list_MAPE_queue = []
    list_MAE_queue = []
    list_MAPE_nVehSeen = []
    list_MAE_nVehSeen = []
    list_MAPE_liu = []
    list_MAE_liu = []

    for lane in list_lanes:
        print('plot for lane:', lane)
        
        #---- plot for queue -------------
        fig = plt.figure()
        fig.set_figheight(3)
        fig.set_figwidth(12)
            
        time_liu_results = df_liu_results.loc[:, (lane, 'phase end')]
        #print('time_liu_results:', time_liu_results)
        
        ground_truth_old, = plt.plot(time_liu_results[:],
                                 df_liu_results.loc[:, (lane, 'ground-truth')],
                                 c='b', label= 'Ground-truth')
        liu_estimation, = plt.plot(time_liu_results[:], 
                                   df_liu_results.loc[:, (lane, 'estimated hybrid')],
                                   c='r', label= 'Liu et al.')
        
        ground_truth_new, = plt.plot(time_predictions, df_predictions_1.loc[:, (lane, 'ground-truth queue')],
                                   c='b', label= 'ground-truth')
        
        dl_prediction_1, = plt.plot(time_predictions, df_predictions_1.loc[:, (lane, 'prediction queue')],
                                   c='g', label= 'DL model')

        plt.legend(handles=[ground_truth_new, liu_estimation, dl_prediction_1], fontsize = 18)
                
        plt.xticks(np.arange(0, 6000, 250))
        plt.xticks(fontsize=18)
        plt.yticks(np.arange(0, 800, 100))
        plt.yticks(fontsize=18)
        plt.xlim(time_predictions[0],time_predictions[-1])
        plt.ylim(0, 300)
        if lane == 'bottom2to2/0_2':
            plt.ylim(0,300)
        if lane == '0/0to1/0_0':
            plt.ylim(0,300)
        
        #TODO: implement background color by using tls data
        
        plt.xlabel('time [s]', fontsize = 18)
        plt.ylabel('queue length [m]', fontsize = 18)
        plt.show()
        
        if lane in plot_lane_list:
            fig.savefig("DL_result_queue_"+ lane.replace('/', '-') +".pdf", bbox_inches='tight')
        
        # ------- plot for nVehSeen ----------------------------------------------------
        fig1 = plt.figure()
        fig1.set_figheight(3)
        fig1.set_figwidth(12)
        
        ground_truth_new, = plt.plot(time_predictions, df_predictions_1.loc[:, (lane, 'ground-truth nVehSeen')],
                                   c='b', label= 'ground-truth')
        
        prediction_nVehSeen, = plt.plot(time_predictions, df_predictions_1.loc[:, (lane, 'prediction nVehSeen')],
                                   c='g', label= 'DL model')
            
        plt.legend(handles=[ground_truth_new, prediction_nVehSeen], fontsize = 18)
                
        plt.xticks(np.arange(0, 6000, 250))
        plt.xticks(fontsize=18)
        plt.yticks(np.arange(0, 800, 10))
        plt.yticks(fontsize=18)
        plt.xlim(time_predictions[0],time_predictions[-1])
        plt.ylim(0, 20)
        
        if lane == 'bottom2to2/0_2':
            plt.ylim(0,35)
        
        #TODO: implement background color by using tls data
        
        plt.xlabel('time [s]', fontsize = 18)
        plt.ylabel('nVehSeen', fontsize = 18)
        plt.show()
        
        if lane in plot_lane_list:
            fig1.savefig("DL_result_nVehSeen_"+ lane.replace('/', '_') +".pdf", bbox_inches='tight')
        
        
        print('MAPE for df_predictions_1')
        MAPE_queue, MAE_queue, MAPE_nVehSeen, MAE_nVehSeen = calc_MAPE_of_predictions(lane, df_predictions_1)
        MAPE_liu, MAE_liu = calc_MAPE_of_liu(lane, df_liu_results)

        list_MAPE_queue.append(MAPE_queue)
        list_MAE_queue.append(MAE_queue)
        list_MAPE_nVehSeen.append(MAPE_nVehSeen)
        list_MAE_nVehSeen.append(MAE_nVehSeen)
        list_MAPE_liu.append(MAPE_liu)
        list_MAE_liu.append(MAE_liu)       
        print('-------------------------------------------------------------------------')
    
    print('Metrics for the entire network:\n\nAverage MAPE_queue:', np.mean(list_MAPE_queue))
    print('Average MAE_queue:', np.mean(list_MAE_queue))  
    print('Average MAPE_nVehSeen:', np.mean(list_MAPE_nVehSeen))  
    print('Average MAE_nVehSeen:', np.mean(list_MAE_nVehSeen))  
    print('Average MAPE_liu:', np.mean(list_MAPE_liu))  
    print('Average MAE_liu:', np.mean(list_MAE_liu))
        
def plot_predictions_2_df(df_predictions_1, df_liu_results, df_predictions_2):
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
        
        ground_truth_old, = plt.plot(time_liu_results[:],
                                 df_liu_results.loc[:, (lane, 'ground-truth')],
                                 c='b', label= 'Ground-truth')
        liu_estimation, = plt.plot(time_liu_results[:], 
                                   df_liu_results.loc[:, (lane, 'estimated hybrid')],
                                   c='r', label= 'Liu et al.')
        
        ground_truth_new, = plt.plot(time_predictions, df_predictions_1.loc[:, (lane, 'ground-truth queue')],
                                   c='g', label= 'ground-truth new')
        
        dl_prediction_1, = plt.plot(time_predictions, df_predictions_1.loc[:, (lane, 'prediction queue')],
                                   c='g', label= 'long time seq')
        

             
        plt.legend(handles=[ground_truth_old, liu_estimation, ground_truth_new, dl_prediction_1], fontsize = 18)
                
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
    ground_truth_queue = np.reshape(ground_truth_queue.values, (ground_truth_queue.values.shape[0]))
    prediction_queue = np.reshape(prediction_queue.values, (prediction_queue.values.shape[0]))
    
    MAPE_queue = K.eval(mean_absolute_percentage_error(ground_truth_queue, prediction_queue))
    print('MAPE queue:', MAPE_queue)
    MAE_queue = K.eval(mean_absolute_error(ground_truth_queue, prediction_queue))
    print('MAE queue:', MAE_queue)
    
    ground_truth_nVehSeen = df_predictions.loc[:, (lane, 'ground-truth nVehSeen')]
    prediction_nVehSeen = df_predictions.loc[:, (lane, 'prediction nVehSeen')]  
    ground_truth_nVehSeen = np.reshape(ground_truth_nVehSeen.values, (ground_truth_nVehSeen.values.shape[0]))
    prediction_nVehSeen = np.reshape(prediction_nVehSeen.values, (prediction_nVehSeen.values.shape[0]))
    
    MAPE_nVehSeen = K.eval(mean_absolute_percentage_error(ground_truth_nVehSeen, prediction_nVehSeen))
    print('MAPE nVehSeen:', MAPE_nVehSeen)
    
    MAE_nVehSeen = K.eval(mean_absolute_error(ground_truth_nVehSeen, prediction_nVehSeen))
    print('MAE nVehSeen:', MAE_nVehSeen)
    
    return MAPE_queue, MAE_queue, MAPE_nVehSeen, MAE_nVehSeen
    
def calc_MAPE_of_liu(lane, df_liu_results):
    ground_truth = df_liu_results.loc[:, (lane, 'ground-truth')]
    liu_estimations = df_liu_results.loc[:, (lane, 'estimated hybrid')]
    ground_truth = np.reshape(ground_truth.values, (ground_truth.values.shape[0]))
    liu_estimations = np.reshape(liu_estimations.values, (liu_estimations.values.shape[0]))
    
    MAPE_liu = K.eval(mean_absolute_percentage_error(ground_truth, liu_estimations))
    print('MAPE liu:', MAPE_liu)
    
    MAE_liu = K.eval(mean_absolute_error(ground_truth, liu_estimations))
    print('MAE liu:', MAE_liu)
    
    return MAPE_liu, MAE_liu
