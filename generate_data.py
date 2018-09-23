#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 17:04:57 2018

@author: simon
"""
from __future__ import division
from trafficgraphnn.genconfig import ConfigGenerator
from trafficgraphnn.sumo_network import SumoNetwork
import numpy as np

import pickle
import os
from trafficgraphnn.liumethod import LiuEtAlRunner
from trafficgraphnn.preprocess_data import PreprocessData

### Configuration of the Network ###
grid_number = 3 #TODO: make num lanes adjustable
#N = 120 #number of lanes after getting subgraph
grid_length = 600 #meters
num_lanes =3

### Configuration of the Simulation ###
end_time = 1500 #seconds

period_1_2 = 0.4
period_3_4 = 0.4
period_5 = 0.4

period_test = 0.4

binomial = 2
seed = 50
fringe_factor = 1000

### Configuration of Liu estimation ###
use_started_halts = False #use startet halts as ground truth data or maxJamLengthInMeters
show_plot = False
show_infos = False

### Configuration for preprocessing the detector data
average_interval = 10  #Attention -> right now no other average interval than 1 is possible  -> bugfix necessary!
sample_size = 15     #number of steps per sample in size of average interval
interpolate_ground_truth = True #interpolate ground-truth data with np.linspace


number_of_simulations = 2
number_of_simulations_test = 2

# ------------- End of configuration -------------------------

path = 'train_test_data/'
try:  
    os.mkdir(path)
except OSError:  
    print ("Creation of the directory %s failed" % path)
else:  
    print ("Successfully created the directory %s " % path)

### Creating Network and running simulation
config = ConfigGenerator(net_name='test_net')

# Parameters for network, trips and sensors (binomial must be an integer!!!)
config.gen_grid_network(grid_number = grid_number, grid_length = grid_length, num_lanes = num_lanes, simplify_tls = False)
config.gen_rand_trips(period = period_1_2, binomial = binomial, seed = seed, end_time = end_time, fringe_factor = fringe_factor)

config.gen_e1_detectors(distance_to_tls=[5, 125], frequency=1)
config.gen_e2_detectors(distance_to_tls=0, frequency=1)
config.define_tls_output_file()




for train_num in range(number_of_simulations):
    
    if train_num <= 1:
        period = period_1_2
    elif train_num >=2 and train_num <=3:
        period = period_3_4
    else:
        period = period_5
            

    # run the simulation to create output files
    list_X = []
    list_Y = []
    list_A = []
    
    for simu_num in range(2):
        config.gen_rand_trips(period = period, binomial = binomial, seed = seed, end_time = end_time, fringe_factor = fringe_factor)
        sn = SumoNetwork.from_gen_config(config, lanewise=True)
        sn.run()
        print('Simulation run number', simu_num, 'finished')
    
    
        ### Running the Liu Estimation
        #creating liu runner object
        liu_runner = LiuEtAlRunner(sn, store_while_running = True, use_started_halts = use_started_halts, simu_num = simu_num)
        
        # caluclating the maximum number of phases and run the estimation
        max_num_phase = liu_runner.get_max_num_phase(end_time)
        liu_runner.run_up_to_phase(max_num_phase)
        
        # show results for every lane
        liu_runner.plot_results_every_lane(show_plot = show_plot, show_infos = show_infos)
        
        
        ### preprocess data for deep learning model
        preprocess = PreprocessData(sn, simu_num)
        preproccess_end_time = preprocess.get_preprocessing_end_time(liu_runner.get_liu_lane_IDs(), average_interval)
        A, X, Y, order_lanes = preprocess.preprocess_A_X_Y(
                average_interval = average_interval, sample_size = sample_size, start = 200, end = preproccess_end_time, 
                simu_num = simu_num, interpolate_ground_truth = interpolate_ground_truth)
	#NOTE: in this simulation A is an eye matrix    

        list_A.append(A)
        list_X.append(X)
        list_Y.append(Y)
        
        
    ### ATTENTION: The arrangement of nodes can be shuffled between X_train_tens, X_val_tens and X_test_tens!!! Is this a problem???
    X_train_tens = list_X[0]
    X_val_tens = list_X[1]   
    Y_train_tens = list_Y[0]
    Y_val_tens = list_Y[1]    
    A = list_A[0] # A does not change 
    
    #Store X, Y 
    np.save('train_test_data/X_train_tens' + str(train_num) + '.npy', X_train_tens)
    np.save('train_test_data/Y_train_tens' + str(train_num) + '.npy', Y_train_tens)
    np.save('train_test_data/X_val_tens' + str(train_num) + '.npy', X_val_tens)
    np.save('train_test_data/Y_val_tens' + str(train_num) + '.npy', Y_val_tens)
    np.save('train_test_data/A' + str(train_num) + '.npy', A)
    print('save X and Y to npy file for simulation number ', train_num)
    
# --------------------- generating test data -----------------------
for num_test_simu in range(111111, 111111 + number_of_simulations_test):

    #run simulation for generating test data
    config.gen_rand_trips(period = period_test, binomial = binomial, seed = seed, end_time = end_time, fringe_factor = fringe_factor)
    sn = SumoNetwork.from_gen_config(config, lanewise=True)
    sn.run()
    print('Simulation run number', num_test_simu, 'finished')
    
    ### Running the Liu Estimation
    #creating liu runner object
    liu_runner = LiuEtAlRunner(sn, store_while_running = True, use_started_halts = use_started_halts, simu_num = num_test_simu)
    
    # caluclating the maximum number of phases and run the estimation
    max_num_phase = liu_runner.get_max_num_phase(end_time)
    liu_runner.run_up_to_phase(max_num_phase)
    
    # show results for every lane
    liu_runner.plot_results_every_lane(show_plot = show_plot, show_infos = show_infos)
    
    
    ### preprocess data for deep learning model
    preprocess = PreprocessData(sn, num_test_simu)
    preproccess_end_time = preprocess.get_preprocessing_end_time(liu_runner.get_liu_lane_IDs(), average_interval)
    A, X_test_tens, Y_test_tens, order_lanes_test = preprocess.preprocess_A_X_Y(
            average_interval = average_interval, sample_size = sample_size, start = 200, end = preproccess_end_time, 
            simu_num = num_test_simu, interpolate_ground_truth = interpolate_ground_truth)
    
    #--store test data ----------
    np.save('train_test_data/X_test_tens_' + str(num_test_simu) +'.npy', X_test_tens)
    np.save('train_test_data/Y_test_tens' + str(num_test_simu) +'.npy', Y_test_tens)
    np.save('train_test_data/A_test' + str(num_test_simu) +'.npy', A)
    np.save('train_test_data/average_interval' + str(num_test_simu) +'.npy', average_interval)
    
    with open('train_test_data/order_lanes_test' + str(num_test_simu) +'.txt', "wb") as fp:   #Pickling
        pickle.dump(order_lanes_test, fp)
    
    print('save X,Y and order of lanes for testing to npy file for test_simulation number:', num_test_simu)
