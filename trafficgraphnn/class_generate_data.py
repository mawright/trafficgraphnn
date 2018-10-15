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

class GenerateData(object):
    def __init__(self,
                number_of_simulations = 2,
                order_of_lanes = None,
                #ATTENTION! When start_index != 0, no A matrix is saved! start_index 
                #is just to extend previous simulation runs with the same orderof lanes!
                start_index = 0,  #index where names of the generation should start
                grid_number = 3, #TODO: make num lanes adjustable
                grid_length = 600, 
                num_lanes =3,
                
                end_time = 1400, #seconds                
                period_lower_bound = 0.4, #lower bound for randomized period
                period_upper_bound = 0.4, #upper bound for randomized period

                binomial = 2,
                seed = 50,
                fringe_factor = 1000,
           
                ### Configuration of Liu estimation ###
                use_started_halts = False, #use startet halts as ground truth data or maxJamLengthInMeters
                show_plot = False,
                show_infos = False,
                
                ### Configuration for preprocessing the detector data
                average_interval = 10,
                sample_time_sequence = False,
                sample_size = 15 ,    #number of steps per sample in size of average interval; only neccesary if 
                interpolate_ground_truth = True #interpolate ground-truth data with np.linspace
                ):

        self.number_of_simulations = number_of_simulations
        self.order_of_lanes = order_of_lanes
        self.start_index = start_index
        self.grid_number = grid_number
        self.grid_length = grid_length
        self.num_lanes = num_lanes
        self.end_time = end_time
        self.period_lower_bound = period_lower_bound
        self.period_upper_bound = period_upper_bound
        self.binomial = binomial
        self.seed = seed
        self.fringe_factor = fringe_factor
        self.use_started_halts = use_started_halts
        self.show_plot = show_plot
        self.show_infos = show_infos
        self.average_interval = average_interval
        self.sample_time_sequence = sample_time_sequence
        self.sample_size = sample_size
        self.interpolate_ground_truth = interpolate_ground_truth

    def generate_X_Y_A(self):
        ### Creating Network and running simulation
        config = ConfigGenerator(net_name='new_data')
        path = config.get_preprocessed_data_dir()
        
        # Parameters for network, trips and sensors (binomial must be an integer!!!)
        config.gen_grid_network(grid_number = self.grid_number, 
                                grid_length = self.grid_length, 
                                num_lanes = self.num_lanes, 
                                simplify_tls = False)
        config.gen_rand_trips(period = self.period_lower_bound, 
                              binomial = self.binomial, 
                              seed = self.seed, 
                              end_time = self.end_time, 
                              fringe_factor = self.fringe_factor)
        
        config.gen_e1_detectors(distance_to_tls=[5, 125], frequency=1)
        config.gen_e2_detectors(distance_to_tls=0, frequency=1)
        config.define_tls_output_file()
        
        try:  
            os.mkdir(path)
        except OSError:  
            print ("Creation of the directory %s failed" % path)
        else:  
            print ("Successfully created the directory %s " % path)
        
        end = self.start_index + self.number_of_simulations
        for simu_num in range(self.start_index, end):            
            self.seed += 1
            #generate random period (traffic arrival rate)
            period = np.random.uniform(low = self.period_lower_bound,
                                       high = self.period_upper_bound)
                    
            config.gen_rand_trips(period = period, binomial = self.binomial, 
                                  seed = self.seed, end_time = self.end_time, 
                                  fringe_factor = self.fringe_factor)
            
            sn = SumoNetwork.from_gen_config(config, lanewise=True)
            sn.run()
            print('Simulation run number', simu_num, 'finished')

            ### Running the Liu Estimation
            #creating liu runner object
            liu_runner = LiuEtAlRunner(sn, 
                                       store_while_running = True, 
                                       use_started_halts = self.use_started_halts, 
                                       simu_num = simu_num)
            
            # caluclating the maximum number of phases and run the estimation
            max_num_phase = liu_runner.get_max_num_phase(self.end_time)
            liu_runner.run_up_to_phase(max_num_phase)
            
            # show results for every lane
            liu_runner.plot_results_every_lane(show_plot = self.show_plot, 
                                               show_infos = self.show_infos)

            ### preprocess data for deep learning model
            preprocess = PreprocessData(sn, simu_num)
            preproccess_end_time = preprocess.get_preprocessing_end_time(liu_runner.get_liu_lane_IDs(), 
                                                                         self.average_interval)
            A_list, X, Y, order_lanes = preprocess.preprocess_A_X_Y(
                    average_interval = self.average_interval, 
                    sample_size = self.sample_size, 
                    start = 200, 
                    end = preproccess_end_time, 
                    simu_num = simu_num, 
                    interpolate_ground_truth = self.interpolate_ground_truth,
                    sample_time_sequence = self.sample_time_sequence,
                    ord_lanes = self.order_of_lanes)
            
            #Store X, Y 
            np.save(path + '/X_' + str(simu_num) + '.npy', X)
            np.save(path + '/Y_' + str(simu_num) + '.npy', Y)
            print('save X and Y to npy file for simulation number ', simu_num)
            
            if simu_num == 0: #only for the very first simulation; pay attention to self.start_index
                np.save(path + '/A_downstream.npy', A_list[0])
                np.save(path + '/A_upstream.npy', A_list[1])
                np.save(path + '/A_neighbors.npy', A_list[2])
                np.save(path + '/average_interval.npy', self.average_interval)
                with open(path + '/order_lanes.txt', "wb") as fp:   #Pickling
                    pickle.dump(order_lanes, fp)
                print('Save A, average interval and order of lanes for testing to npy file')
                
    def get_order_of_lanes(self, data_path):
        with open(data_path +'order_lanes.txt', "rb") as fp:   # Unpickling
            order_lanes = pickle.load(fp)
            return order_lanes
        
    def set_order_of_lanes(self, data_path):   
        self.order_of_lanes = self.get_order_of_lanes(data_path)
        print('Set order of lanes')
            
