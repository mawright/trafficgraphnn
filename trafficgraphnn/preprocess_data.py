#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 16:21:00 2018

@author: simon
"""
import logging
import os
import sys
import xml.etree.cElementTree as et
import networkx as nx
import math

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from trafficgraphnn.utils import iterfy
from trafficgraphnn.get_tls_data import get_tls_data
from trafficgraphnn.sumo_network import SumoNetwork


_logger = logging.getLogger(__name__)

class PreprocessData(object):
    def __init__(self, sumo_network, simu_num):
        self.sumo_network = sumo_network
        self.graph = self.sumo_network.get_graph()
        self.liu_results_path = os.path.join(os.path.dirname(
                self.sumo_network.netfile), 'liu_estimation_results'+ str(simu_num) + '.h5')
        self.detector_data_path =  os.path.join(os.path.dirname(
                self.sumo_network.netfile), 'output')
        self.df_liu_results = pd.DataFrame()
        self.simu_num = simu_num
        
        try:
            self.df_liu_results = pd.read_hdf(self.liu_results_path)
        except IOError:
            print('An error occured trying to read the hdf file.')
        
        self.parsed_xml_tls = None
       
    def get_liu_results(self, lane_ID, phase):
        series_results = self.df_liu_results.loc[phase, lane_ID]        
        # return series with parameter time, ground-truth,
        # estimation hybrid/ pure liu
        return series_results
        
    def get_phase_for_time(self, lane_ID, point_of_time):
        #this code should give the phase for a specific point of time and lane back
        time_lane = self.df_liu_results.loc[:, (lane_ID, 'time')] 
               
        for phase in range(1, len(time_lane)):
            phase_start = self.df_liu_results.loc[phase, (lane_ID, 'phase start')]
            phase_end = self.df_liu_results.loc[phase, (lane_ID, 'phase end')]
            if point_of_time >= phase_start and point_of_time < phase_end:
                return phase
            
        print('Point of time out of time window!')
        return np.nan
        
    def preprocess_detector_data(self, average_intervals, num_index=0,  average_over_cycle = False):
        #code that is averaging over specific intervals 
        #(eg. 5s, 15s, 30s; given in a list) and an option to average 
        #over a whole cycle. For each averaging interval a new .h5 file is created
        
        file_list = os.listdir(self.detector_data_path)
        str_e1 = 'e1'
        str_tls = 'tls_output.xml'
       
        for interval in average_intervals:
            print('processing interval ', interval)
            self.df_interval_results = pd.DataFrame() #reset df for every interval
            
            for filename in file_list:
                if str_e1 in filename:                    
                    df_detector = self.process_e1_data(filename, interval)
                    self.df_interval_results = pd.concat(
                            [self.df_interval_results, df_detector], axis = 1)  
                                  
            self.df_interval_results.to_hdf(os.path.join(os.path.dirname(self.sumo_network.netfile),
                            'e1_detector_data_'+ str(interval) + '_seconds'+ str(num_index) + '.h5'),
                            key = 'preprocessed e1_detector_data')
                    
        if average_over_cycle == True:
            self.df_interval_results = pd.DataFrame() #reset df for every interval
            for filename in file_list:
                if str_tls in filename:
                    filename_tls = filename
            
            for filename in file_list:
                if str_e1 in filename:
                    lane_id = self.get_lane_id_from_filename(filename)             
                    phase_length, phase_start = self.calc_tls_data(lane_id, filename_tls)
#                    print('lane_id:', lane_id)
#                    print('phase_length:', phase_length)
#                    print('phase_start:', phase_start)
                    df_detector = self.process_e1_data(filename, phase_length, 
                                start_time = phase_start,  average_over_cycle = True)
                    self.df_interval_results = pd.concat(
                            [self.df_interval_results, df_detector], axis = 1)
            self.df_interval_results.to_hdf(os.path.join(os.path.dirname(self.sumo_network.netfile),
                'e1_detector_data_tls_interval' + str(num_index) + '.h5'),
                key = 'preprocessed e1_detector_data')         
               
    def process_e1_data(self, filename, interval, start_time = 0,  average_over_cycle = False):
        append_cnt=0    #counts how often new data were appended
        phase_cnt=0 #counts the phases
        list_nVehContrib = []
        list_flow = []
        list_occupancy = []
        list_speed = []
        list_length = []
        
        memory_nVehContrib = []
        memory_flow = []
        memory_occupancy = []
        memory_speed = []
        memory_length = []
        memory_end_time = []
        memory_phases = []
        memory_phase_start = []
        memory_phase_end = []
        
        df_detector = pd.DataFrame()
        
        for event, elem in et.iterparse(os.path.join(self.detector_data_path, filename)):
            
            if elem.tag == 'interval' and int(float(elem.attrib.get('end'))) >= start_time:
            
                list_nVehContrib.append(float(elem.attrib.get('nVehContrib')))
                list_flow.append(float(elem.attrib.get('flow')))
                list_occupancy.append(float(elem.attrib.get('occupancy')))
                list_speed.append(float(elem.attrib.get('speed')))
                list_length.append(float(elem.attrib.get('length')))            
                append_cnt += 1
                
                if append_cnt == interval:
                    
                    append_cnt = 0
                    phase_cnt += 1
                    #process data
                    memory_nVehContrib.append(sum(list_nVehContrib))
                    memory_flow.append(sum(list_flow)/interval)                    
                    memory_occupancy.append(sum(list_occupancy)/interval)
                    
                    cnt_veh = 0
                    sum_speed = 0
                    sum_length = 0
                    for speed, length in zip(list_speed, list_length):
                        if speed != -1: #when speed != -1 is also length != -1
                            sum_speed = sum_speed + speed
                            sum_length = sum_length + length
                            cnt_veh += 1
                    if cnt_veh > 0:
                        memory_speed.append(sum_speed/cnt_veh)
                        memory_length.append(sum_length/cnt_veh)
                    else:
                        memory_speed.append(-1)
                        memory_length.append(-1)                        
                        
                    #append data to dataframe          
                    memory_end_time.append(int(float(elem.attrib.get('end'))))
                    memory_phases.append(phase_cnt)
                    detector_id = elem.attrib.get('id')
                    if average_over_cycle == True:
                        memory_phase_start.append(start_time + phase_cnt*interval)
                        memory_phase_end.append(start_time + (phase_cnt+1)*interval)
                    
                    
                    #reset temporary lists
                    list_nVehContrib = []
                    list_flow = []
                    list_occupancy = []
                    list_speed = []
                    list_length = []
                                
        iterables = [[detector_id], [
                'nVehContrib', 'flow', 'occupancy', 'speed', 'length']]
        index = pd.MultiIndex.from_product(iterables, names=['detector ID', 'values'])
        if average_over_cycle == False:
            df_detector = pd.DataFrame(index = [memory_end_time], columns = index)
        else:
            df_detector = pd.DataFrame(index = [memory_phases], columns = index)
        df_detector.index.name = 'end time'
        
        # fill interval_series with values
        df_detector[detector_id, 'nVehContrib'] = memory_nVehContrib
        df_detector[detector_id, 'flow'] = memory_flow
        df_detector[detector_id, 'occupancy'] = memory_occupancy
        df_detector[detector_id, 'speed'] = memory_speed
        df_detector[detector_id, 'length'] = memory_length
        if average_over_cycle == True:
            df_detector[detector_id, 'phase start'] = memory_phase_start
            df_detector[detector_id, 'phase end'] = memory_phase_end
       
        return df_detector 

    def calc_tls_data(self, lane_id, filename_tls):
        
        if self.parsed_xml_tls == None:
            try:
                self.parsed_xml_tls = et.parse(
                        os.path.join(self.detector_data_path, filename_tls))
            except:
                IOError('Could not load tls_xml_file')    
                
        phase_start, phase_length, _ = get_tls_data(self.parsed_xml_tls, lane_id)

        return phase_length, phase_start
        

    def get_lane_id_from_filename(self, filename):
        search_str_start = 'e1_'
        search_str_end = '_output.xml'
        index_start = filename.find(search_str_start) + 3
        index_end = filename.find(search_str_end) - 2
        lane_id = filename[index_start:index_end]
        return lane_id.replace('-', '/')
    
    def preprocess_train_val_test(self, 
                           average_interval = 1, 
                           sample_size = 10,
                           train_start = 200, 
                           train_end = 700, 
                           test_start = 700, 
                           test_end = 1200, 
                           val_start = 1200, 
                           val_end = 1700):
        """
        -Returns the adjacency matrix in SciPy sparse matrix.
        -Returns X_train, X_test and X_val, that are all differrent time series 
        of an interval of all advanced e1 detectors in the network
        -Returns Y_train, Y_test, Y_val, that are the appropriate ground-truth 
        length data for each lane        
        """       
        if not os.path.isfile(os.path.join(os.path.dirname(self.sumo_network.netfile),
            'e1_detector_data_'+ str(average_interval) + '_seconds.h5')):
                self.preprocess_detector_data([average_interval])
        
        df_detector = pd.read_hdf(os.path.join(os.path.dirname(self.sumo_network.netfile),
            'e1_detector_data_' + str(average_interval) + '_seconds.h5'))
        
        try: 
            df_liu_results = pd.read_hdf(os.path.join(os.path.dirname(self.sumo_network.netfile),
                    'liu_estimation_results' + str(simu_num) + '.h5'))
        except IOError:
            print('No file for liu estimation results found.')
            
        subgraph = self.get_subgraph(self.graph)
        
        X_train, Y_train = self.calc_X_and_Y(subgraph, df_detector, df_liu_results,
                            train_start, train_end, average_interval, sample_size)
        X_test, Y_test = self.calc_X_and_Y(subgraph, df_detector, df_liu_results,
                            test_start, test_end, average_interval, sample_size)
        X_val, Y_val = self.calc_X_and_Y(subgraph, df_detector, df_liu_results,
                            val_start, val_end, average_interval, sample_size)
       
        A = nx.adjacency_matrix(subgraph)
        
        return A, X_train, Y_train, X_test, Y_test, X_val, Y_val
    
    
    
    def preprocess_X_Y(self,
                       average_interval =1, 
                       sample_size = 10, 
                       start = 200, 
                       end = 2000, 
                       simu_num = 0):
        
        #if not os.path.isfile(os.path.join(os.path.dirname(self.sumo_network.netfile),
        #                                   'e1_detector_data_'+ str(average_interval) + '_seconds' + str(simu_num) + '.h5')):
        self.preprocess_detector_data([average_interval], num_index = simu_num)
        
        df_detector = pd.read_hdf(os.path.join(os.path.dirname(self.sumo_network.netfile),
                                               'e1_detector_data_' + str(average_interval) + '_seconds' + str(simu_num) + '.h5'))
 
        try: 
            df_liu_results = pd.read_hdf(os.path.join(os.path.dirname(self.sumo_network.netfile),
                    'liu_estimation_results' + str(simu_num) + '.h5'))
        except IOError:
            print('No file for liu estimation results found.') 
            
        subgraph = self.get_subgraph(self.graph)
        
        X, Y = self.calc_X_and_Y(subgraph, df_detector, df_liu_results,
                            start, end, average_interval, sample_size)
        
        A = nx.adjacency_matrix(subgraph)
        
        return A, X, Y

         
    def calc_X_and_Y(self, subgraph, df_detector, df_liu_results, start_time, end_time, average_interval, sample_size):
        #X = tf.Variable([[len(subgraph.nodes)], [8], [end_time-start_time]])
        #X = tf.Variable([])
        #X_tensor = np.array([]) # 3D array ->later convert with tf.convert_to_tensor
        #Y = np.array([])
        
        #dimesion: time x nodes x features
        X_unsampled = np.zeros((int((end_time-start_time)/average_interval), len(subgraph.nodes), 8))
        Y_unsampled = np.zeros((int((end_time-start_time)/average_interval), len(subgraph.nodes), 1))
        
        for lane, cnt_lane in zip(subgraph.nodes, range(len(subgraph.nodes))): 
            # faster solution: just access df once and store data in arrays or lists
            lane_id = lane.replace('/', '-')
            stopbar_detector_id = 'e1_' + str(lane_id) + '_0'
            adv_detector_id = 'e1_' + str(lane_id) + '_1'      
        
            X_unsampled[:, cnt_lane, 0] = df_detector[stopbar_detector_id]['nVehContrib'][start_time+1:end_time]
            X_unsampled[:, cnt_lane, 1] = df_detector[stopbar_detector_id]['flow'][start_time+1:end_time]
            X_unsampled[:, cnt_lane, 2] = df_detector[stopbar_detector_id]['occupancy'][start_time+1:end_time]
            X_unsampled[:, cnt_lane, 3] = df_detector[stopbar_detector_id]['speed'][start_time+1:end_time]
            X_unsampled[:, cnt_lane, 4] = df_detector[adv_detector_id]['nVehContrib'][start_time+1:end_time]
            X_unsampled[:, cnt_lane, 5] = df_detector[adv_detector_id]['flow'][start_time+1:end_time]
            X_unsampled[:, cnt_lane, 6] = df_detector[adv_detector_id]['occupancy'][start_time+1:end_time]
            X_unsampled[:, cnt_lane, 7] = df_detector[adv_detector_id]['speed'][start_time+1:end_time]
                  
            Y_unsampled[:, cnt_lane, 0] = self.get_ground_truth_array(start_time, end_time, lane, average_interval)

        #print('X_unsampled:', X_unsampled)
        #print('Y_unsampled:', Y_unsampled)
        
        #sample X_unsampled and Y_unsampled 
        #sample_size is the NUMBER of time steps we want to sample
        #math.ceil(float(X_unsampled.size[0]/sample_size))
        num_samples = math.ceil(X_unsampled.shape[0]/sample_size)
        print('number of samples:', num_samples)
        X_sampled = np.zeros((num_samples, sample_size, len(subgraph.nodes), 8))
        Y_sampled = np.zeros((num_samples, sample_size, len(subgraph.nodes), 1))
        for cnt_sample in range(num_samples):
            sample_start = cnt_sample*sample_size
            sample_end = sample_start + sample_size
            
            if X_unsampled[sample_start:sample_end][:][:].shape[0] == sample_size:                
                X_sampled[cnt_sample][:][:][:] = X_unsampled[sample_start:sample_end][:][:] 
                Y_sampled[cnt_sample][:][:][:] = Y_unsampled[sample_start:sample_end][:][:]
            else:
                # implement zero padding
                diff_samples = sample_size - X_unsampled[sample_start:sample_end][:][:].shape[0]
                X_zero_padding = np.vstack((X_unsampled[sample_start:sample_end][:][:],
                                           np.zeros((diff_samples, len(subgraph.nodes), 8))))
                Y_zero_padding = np.vstack((Y_unsampled[sample_start:sample_end][:][:],
                           np.zeros((diff_samples, len(subgraph.nodes), 1))))
                X_sampled[cnt_sample][:][:][:] = X_zero_padding
                Y_sampled[cnt_sample][:][:][:] = Y_zero_padding
            
        #print('X_sampled:', X_sampled)
        #print('Y_sampled:', Y_sampled)

        return X_sampled, Y_sampled
    
    def get_subgraph(self, graph):       
        node_sublist = [lane[0] for lane in self.graph.nodes(data=True) if lane[1] != {}]
        #print('node_sublist:', node_sublist)
        sub_graph = graph.subgraph(node_sublist)
        return sub_graph
    
    def get_ground_truth_array(self, start_time, end_time, lane_id, average_interval):
        #ATTENTION!!! What to do when average interval is not 1??
        print('self.df_liu_results')
        print(self.df_liu_results)
        time_lane = self.df_liu_results.loc[:, (lane_id, 'time')] 
        arr_ground_truth = np.array([])
        print('time_lane:', time_lane)       
        for phase in range(1, len(time_lane)+1):
            phase_start = self.df_liu_results.loc[phase, (lane_id, 'phase start')]
            if phase == 1:
                first_phase_start = phase_start
            phase_end = self.df_liu_results.loc[phase, (lane_id, 'phase end')]
            ground_truth = self.df_liu_results.loc[phase, (lane_id, 'ground-truth')]            
            temp_arr_ground_truth = np.full((int(phase_end-phase_start), 1), ground_truth)            
            if arr_ground_truth.size == 0:
                arr_ground_truth = temp_arr_ground_truth
            else:
                arr_ground_truth = np.vstack((arr_ground_truth, temp_arr_ground_truth))
            print('first_phase_start:', first_phase_start)
            print('ground_truth:', ground_truth)
            #print('temp_arr_ground_truth:', temp_arr_ground_truth)
            print('arr_ground_truth.shape:', arr_ground_truth.shape)
            print('temp_arr_ground_truth.shape:', temp_arr_ground_truth.shape)
            #print('arr_ground_truth:', arr_ground_truth)  
            print('--------')             
            
        #crop the right time-window in seconds
        print('start_time:', start_time)
        print('first_phase_start:', first_phase_start)
        print('end_time:', end_time)
        arr_ground_truth_1second = arr_ground_truth[int(start_time-first_phase_start):int(end_time-first_phase_start), 0]
        print('arr_ground_truth_1second.shape', arr_ground_truth_1second.shape)
        #just take the points of data from the average -interval
        #example: average interval 5 sec -> take one value every five seconds!
        arr_ground_truth_average_interval = arr_ground_truth_1second[0::average_interval]
        return arr_ground_truth_average_interval
        
    def unload_data(self):
        #this code should unload all the data and give memory free
        self.parsed_xml_tls = None
        pass
    
    def get_preprocessing_end_time(self, liu_lanes):
        list_end_time = []
        print('liu_lanes:', liu_lanes)
        for lane in liu_lanes:
            df_last_row = self.df_liu_results.iloc[-1, :]  
            print('df_last_row:', df_last_row)
            list_end_time.append(df_last_row.loc[(lane,'phase end')])
        print('list_end_time:', list_end_time)
        print('minimum of list_end_time:', min(list_end_time))
        return min(list_end_time)
        
        
    