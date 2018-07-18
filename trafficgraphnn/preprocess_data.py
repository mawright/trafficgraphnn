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

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from trafficgraphnn.utils import iterfy


_logger = logging.getLogger(__name__)

class PreprocessData(object):
    def __init__(self, sumo_network):
        self.sumo_network = sumo_network
        self.graph = self.sumo_network.get_graph()
        self.liu_results_path = os.path.join(os.path.dirname(
                self.sumo_network.netfile), 'liu_estimation_results.h5')
        self.detector_data_path =  os.path.join(os.path.dirname(
                self.sumo_network.netfile), 'output')
        self.df_liu_results = pd.DataFrame()
        try:
            self.df_liu_results = pd.read_hdf(self.liu_results_path)
        except IOError:
            print('An error occured trying to read the hdf file.')
                

        
    def get_liu_results(self, lane_ID, phase):
        series_results = self.df_liu_results.loc[phase, lane_ID]        
        # return series with parameter time, ground-truth,
        # estimation hybrid/ pure liu
        return series_results
        
    def get_phase_for_time(self, lane_ID, point_of_time):
        #this code should give the phase for a specific point of time and lane back
        time_lane = self.df_liu_results.loc[:, (lane_ID, 'time')] 
               
        for phase in range(1, len(time_lane)):
            if (phase > 2 and 
                point_of_time <= time_lane[phase] and 
                point_of_time > time_lane[phase-1]
                ):                
                return phase
            
            if (point_of_time <= time_lane[1]):
                return 1
            
    
    def preprocess_detector_data(self, average_intervals, average_over_cycle = False):
        #code that is averaging over specific intervals 
        #(eg. 5s, 15s, 30s; given in a list) and an option to average 
        #over a whole cycle. For each averaging interval a new .h5 file is created
        
        file_list = os.listdir(self.detector_data_path)
        str_e1 = 'e1'
       
        for interval in average_intervals:
            print('processing interval ', interval)
            self.df_interval_results = pd.DataFrame() #reset df for every interval
            
            for filename in file_list:
                if str_e1 in filename:                    
                    df_detector = self.process_e1_data(filename, interval)
                    self.df_interval_results = pd.concat(
                            [self.df_interval_results, df_detector], axis = 1)                   
                  
            self.df_interval_results.to_hdf(os.path.join(os.path.dirname(self.sumo_network.netfile),
                            'e1_detector_data_'+ str(interval) + '_seconds.h5'),
                            key = 'df_estimation_results')
                
    def process_e1_data(self, filename, interval):
        append_cnt=0    #counts how often new data were appended
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
        
        df_detector = pd.DataFrame()
        
        for event, elem in et.iterparse(os.path.join(self.detector_data_path, filename)):
            
            if elem.tag == 'interval':
            
                list_nVehContrib.append(float(elem.attrib.get('nVehContrib')))
                list_flow.append(float(elem.attrib.get('flow')))
                list_occupancy.append(float(elem.attrib.get('occupancy')))
                list_speed.append(float(elem.attrib.get('speed')))
                list_length.append(float(elem.attrib.get('length')))            
                append_cnt += 1
                
                if append_cnt == interval:
                    
                    append_cnt = 0
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
                    detector_id = elem.attrib.get('id')
                    
                    #reset temporary lists
                    list_nVehContrib = []
                    list_flow = []
                    list_occupancy = []
                    list_speed = []
                    list_length = []
                                
        iterables = [[detector_id], [
                'nVehContrib', 'flow', 'occupancy', 'speed', 'length']]
        index = pd.MultiIndex.from_product(iterables, names=['detector ID', 'values'])
        df_detector = pd.DataFrame(index = [memory_end_time], columns = index)
        df_detector.index.name = 'end time'
        
        # fill interval_series with values
        df_detector[detector_id, 'nVehContrib'] = memory_nVehContrib
        df_detector[detector_id, 'flow'] = memory_flow
        df_detector[detector_id, 'occupancy'] = memory_occupancy
        df_detector[detector_id, 'speed'] = memory_speed
        df_detector[detector_id, 'length'] = memory_length
       
        return df_detector 


    def unload_data(self):
        #this code should unload all the data and give memory free
        pass
    