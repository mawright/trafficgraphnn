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
from trafficgraphnn.get_tls_data import get_tls_data

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
                            'e1_detector_data_'+ str(interval) + '_seconds.h5'),
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
                'e1_detector_data_tls_interval.h5'), key = 'preprocessed e1_detector_data')         
                
                    
                
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
        
    def unload_data(self):
        #this code should unload all the data and give memory free
        self.parsed_xml_tls = None
        pass
    