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
            
    
    def preprocess_detector_data(self, average_interval, average_over_cycle = False):
        #code that is averaging over specific intervals 
        #(eg. 5s, 15s, 30s; given in a list) and an option to average 
        #over a whole cycle. For each averaging interval a new h5 file is created
        pass
    
    def unload_data(self):
        #this code should unload all the data and give memory free
        pass
    