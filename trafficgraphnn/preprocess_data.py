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
    def __init__(self, sumo_net):
        self.sumo_net = sumo_net
        self.liu_results_path = os.path.join(os.path.dirname(
                self.sumo_net.netfile), 'liu_estimation_results.h5')
        
    def get_liu_results(self, phase, lane_ID):
        #this code should return a the results of the liu_method 
        #for a specific lane and phase    
        
        
        pass