#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 11:42:42 2018

To generate traffic light phases where every lane is prioritized. 
Just for 3 lanes per edge!
@author: simon
"""

import os, sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:   
    sys.exit("please declare environment variable 'SUMO_HOME'")

import xml.etree.cElementTree as et


def tls_config(network_path):
    
    #Attention: This is the configuration for tl for 3 lane grid network!
    phase_duration = ['33', '3', '33', '3', '33', '3', '33', '3']
    phase_state = ['GGGGGGrrrrrrrrrrrrrrrrrr', 'yyyyyyrrrrrrrrrrrrrrrrrr',
                   'rrrrrrrrrrrrGGGGGGrrrrrr', 'rrrrrrrrrrrryyyyyyrrrrrr',
                   'rrrrrrrrrrrrrrrrrrGGGGGG', 'rrrrrrrrrrrrrrrrrryyyyyy',
                   'rrrrrrGGGGGGrrrrrrrrrrrr', 'rrrrrryyyyyyrrrrrrrrrrrr'
                   ]
    
    parsed_net_tree = et.parse(network_path)
    
    root = parsed_net_tree.getroot()
    for child in root:
        if child.tag == 'tlLogic':

            grandchildren = child.getchildren()
            if len(grandchildren) == 8:
                for phase, cnt in zip(grandchildren, range(0, 8)):
                    child.remove(phase)
                    attribute = {}
                    attribute['duration'] = phase_duration[cnt]
                    attribute['state'] = phase_state[cnt]
                    et.SubElement(child, 'phase', attribute)
                    
    parsed_net_tree.write(network_path)

