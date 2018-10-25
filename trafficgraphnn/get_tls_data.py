#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 14:58:02 2018

@author: simon
"""


def get_tls_data(parsed_xml_tls, lane_id):

    search_finished = False
    duration_green_light = 0
    marker_toLane = None
    for node in parsed_xml_tls.getroot():
        fromLane = str(node.attrib.get('fromLane'))
        toLane = str(node.attrib.get('toLane'))
        end = float(node.attrib.get('end'))

        if fromLane == lane_id and marker_toLane == toLane and search_finished == False: #find tls for 2nd time
            phase_end = float(node.attrib.get('end'))
            phase_length = int(phase_end - phase_start)
            search_finished = True

        if fromLane == lane_id:
            #searching for the longest duration in cycle
            if float(node.attrib.get('duration')) > duration_green_light:
                duration_green_light = float(node.attrib.get('duration'))
                marker_toLane = str(node.attrib.get('toLane'))
                phase_start = end
                search_finished = False

    return phase_start, phase_length, duration_green_light
