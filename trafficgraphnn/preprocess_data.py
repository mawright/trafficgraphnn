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
import keras.backend as K

from trafficgraphnn.get_tls_data import get_tls_data
from trafficgraphnn.sumo_network import SumoNetwork
from trafficgraphnn.utils import E2IterParseWrapper

_logger = logging.getLogger(__name__)

class PreprocessData(object):
    def __init__(self, sumo_network, simu_num):
        self.sumo_network = sumo_network
        self.graph = self.sumo_network.get_graph()
        self.simu_num = simu_num
        self.liu_results_path = os.path.join(os.path.dirname(
                self.sumo_network.netfile), 'liu_estimation_results'+ str(self.simu_num) + '.h5')
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
            phase_start = self.df_liu_results.loc[phase, (lane_ID, 'phase start')]
            phase_end = self.df_liu_results.loc[phase, (lane_ID, 'phase end')]
            if point_of_time >= phase_start and point_of_time < phase_end:
                return phase
            
        print('Point of time out of time window!')
        return np.nan 
    
    def get_nVehSeen_from_e2(self, detector_ID, start_time, duration_in_sec, average_interval, num_rows):
        seq_nVehSeen = np.zeros((start_time+duration_in_sec))
        file_list = os.listdir(self.detector_data_path)
        for filename in file_list:
            if detector_ID in filename:
                file_path = os.path.join(self.detector_data_path, filename)
                parsed_xml_e2_detector = E2IterParseWrapper(file_path, True)
                for interval in parsed_xml_e2_detector.iterate_until(start_time+duration_in_sec):
                    interval_time = int(float(interval.attrib.get('begin')))
                    det_id = interval.attrib.get('id')
                    nVehSeen = interval.attrib.get('nVehSeen')
                    assert det_id == detector_ID
                    seq_nVehSeen[interval_time] = nVehSeen
                    
                #crop out the right time frame
                seq_nVehSeen = seq_nVehSeen[start_time:start_time+duration_in_sec]
                
                #average over average interval
                averaged_nVehSeen = np.zeros((num_rows))
                for index in range(num_rows):
                    avrg_start = index*average_interval
                    avrg_end = avrg_start+average_interval
                    averaged_nVehSeen[index] = np.average(seq_nVehSeen[avrg_start:avrg_end])
                return averaged_nVehSeen

    def preprocess_detector_data(self, average_intervals, num_index=0, average_over_cycle=False):
        # code that is averaging over specific intervals
        # (eg. 5s, 15s, 30s; given in a list) and an option to average
        # over a whole cycle. For each averaging interval a new .h5 file is created

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
                            key = 'preprocessed_e1_detector_data')

        if average_over_cycle == True:
            self.df_interval_results = pd.DataFrame() #reset df for every interval
            for filename in file_list:
                if str_tls in filename:
                    filename_tls = filename

            for filename in file_list:
                if str_e1 in filename:
                    lane_id = self.get_lane_id_from_filename(filename)
                    phase_length, phase_start = self.calc_tls_data(lane_id)
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

    def calc_tls_data(self, lane_id):
        reader = self.net_reader.lane_readers[lane_id]

        return reader.phase_length, reader.phase_start

    def get_lane_id_from_filename(self, filename):
        search_str_start = 'e1_'
        search_str_end = '_output.xml'
        index_start = filename.find(search_str_start) + 3
        index_end = filename.find(search_str_end) - 2
        lane_id = filename[index_start:index_end]
        return lane_id.replace('-', '/')

    def preprocess_A_X_Y(self,
                       average_interval =1,
                       sample_size = 10,
                       start = 200,
                       end = 2000,
                       simu_num = 0,
                       interpolate_ground_truth = False,
                       test_data = False,
                       sample_time_sequence = False,
                       ord_lanes = None):

        self.preprocess_detector_data([average_interval], num_index = simu_num)

        df_detector = pd.read_hdf(os.path.join(os.path.dirname(self.sumo_network.netfile),
                                               'e1_detector_data_' + str(average_interval) + '_seconds' + str(simu_num) + '.h5'))

        try:
            if test_data:
                df_liu_results = pd.read_hdf(os.path.join(os.path.dirname(self.sumo_network.netfile),
                        'liu_estimation_results_test_data_' + str(simu_num) + '.h5'))
            else:
                df_liu_results = pd.read_hdf(os.path.join(os.path.dirname(self.sumo_network.netfile),
                        'liu_estimation_results' + str(simu_num) + '.h5'))

        except IOError:
            print('No file for liu estimation results found.')

        if ord_lanes == None and simu_num == 0:
            subgraph = self.get_subgraph(self.graph)
            ord_lanes = [lane for lane in subgraph.nodes]
            A_down, A_up, A_neigh = self.get_all_A(subgraph, ord_lanes)
            A_list = [A_down, A_up, A_neigh]
        else:
            A_list = None #no A needs to be exported

        X, Y = self.calc_X_and_Y(ord_lanes, df_detector, df_liu_results,
                            start, end, average_interval, sample_size,
                            interpolate_ground_truth, sample_time_sequence = sample_time_sequence)

        return A_list, X, Y, ord_lanes

    def calc_X_and_Y(self,
                     ord_lanes,
                     df_detector,
                     df_liu_results,
                     start_time,
                     end_time,
                     average_interval,
                     sample_size,
                     interpolate_ground_truth,
                     sample_time_sequence = False
                     ):
        '''
        Takes the dataframe from the e1 detectors and liu results as input and gives the X and Y back.

        '''

        for lane, cnt_lane in zip(ord_lanes, range(len(ord_lanes))):
            # faster solution: just access df once and store data in arrays or lists
            lane_id = lane.replace('/', '-')
            stopbar_detector_id = 'e1_' + str(lane_id) + '_0'
            adv_detector_id = 'e1_' + str(lane_id) + '_1'
            e2_detector_id = 'e2_' + str(lane_id) + '_0'

            if cnt_lane == 0:
                #dimesion: time x nodes x features
                num_rows = len(df_detector.loc[start_time:end_time, (stopbar_detector_id, 'nVehContrib')]) -1 # (-1):last row is belongs to the following average interval
                duration_in_sec = end_time-start_time
                X_unsampled = np.zeros((num_rows, len(ord_lanes), 8))
                Y_unsampled = np.zeros((num_rows, len(ord_lanes), 2))

            X_unsampled[:, cnt_lane, 0] = df_detector.loc[start_time:end_time-average_interval, (stopbar_detector_id, 'nVehContrib')]
            X_unsampled[:, cnt_lane, 1] = df_detector.loc[start_time:end_time-average_interval, (stopbar_detector_id, 'occupancy')]
            X_unsampled[:, cnt_lane, 2] = df_detector.loc[start_time:end_time-average_interval, (stopbar_detector_id, 'speed')]
            X_unsampled[:, cnt_lane, 3] = df_detector.loc[start_time:end_time-average_interval, (adv_detector_id, 'nVehContrib')]
            X_unsampled[:, cnt_lane, 4] = df_detector.loc[start_time:end_time-average_interval, (adv_detector_id, 'occupancy')]
            X_unsampled[:, cnt_lane, 5] = df_detector.loc[start_time:end_time-average_interval, (adv_detector_id, 'speed')]
            X_unsampled[:, cnt_lane, 6] = self.get_tls_binary_signal(start_time, duration_in_sec, lane, average_interval, num_rows)
            X_unsampled[:, cnt_lane, 7] = self.get_ground_truth_array(start_time,
                                                   duration_in_sec, lane, average_interval, num_rows,   #use liu results as features
                                                   interpolate_ground_truth = interpolate_ground_truth,
                                                   ground_truth_name = 'estimated hybrid')

            Y_unsampled[:, cnt_lane, 0] = self.get_ground_truth_array(start_time,
                       duration_in_sec, lane, average_interval, num_rows,
                       interpolate_ground_truth = interpolate_ground_truth,
                       ground_truth_name = 'ground-truth')

            Y_unsampled[:, cnt_lane, 1] = self.get_nVehSeen_from_e2(e2_detector_id,
                       start_time, duration_in_sec, average_interval, num_rows)

        if sample_time_sequence:

            num_samples = math.ceil(X_unsampled.shape[0]/sample_size)
            print('number of samples:', num_samples)
            X_sampled = np.zeros((num_samples, sample_size, len(ord_lanes), 8))
            Y_sampled = np.zeros((num_samples, sample_size, len(ord_lanes), 2))
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
                                               np.zeros((diff_samples, len(ord_lanes), 8))))
                    Y_zero_padding = np.vstack((Y_unsampled[sample_start:sample_end][:][:],
                               np.zeros((diff_samples, len(ord_lanes), 2))))
                    X_sampled[cnt_sample][:][:][:] = X_zero_padding
                    Y_sampled[cnt_sample][:][:][:] = Y_zero_padding

            return X_sampled, Y_sampled #dimension (num_samples x num_timesteps x num_lanes x num_features)
        else:
            #reshape arrays, that they have the same output shape than sampeled X and Y
            X_reshaped = np.reshape(X_unsampled, (1, X_unsampled.shape[0], X_unsampled.shape[1], X_unsampled.shape[2]))
            Y_reshaped = np.reshape(Y_unsampled, (1, Y_unsampled.shape[0], Y_unsampled.shape[1], Y_unsampled.shape[2]))
            return X_reshaped, Y_reshaped #dimension ( 1 x num_timesteps x num_lanes x num_features)

    def get_subgraph(self, graph):
        node_sublist = [lane[0] for lane in self.graph.nodes(data=True) if lane[1] != {}]
        sub_graph = graph.subgraph(node_sublist)
        return sub_graph

    def get_ground_truth_array(self,
                               start_time,
                               duration_in_sec,
                               lane_id, average_interval,
                               num_rows,
                               interpolate_ground_truth= False,
                               ground_truth_name = 'ground-truth'):
        """Returns the array (either ground truth or liu results) for a specific lane for a time period for an specific interval

        "start time" corresponds to start of the ground truth data, int
        "duration_in_sec" duration of the time period in seconds, int
        "lane_id" lane_id for the specific lane, str
        "average_interval" interval over which the ground truth data are averaged, int
        "num_rows" number of rows in the X vector -> ensures, that no missmatch error occurs, int
        "interpolate_ground_truth": if ground truth data are as a step function in the array or if they are linear interpolated, bool
        "ground_truth_name": can be either "ground-truth" or "estimated hybrid" or "estimated pure liu", str

        :return: Array with ground truth data
        :rtype: numpy array
        """

        #ATTENTION!!! What to do when average interval is not 1??
        time_lane = self.df_liu_results.loc[:, (lane_id, 'time')]
        arr_ground_truth = np.array([])
        for phase in range(len(time_lane)):
            phase_start = self.df_liu_results.loc[phase, (lane_id, 'phase start')]
            phase_end = self.df_liu_results.loc[phase, (lane_id, 'phase end')]
            if phase == 0:
                first_phase_start = phase_start
                previous_ground_truth = 0

            ground_truth = self.df_liu_results.loc[phase, (lane_id, ground_truth_name)]
            if interpolate_ground_truth == False:
                temp_arr_ground_truth = np.full((int(phase_end-phase_start), 1), ground_truth)
            else:
                temp_arr_ground_truth = np.linspace(previous_ground_truth, ground_truth, num = int(phase_end-phase_start))
                temp_arr_ground_truth = np.reshape(temp_arr_ground_truth, (int(phase_end-phase_start), 1))

            if arr_ground_truth.size == 0:
                arr_ground_truth = temp_arr_ground_truth
            else:
                arr_ground_truth = np.vstack((arr_ground_truth, temp_arr_ground_truth))

            previous_ground_truth = ground_truth

        #crop the right time-window in seconds

        lower_bound = int(start_time)-int(first_phase_start)
        upper_bound = int(start_time)+duration_in_sec-int(first_phase_start)
        arr_ground_truth_1second = np.reshape(arr_ground_truth[lower_bound:upper_bound, 0], upper_bound-lower_bound)
        #just take the points of data from the average -interval
        #example: average interval 5 sec -> take one value every five seconds!
        arr_ground_truth_average_interval = arr_ground_truth_1second[0::average_interval]
        arr_ground_truth_average_interval = arr_ground_truth_average_interval[0:num_rows] #make sure, that no dimension problems occur
        return arr_ground_truth_average_interval

        
    
    def get_tls_binary_signal(self, start_time, duration_in_sec, lane_id, average_interval, num_rows):
        time_lane = self.df_liu_results.loc[:, (lane_id, 'time')]
        arr_tls_binary = np.array([])
        for phase in range(len(time_lane)):
            phase_start = self.df_liu_results.loc[phase, (lane_id, 'phase start')]
            phase_end = self.df_liu_results.loc[phase, (lane_id, 'phase end')]
            green_start = self.df_liu_results.loc[phase, (lane_id, 'tls start')]
            if phase == 0:
                first_phase_start = phase_start
            temp_array_red_phase = np.full((int(green_start-phase_start), 1), 0) #red phase from phase start to green start (0)
            temp_array_green_phase = np.full((int(phase_end-green_start), 1), 1) #green phase from green start to phase end (1)
            if arr_tls_binary.size == 0:
                arr_tls_binary = temp_array_red_phase
                arr_tls_binary = np.vstack((arr_tls_binary, temp_array_green_phase))
            else:
                arr_tls_binary =  np.vstack((arr_tls_binary, temp_array_red_phase))
                arr_tls_binary = np.vstack((arr_tls_binary, temp_array_green_phase))

        lower_bound = int(start_time)-int(first_phase_start)
        upper_bound = int(start_time)+duration_in_sec-int(first_phase_start)
        arr_tls_binary_1second = np.reshape(arr_tls_binary[lower_bound:upper_bound, 0], upper_bound-lower_bound)
        #just take the points of data from the average -interval
        #example: average interval 5 sec -> take one value every five seconds!
        arr_tls_binary_average_interval = arr_tls_binary_1second[0::average_interval]
        arr_tls_binary_average_interval = arr_tls_binary_average_interval[0:num_rows] #make sure, that no dimension problems occur
        return arr_tls_binary_average_interval

    def unload_data(self):
        #this code should unload all the data and give memory free
        self.parsed_xml_tls = None
        pass

    def get_preprocessing_end_time(self, liu_lanes, average_interval):
        list_end_time = []
        for lane in liu_lanes:
            df_last_row = self.df_liu_results.iloc[-1, :]
            list_end_time.append(df_last_row.loc[(lane,'phase end')])
            end_time = min(list_end_time)-10 #10seconds reserve, otherwise complications occur
            end_time = int(end_time/average_interval) * average_interval #make sure, that end time is matching with average_interval
            self.preprocess_end_time = end_time
            return end_time

    def get_A_for_neighboring_lanes(self, order_lanes):
        indexes = [] #list with tuple of indexes (input_lane_index, neighbor_lane_index)
        for lane_id, input_lane_index in zip(order_lanes, range(len(order_lanes))):
            neighbor_lanes = self.sumo_network.get_neighboring_lanes(lane_id, include_input_lane=True)
            for neighbor in neighbor_lanes:
                if neighbor in order_lanes:
                    neighbor_lane_index = order_lanes.index(neighbor)
                    indexes.append((input_lane_index, neighbor_lane_index))

        N = len(order_lanes)
        A = np.zeros((N, N))
        for index_tuple in indexes:
            A[index_tuple[0], index_tuple[1]] = 1
        A = A + np.transpose(A)
        A = np.minimum(A, np.ones((N,N)))
        return A

    def get_all_A(self, subgraph, order_lanes):
        N = len(order_lanes)

        #get adjacency matrix for all downstream connenctions:
        A_down = nx.adjacency_matrix(subgraph)
        A_down = np.eye(N, N) + A_down
        A_down = np.minimum(A_down, np.ones((N,N)))
        #get adjacency matrix for all upstream connections
        A_up = np.transpose(A_down)

        #get adjacency matrix for all neigboring lanes (e.g. for lane changing)
        A_neighbors = self.get_A_for_neighboring_lanes(order_lanes) #Skip using neighbors for the beginning
        A_neighbors = np.eye(N, N) + A_neighbors
        A_neighbors = np.minimum(A_neighbors, np.ones((N,N)))

        return A_down, A_up, A_neighbors


def reshape_for_3Dim(input_arr):
    '''
    reshapes input from shape (samples x timesteps x lanes x features)
    to output of shape (samples*lanes x timesteps x features)
    '''

    assert len(input_arr.shape) == 4
    input_arr_dim = input_arr.get_shape().as_list()
    num_samples = input_arr_dim[0]
    num_timesteps = input_arr_dim[1]
    num_lanes = input_arr_dim[2]
    num_features = input_arr_dim[3]

    # we need to use permute_dimensions first to (samples, lanes, timesteps, features)
    input_arr_permuted = K.permute_dimensions(input_arr, (0, 2, 1, 3))

    #then we can reshape
    output_arr = K.reshape(input_arr_permuted, (num_samples*num_lanes, -1, num_features))

#    output_arr = np.zeros((num_samples*num_lanes, num_timesteps, num_targets))
#    for sample in range(num_samples):
#        for lane in range(num_lanes):
#            lane_data = input_arr[sample, :, lane, :]
#            output_arr[sample * num_lanes + lane, :, :] = lane_data
    return output_arr

def reshape_for_4Dim(input_arr):
    '''
    reshapes input from shape (samples*lanes x timesteps x features)
    to output of shape (samples x timesteps x lanes x features)

    in future:
    input list: [input_array, number of lanes]; neccesary to do it with a list because of keras lambda layer
    additional info: number of lanes per road network; neccessary for reshaping;
                                    information about num_lanes got lost during reshape_for_LSTM
    '''
#    sess = tf.Session()
#    input_arr = input_list[0]
    input_arr_dim = input_arr.get_shape().as_list()

    num_lanes = 120 #of course not a solution, but multiple Inputs into lambda layer does not work right now...
#    num_lanes = int(K.eval(input_list[1]))
#    num_lanesXsamples = input_arr_dim[0]
    num_timesteps = input_arr_dim[1]
    num_features = input_arr_dim[2]

    assert len(input_arr.shape) == 3

    #reshape tensor back to dimension (samples x lanes x timesteps x features)
    input_arr_reshaped = K.reshape(input_arr, (-1, num_lanes, num_timesteps, num_features))

    #permute array to (samples x timesteps x lanes x features)
    output_arr = K.permute_dimensions(input_arr_reshaped, (0, 2, 1, 3))

# alternative approch, can be deleted as soon as upper approach works
#    output_arr = np.zeros((num_lanesXsamples//num_lanes, num_timesteps, num_lanes, num_features))
#
#    cnt_lanes = 0
#    cnt_samples = 0
#    for lane in range(num_lanesXsamples):
#        if cnt_lanes == num_lanes:
#            cnt_samples += 1
#            cnt_lanes = 0
#        lane_data = input_arr[lane, :, :]
#        lane_index = lane - num_lanes*cnt_samples
#        output_arr[cnt_samples, :, lane_index, :] = lane_data
#        cnt_lanes += 1

    return output_arr

