#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 16:21:00 2018

@author: simon
"""
import logging
import os
from xml.etree import cElementTree as et
import networkx as nx
import math
import re
from collections import OrderedDict
import six
from contextlib import ExitStack
import warnings
import tables

import numpy as np
import pandas as pd

from trafficgraphnn.sumo_output_reader import SumoLaneOutputReader, SumoNetworkOutputReader
from trafficgraphnn.utils import (
    E1IterParseWrapper, E2IterParseWrapper, DetInfo, get_preprocessed_filenames,
    get_sim_numbers_in_preprocess_store, sumo_output_xmls_to_hdf)
from trafficgraphnn.liumethod import LiuEtAlRunner
from trafficgraphnn.load_data import pad_value_for_feature

_logger = logging.getLogger(__name__)


class PreprocessData(object):
    def __init__(self,
                 sumo_network,
                 preproc_file_number=None,
                 lane_order=None,
                 detector_data_dir='output',
                 extracted_detector_hdf=None):
        self.sumo_network = sumo_network
        if preproc_file_number is None:
            preproc_file_number = self._next_file_number()
        self.preproc_file_number = preproc_file_number
        self.liu_results_path = os.path.join(os.path.dirname(
                self.sumo_network.netfile),
                'liu_estimation_results' + str(self.preproc_file_number) + '.h5')
        self.detector_data_path =  os.path.join(os.path.dirname(
                self.sumo_network.netfile), detector_data_dir)
        self.df_liu_results = pd.DataFrame()

        self.preprocess_file = os.path.join(
            os.path.dirname(self.sumo_network.netfile),
            'preprocessed_data', 'sim_{:04}.h5'.format(preproc_file_number))
        if not os.path.exists(os.path.dirname(self.preprocess_file)):
            os.makedirs(os.path.dirname(self.preprocess_file))

        self.extracted_detector_hdf = extracted_detector_hdf

        self.lanes = [
            lane.getID() for edge in self.sumolib_net.getEdges()
            for lane in edge.getLanes() if len(lane.getOutgoing()) > 0]

        if lane_order is not None:
            self.lanes = [lane for lane in lane_order if lane in self.lanes]

    def run_defaults(self, lanes=None, complevel=5, complib='blosc:lz4'):
        raw_xml_filename =self.detector_xmls_to_df_hdf(complevel=complevel,
                                                       complib=complib)
        self.run_liu_method(raw_xml_filename)
        self.read_data(complevel=complevel, complib=complib)
        self.extract_liu_results(complevel=complevel, complib=complib)
        self.write_per_lane_table(complevel=complevel, complib=complib)

    def detector_xmls_to_df_hdf(self, complevel=5, complib='blosc:lz4'):
        filename = sumo_output_xmls_to_hdf(self.detector_data_path,
                                           complevel=complevel,
                                           complib=complib)
        return filename

    def run_liu_method(self, input_data_hdf=None, max_phase=np.inf):
        liu_runner = LiuEtAlRunner(
            self.sumo_network, lane_subset=self.lanes,
            sim_num=self.preproc_file_number,
            input_data_hdf_file=input_data_hdf)
        num_liu_phases = liu_runner.get_max_num_phase()
        end_phase = min(num_liu_phases, max_phase)
        liu_runner.run_up_to_phase(end_phase)
        liu_runner.reader.close_hdfstores()

    def extract_liu_results(self, lanes=None, complevel=5, complib='blosc:lz4'):
        if lanes is None:
            lanes = self.lanes
        liu_output_file = os.path.join(os.path.dirname(self.sumo_network.netfile),
                                       'liu_estimation_results{}.h5'.format(self.preproc_file_number))

        with pd.HDFStore(liu_output_file) as liu_store:
            try:
                liu_df = liu_store.get('df_estimation_results')
            except KeyError:
                _logger.warning(
                    "Liu results not found. Try running the 'run_liu_method() method.")
                return

        with pd.HDFStore(self.preprocess_file, complevel=complevel, complib=complib) as store:
            sim_number = self._next_simulation_number(store)
            for lane in lanes:
                lane_df = liu_df.iloc[:, (liu_df.columns.get_level_values(0) == lane)
                                          & liu_df.columns.get_level_values(1).isin(
                                              ['time', 'estimated hybrid',
                                               'estimated hybrid (veh)'])]
                lane_df.columns = ['time', 'liu_estimated_m',
                                   'liu_estimated_veh']
                lane_df = lane_df.drop_duplicates()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', tables.NaturalNameWarning)
                    store.append('{}/liu_results/_{:04}'.format(lane, sim_number), lane_df)
        _logger.info('Added liu results to file %s', store.filename)

    def read_data(self, start_time=0, end_time=np.inf,
                  e1_features=None, e2_features=None,
                  complib='blosc:lz4', complevel=5):

        light_timings = self.read_light_timings(start_time=start_time, end_time=end_time)

        with ExitStack() as store_stack:
            store = store_stack.enter_context(
                pd.HDFStore(self.preprocess_file, complevel=complevel, complib=complib))
            if self.extracted_detector_hdf is not None:
                extracted_detector_store = store_stack.enter_context(
                    pd.HDFStore(self.extracted_detector_hdf, 'r'))
            else:
                extracted_detector_store = None

            sim_number = self._next_simulation_number(store)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', tables.NaturalNameWarning)
                for lane in self.lanes:
                    e1_data = self.read_e1_data_for_lane(
                        lane, start_time, end_time,
                        features=e1_features,
                        extracted_detector_store=extracted_detector_store)
                    for det, data in e1_data.items():
                        store.append('{}/e1/{}/_{:04}'.format(lane, det, sim_number),
                                     data)

                    e2_data = self.read_e2_data_for_lane(
                        lane, start_time, end_time,
                        features=e2_features,
                        extracted_detector_store=extracted_detector_store)
                    for det, data in e2_data.items():
                        store.append('{}/e2/{}/_{:04}'.format(lane, det, sim_number),
                                     data)

                    store.append('{}/green/_{:04}'.format(lane, sim_number),
                                light_timings[lane].rename('green'))
            self.store_adjacency_tables(complevel=complevel, complib=complib)

    def store_adjacency_tables(self,
                               downstream=True,
                               upstream=True,
                               neighboring_lanes=True,
                               turn=False,
                               thru=False,
                               complevel=5,
                               complib='blosc:lz4'):

        with pd.HDFStore(self.preprocess_file, complevel=complevel, complib=complib) as store:
            lanes = self.lanes
            if downstream:
                store.put('A_downstream',
                          nx.to_pandas_adjacency(self.graph,
                                                 nodelist=lanes).astype('bool'))
            if upstream:
                store.put('A_upstream',
                          nx.to_pandas_adjacency(self.graph.reverse(),
                                                 nodelist=lanes).astype('bool'))
            if neighboring_lanes:
                A = self.sumo_network.get_lane_graph_for_neighboring_lanes(
                                include_self_adjacency=False)
                store.put('A_neighbors',
                          nx.to_pandas_adjacency(A, nodelist=lanes).astype('bool'))
            if turn:
                A = self.sumo_network.get_lane_graph_for_turn_movements()
                store.put('A_turn_movements',
                          nx.to_pandas_adjacency(A, nodelist=lanes).astype('bool'))
            if thru:
                A = self.sumo_network.get_lane_graph_for_thru_movements()
                store.put('A_through_movements',
                          nx.to_pandas_adjacency(A, nodelist=lanes).astype('bool'))

    def write_per_lane_table(self,
                             X_cont_features=['nvehContrib',
                                              'occupancy',
                                              'speed',
                                              'green'],
                             Y_cont_features=['nVehSeen',
                                              'maxJamLengthInMeters',
                                              'maxJamLengthInVehicles'],
                             X_on_green_features=['liu_estimated_m',
                                                  'liu_estimated_veh'],
                             Y_on_green_features=['maxJamLengthInMeters',
                                                  'maxJamLengthInVehicles'],
                             complib='blosc:lz4', complevel=5,
                             X_on_green_empty_value=-1,
                             delete_intermediate_tables=True):

        with pd.HDFStore(self.preprocess_file, complevel=complevel, complib=complib) as store:
            _logger.info('Writing X and Y matrices to file %s...', store.filename)
            sim_number = self._next_simulation_number(store)
            sim_string = '_{:04}'.format(sim_number)
            for lane in self.lanes:
                e1s = self.lane_detectors_sorted_by_position(lane, 'e1')
                dfs_e1 = OrderedDict((det.id,
                                      store['{}/e1/{}/{}'.format(lane, det.id, sim_string)])
                                     for det in e1s)
                X_dfs_e1 = OrderedDict((id, df[[feat for feat in X_cont_features
                                        if feat in df]])
                                       for id, df in dfs_e1.items())
                Y_dfs_e1 = OrderedDict((id, df[[feat for feat in Y_cont_features
                                        if feat in df]])
                                       for id, df in dfs_e1.items())

                e2s = self.lane_detectors_sorted_by_position(lane, 'e2')
                dfs_e2 = OrderedDict((det.id,
                                      store['{}/e2/{}/{}'.format(lane, det.id, sim_string)])
                                     for det in e2s)
                X_dfs_e2 = OrderedDict((id, df[[feat for feat in X_cont_features
                                        if feat in df]])
                                       for id, df in dfs_e2.items())
                Y_dfs_e2 = OrderedDict((id, df[[feat for feat in Y_cont_features
                                        if feat in df]])
                                       for id, df in dfs_e2.items())

                X_df_e1 = pd.concat(X_dfs_e1, axis=1)
                X_df_e2 = pd.concat(X_dfs_e2, axis=1)
                X_df = pd.concat([X_df_e1, X_df_e2], axis=1)

                if any(x in X_on_green_features
                       for x in ['liu_results', 'liu_estimated_m',
                                 'liu_estimated_veh']):
                    df_liu = store['{}/liu_results/{}'.format(lane, sim_string)]
                    X_df = pd.concat([X_df, df_liu.set_index('time')], axis=1)
                    X_df.fillna(pad_value_for_feature, inplace=True)

                df_green = store['{}/green/{}'.format(lane, sim_string)]
                df_green = df_green.reindex_like(X_df)
                df_green = df_green.fillna(method='ffill')
                if 'green' in X_cont_features:
                    X_df = pd.concat([X_df, df_green], axis=1)
                    X_df['green'].fillna(method='ffill', inplace=True)

                X_df.columns = _prune_det_id_in_colnames(
                    _concat_colnames(X_df.columns), lane)
                store.append('{}/X/{}'.format(lane, sim_string), X_df,
                             complevel=5, complib='blosc:lz4')

                Y_df_e1 = pd.concat(Y_dfs_e1, axis=1)
                Y_df_e2 = pd.concat(Y_dfs_e2, axis=1)
                Y_df = pd.concat([Y_df_e1, Y_df_e2], axis=1)

                if len(Y_on_green_features) > 0:
                    for feat in Y_on_green_features:
                        Y_df.loc[:, Y_df.columns.get_level_values(1) == feat
                                 ] = Y_df.loc[:, Y_df.columns.get_level_values(1) == feat
                                              ].where(
                                        df_green.astype('uint8').shift(-1).diff() == 1)

                Y_df.columns = _prune_det_id_in_colnames(
                    _concat_colnames(Y_df.columns), lane)
                Y_df.fillna(pad_value_for_feature, inplace=True)
                store.append('{}/Y/{}'.format(lane, sim_string), Y_df,
                             complevel=5, complib='blosc:lz4')
                if delete_intermediate_tables:
                    for det in e1s:
                        store.remove('{}/e1/{}/{}'.format(lane, det.id, sim_string))
                    for det in e2s:
                        store.remove('{}/e2/{}/{}'.format(lane, det.id, sim_string))
                    try:
                        store.remove('{}/green/{}'.format(lane, sim_string))
                    except KeyError:
                        pass
                    try:
                        store.remove('{}/liu_results/{}'.format(lane, sim_string))
                    except KeyError:
                        pass
        _logger.info('Done.')

    @property
    def graph(self):
        return self.sumo_network.graph

    @property
    def sumolib_net(self):
        return self.sumo_network.net

    def get_liu_results(self, lane_ID, phase):
        # return series with parameter time, ground-truth,
        # estimation hybrid/ pure liu
        return self.df_liu_results.loc[phase, lane_ID]

    def get_phase_for_time(self, lane_ID, point_of_time):
        #this code should give the phase for a specific point of time and lane back
        time_lane = self.df_liu_results.loc[:, (lane_ID, 'time')]

        for phase in range(1, len(time_lane)):
            phase_start = self.df_liu_results.loc[phase, (lane_ID, 'phase start')]
            phase_end = self.df_liu_results.loc[phase, (lane_ID, 'phase end')]
            if point_of_time >= phase_start and point_of_time < phase_end:
                return phase

        _logger.warning('Point of time out of time window!')
        return np.nan

    def get_nVehSeen_from_e2(self, detector_ID, start_time, duration_in_sec, average_interval, num_rows):
        seq_nVehSeen = np.zeros((start_time+duration_in_sec))
        file_list = os.listdir(self.detector_data_path)
        for filename in file_list:
            if detector_ID in filename:
                file_path = os.path.join(self.detector_data_path, filename)
                parsed_xml_e2_detector = E2IterParseWrapper(file_path, True, id_subset=detector_ID)
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
            _logger.info('processing interval %g', interval)
            self.df_interval_results = pd.DataFrame() #reset df for every interval

            for filename in file_list:
                if str_e1 in filename:
                    df_detector = self.process_e1_data(filename, interval)
                    self.df_interval_results = pd.concat(
                            [self.df_interval_results, df_detector], axis = 1)

            self.df_interval_results.to_hdf(os.path.join(
                    os.path.dirname(self.sumo_network.netfile),
                    'e1_detector_data_'+ str(interval) +
                    '_seconds'+ str(num_index) + '.h5'),
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
                key = 'preprocessed_e1_detector_data')

    def lane_detectors_sorted_by_position(self, lane_id, det_type):
        nx_node = self.graph.nodes[lane_id]
        lane_detectors = nx_node['detectors']

        if det_type == 'e1':
            det_type_longname = 'e1Detector'
        elif det_type == 'e2':
            det_type_longname = 'e2Detector'
        else:
            raise ValueError('Unknown detector type {}'.format(det_type))

        dets_of_type = [DetInfo(k, v) for k,v in lane_detectors.items()
                        if v['type'] == det_type_longname]

        det_by_pos = sorted(dets_of_type, key=lambda x: x.info['pos'])
        return det_by_pos

    def read_data_for_lane_for_det_type(self, lane_id, det_type, features,
                                        start_time=0, end_time=np.inf,
                                        extracted_detector_store=None):
        if det_type == 'e1':
            parser_class = E1IterParseWrapper
        elif det_type == 'e2':
            parser_class = E2IterParseWrapper
        else:
            raise ValueError('Unknown detector type {}'.format(det_type))

        det_by_pos = self.lane_detectors_sorted_by_position(lane_id, det_type)

        lane_data = OrderedDict()
        try:
            features.remove('pos')
            pos = [float(detector.info['pos']) for detector in det_by_pos]
        except ValueError:
            pass
        try:
            features.remove('rel_pos')
            nx_node = self.graph.nodes[lane_id]
            rel_pos = [float(detector.info['pos']) / float(nx_node['length'])
                       for detector in det_by_pos]
        except ValueError:
            pass

        for d, detector in enumerate(det_by_pos):
            if extracted_detector_store is None:
                filename = os.path.join(os.path.dirname(self.sumo_network.netfile),
                                        detector.info['file'])

                parser = parser_class(filename, validate=True, id_subset=detector.id)
                data = {feat: [] for feat in features}
                interval_begins = []
                for _ in parser.iterate_until(start_time):
                    continue
                for interval in parser.iterate_until(end_time):
                    for feat in features:
                        data[feat].append(float(interval.attrib.get(feat)))
                    interval_begins.append(int(float(interval.attrib.get('begin'))))
                df = pd.DataFrame(data)

                index = pd.Int64Index(interval_begins, name='time')
                df = df.set_index(index)
            else:
                df = extracted_detector_store[f'raw_xml/{detector.id}']
                df = df[features]
                df.index.rename('time', inplace=True)

            try:
                df['pos'] = pos[d]
                df['pos'] = df['pos'].astype('category')
            except UnboundLocalError:
                pass
            try:
                df['rel_pos'] = rel_pos[d]
                df['rel_pos'] = df['rel_pos'].astype('category')
            except UnboundLocalError:
                pass

            lane_data[detector.id] = df

        return lane_data

    def read_e1_data_for_lane(self, lane_id, start_time=0, end_time=np.inf,
                              features=None,
                              extracted_detector_store=None):
        if features is None:
            features = ['nVehContrib',
                        'occupancy',
                        'speed',
                        'pos',
                        'rel_pos']
        return self.read_data_for_lane_for_det_type(
            lane_id, 'e1', features=features,
            start_time=start_time, end_time=end_time,
            extracted_detector_store=extracted_detector_store)

    def read_e2_data_for_lane(self, lane_id, start_time=0, end_time=np.inf,
                              features=None,
                              extracted_detector_store=None):
        if features is None:
            features = ['nVehSeen',
                        'maxJamLengthInMeters',
                        'maxJamLengthInVehicles']

        return self.read_data_for_lane_for_det_type(
            lane_id, 'e2', features=features,
            start_time=start_time, end_time=end_time,
            extracted_detector_store=extracted_detector_store)

    def read_light_timings(self, lane_subset=None, start_time=0, end_time=np.inf):
        net_reader = SumoNetworkOutputReader(self.sumo_network)

        if lane_subset is None:
            lane_subset = self.lanes
        else:
            lane_subset = [lane for lane in lane_subset if lane in self.lanes]

        for lane_id in lane_subset:
            lane = self.sumolib_net.getLane(lane_id)
            try:
                out_lane = lane.getOutgoing()[0].getToLane()
            except IndexError:  # this lane has no outgoing lanes
                continue
            lane_reader = SumoLaneOutputReader(
                lane, out_lane, net_reader, self.extracted_detector_hdf)
            for conn in lane.getOutgoing()[1:]:
                lane_reader.add_out_lane(conn.getToLane().getID())
        net_reader.parse_phase_timings()

        green_intervals_by_lane = {
            lane: net_reader.lane_readers[lane].green_intervals
            for lane in net_reader.lane_readers.keys()}

        if np.isinf(end_time):
            end_time = max(interval[-1][-1] for interval in
                           green_intervals_by_lane.values())

        data_dict = {}
        for lane in lane_subset:
            green_intervals = net_reader.lane_readers[lane].green_intervals
            green_binary = [any(itv[0] <= x < itv[1] for itv in green_intervals)
                            for x in range(start_time, end_time)]
            data_dict[lane] = green_binary

        data_df = pd.DataFrame(data_dict)
        data_df.index.set_names('time', inplace=True)

        return data_df

    def _next_file_number(self):
        data_dir = os.path.join(os.path.dirname(self.sumo_network.netfile),
                                'preprocessed_data')

        files = get_preprocessed_filenames(data_dir)
        fnames = [os.path.basename(f) for f in files]
        numbers = [int(re.search(r'\d+', f).group()) for f in fnames]

        return max(numbers, default=0) + 1

    def _next_simulation_number(self, store):

        sim_numbers = get_sim_numbers_in_preprocess_store(
            store, lane_list=self.lanes)
        return max(sim_numbers, default=0) + 1

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
        raise NotImplementedError
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

        df_detector = pd.read_hdf(os.path.join(
                        os.path.dirname(self.sumo_network.netfile),
                        'e1_detector_data_' + str(average_interval) +
                        '_seconds' + str(simu_num) + '.h5'))

        try:
            if test_data:
                df_liu_results = pd.read_hdf(os.path.join(
                        os.path.dirname(self.sumo_network.netfile),
                        'liu_estimation_results_test_data_' +
                        str(simu_num) + '.h5'))
            else:
                df_liu_results = pd.read_hdf(os.path.join(
                        os.path.dirname(self.sumo_network.netfile),
                        'liu_estimation_results' + str(simu_num) + '.h5'))

        except IOError:
            _logger.warning('No file for liu estimation results found.')

        if ord_lanes == None and simu_num == 0:
            subgraph = self.get_subgraph(self.graph)
            ord_lanes = [lane for lane in subgraph.nodes]
            A_down, A_up, A_neigh = self.get_all_A(subgraph, ord_lanes)
            A_list = [A_down, A_up, A_neigh]
        else:
            A_list = None #no A needs to be exported

        X, Y = self.calc_X_and_Y(ord_lanes,
                                 df_detector,
                                 df_liu_results,
                                 start,
                                 end,
                                 average_interval,
                                 sample_size,
                                 interpolate_ground_truth,
                                 sample_time_sequence = sample_time_sequence)

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
        Takes the dataframe from the e1 detectors and liu results as input and
        gives the X and Y back.

        '''

        for lane, cnt_lane in zip(ord_lanes, range(len(ord_lanes))):
            # faster solution: just access df once and store data in arrays
            lane_id = lane.replace('/', '-')
            stopbar_detector_id = 'e1_' + str(lane_id) + '_0'
            adv_detector_id = 'e1_' + str(lane_id) + '_1'
            e2_detector_id = 'e2_' + str(lane_id) + '_0'

            if cnt_lane == 0:
                #dimesion: time x nodes x features
                num_rows = (len(df_detector.loc[start_time:end_time,
                                    (stopbar_detector_id, 'nVehContrib')]) -1)
                # (-1):last row is belongs to the following average interval
                duration_in_sec = end_time-start_time
                X_unsampled = np.zeros((num_rows, len(ord_lanes), 8))
                Y_unsampled = np.zeros((num_rows, len(ord_lanes), 2))

            X_unsampled[:, cnt_lane, 0] = df_detector.loc[
                                        start_time:end_time-average_interval,
                                        (stopbar_detector_id, 'nVehContrib')]
            X_unsampled[:, cnt_lane, 1] = df_detector.loc[
                                        start_time:end_time-average_interval,
                                        (stopbar_detector_id, 'occupancy')]
            X_unsampled[:, cnt_lane, 2] = df_detector.loc[
                                        start_time:end_time-average_interval,
                                        (stopbar_detector_id, 'speed')]
            X_unsampled[:, cnt_lane, 3] = df_detector.loc[
                                        start_time:end_time-average_interval,
                                        (adv_detector_id, 'nVehContrib')]
            X_unsampled[:, cnt_lane, 4] = df_detector.loc[
                                        start_time:end_time-average_interval,
                                        (adv_detector_id, 'occupancy')]
            X_unsampled[:, cnt_lane, 5] = df_detector.loc[
                                        start_time:end_time-average_interval,
                                        (adv_detector_id, 'speed')]
            X_unsampled[:, cnt_lane, 6] = self.get_tls_binary_signal(
                                                            start_time,
                                                            duration_in_sec,
                                                            lane,
                                                            average_interval,
                                                            num_rows)
            X_unsampled[:, cnt_lane, 7] = self.get_ground_truth_array(
                        start_time,
                        duration_in_sec,
                        lane,
                        average_interval,
                        num_rows,   #use liu results as features
                        interpolate_ground_truth = interpolate_ground_truth,
                        ground_truth_name = 'estimated hybrid')

            Y_unsampled[:, cnt_lane, 0] = self.get_ground_truth_array(
                        start_time,
                        duration_in_sec,
                        lane,
                        average_interval,
                        num_rows,
                        interpolate_ground_truth = interpolate_ground_truth,
                        ground_truth_name = 'ground-truth')

            Y_unsampled[:, cnt_lane, 1] = self.get_nVehSeen_from_e2(
                                                            e2_detector_id,
                                                            start_time,
                                                            duration_in_sec,
                                                            average_interval,
                                                            num_rows)

        if sample_time_sequence:
            num_samples = math.ceil(X_unsampled.shape[0]/sample_size)
            _logger.info('number of samples: %g', num_samples)
            X_sampled = np.zeros((num_samples, sample_size, len(ord_lanes), 8))
            Y_sampled = np.zeros((num_samples, sample_size, len(ord_lanes), 2))

            for cnt_sample in range(num_samples):
                s_start = cnt_sample*sample_size
                s_end = s_start + sample_size
                if (X_unsampled[s_start:s_end][:][:].shape[0] == sample_size):
                    X_sampled[cnt_sample][:][:][:] = X_unsampled[
                                                        s_start:s_end][:][:]
                    Y_sampled[cnt_sample][:][:][:] = Y_unsampled[
                                                        s_start:s_end][:][:]
                else:
                    # implement zero padding
                    diff_samples = (sample_size -
                                    X_unsampled[s_start:s_end][:][:].shape[0])
                    X_zero_padding = np.vstack((
                                        X_unsampled[s_start:s_end][:][:],
                                        np.zeros((diff_samples, len(ord_lanes),
                                        8))))
                    Y_zero_padding = np.vstack((
                                        Y_unsampled[s_start:s_end][:][:],
                                        np.zeros((diff_samples, len(ord_lanes),
                                        2))))
                    X_sampled[cnt_sample][:][:][:] = X_zero_padding
                    Y_sampled[cnt_sample][:][:][:] = Y_zero_padding

            #dimension (num_samples x num_timesteps x num_lanes x num_features)
            return X_sampled, Y_sampled
        else:
            #reshape arrays, that they have the same output shape
            #than sampeled X and Y
            X_reshaped = np.reshape(X_unsampled, (1,
                                                  X_unsampled.shape[0],
                                                  X_unsampled.shape[1],
                                                  X_unsampled.shape[2]))
            Y_reshaped = np.reshape(Y_unsampled, (1,
                                                  Y_unsampled.shape[0],
                                                  Y_unsampled.shape[1],
                                                  Y_unsampled.shape[2]))
            #dimension ( 1 x num_timesteps x num_lanes x num_features)
            return X_reshaped, Y_reshaped

    def get_subgraph(self, graph):
        node_sublist = [lane[0] for lane in self.graph.nodes(data=True)
                                                            if lane[1] != {}]
        #print('node_sublist:', node_sublist)
        sub_graph = graph.subgraph(node_sublist)
        return sub_graph

    def get_ground_truth_array(self,
                               start_time,
                               duration_in_sec,
                               lane_id, average_interval,
                               num_rows,
                               interpolate_ground_truth= False,
                               ground_truth_name = 'ground-truth'):

        """Returns the array (either ground truth or liu results) for
        a specific lane for a time period for an specific interval

        "start time" corresponds to start of the ground truth data, int
        "duration_in_sec" duration of the time period in seconds, int
        "lane_id" lane_id for the specific lane, str
        "average_interval" interval over which the ground truth data are
            averaged, int
        "num_rows" number of rows in the X vector -> ensures, that no
            missmatch error occurs, int
        "interpolate_ground_truth": if ground truth data are as a step function
            in the array or if they are linear interpolated, bool
        "ground_truth_name": can be either "ground-truth" or "estimated hybrid"
            or "estimated pure liu", str

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

            ground_truth = self.df_liu_results.loc[
                                        phase, (lane_id, ground_truth_name)]
            if interpolate_ground_truth == False:
                temp_arr_ground_truth = np.full((int(phase_end-phase_start), 1),
                                                ground_truth)
            else:
                temp_arr_ground_truth = np.linspace(
                                            previous_ground_truth,
                                            ground_truth,
                                            num = int(phase_end-phase_start))
                temp_arr_ground_truth = np.reshape(
                                            temp_arr_ground_truth,
                                            (int(phase_end-phase_start), 1))

            if arr_ground_truth.size == 0:
                arr_ground_truth = temp_arr_ground_truth
            else:
                arr_ground_truth = np.vstack((arr_ground_truth,
                                              temp_arr_ground_truth))

            previous_ground_truth = ground_truth

        #crop the right time-window in seconds

        lower_bound = int(start_time)-int(first_phase_start)
        upper_bound = int(start_time)+duration_in_sec-int(first_phase_start)
        arr_ground_truth_1second = np.reshape(
                                arr_ground_truth[lower_bound:upper_bound, 0],
                                upper_bound-lower_bound)

        #just take the points of data from the average -interval
        #example: average interval 5 sec -> take one value every five seconds!
        arr_ground_truth_average_interval = arr_ground_truth_1second[0::average_interval]
        #make sure that no dimension problem occurs
        arr_ground_truth_average_interval = arr_ground_truth_average_interval[0:num_rows]
        return arr_ground_truth_average_interval



    def get_tls_binary_signal(self,
                              start_time,
                              duration_in_sec,
                              lane_id,
                              average_interval,
                              num_rows):

        time_lane = self.df_liu_results.loc[:, (lane_id, 'time')]
        arr_tls_binary = np.array([])
        for phase in range(len(time_lane)):
            phase_start = self.df_liu_results.loc[phase,
                                                  (lane_id, 'phase start')]
            phase_end = self.df_liu_results.loc[phase,
                                                (lane_id, 'phase end')]
            green_start = self.df_liu_results.loc[phase,
                                                  (lane_id, 'tls start')]
            if phase == 0:
                first_phase_start = phase_start

            #red phase from phase start to green start (0)
            temp_array_red_phase = np.full((int(green_start-phase_start), 1), 0)
            #green phase from green start to phase end (1)
            temp_array_green_phase = np.full((int(phase_end-green_start), 1), 1)
            if arr_tls_binary.size == 0:
                arr_tls_binary = temp_array_red_phase
                arr_tls_binary = np.vstack((arr_tls_binary,
                                            temp_array_green_phase))
            else:
                arr_tls_binary =  np.vstack((arr_tls_binary,
                                             temp_array_red_phase))
                arr_tls_binary = np.vstack((arr_tls_binary,
                                            temp_array_green_phase))

        lower_bound = int(start_time)-int(first_phase_start)
        upper_bound = int(start_time)+duration_in_sec-int(first_phase_start)
        arr_tls_binary_1second = np.reshape(
                                    arr_tls_binary[lower_bound:upper_bound, 0],
                                    upper_bound-lower_bound)
        #just take the points of data from the average -interval
        #example: average interval 5 sec -> take one value every five seconds!
        arr_tls_binary_average_interval = arr_tls_binary_1second[0::average_interval]
        #make sure, that no dimension problems occur
        arr_tls_binary_average_interval = arr_tls_binary_average_interval[0:num_rows]
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
            #10seconds reserve, otherwise complications occur
            end_time = min(list_end_time)-10
            #make sure, that end time is matching with average_interval
            end_time = int(end_time/average_interval) * average_interval
            self.preprocess_end_time = end_time
            return end_time

    def get_A_for_neighboring_lanes(self, order_lanes):
        #list with tuple of indexes (input_lane_index, neighbor_lane_index)
        indexes = []
        for lane_id, input_lane_index in zip(order_lanes, range(len(order_lanes))):
            neighbor_lanes = self.sumo_network.get_neighboring_lanes(
                                                    lane_id,
                                                    include_input_lane=True)
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
        A_neighbors = self.get_A_for_neighboring_lanes(order_lanes)
        A_neighbors = np.eye(N, N) + A_neighbors
        A_neighbors = np.minimum(A_neighbors, np.ones((N,N)))

        return A_down, A_up, A_neighbors


def _prune_det_id_in_colnames(colnames, lane_id):
    return [re.sub(f'{lane_id}_', '', col) for col in colnames]


def _concat_colnames(cols):
    return ['/'.join(col) if not isinstance(col, six.string_types) else col
            for col in cols]
