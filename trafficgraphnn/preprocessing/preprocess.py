"""Preprocessing code for Sumo outputs.

Use to put data into hdf stores with A, X, Y arrays.
"""
import logging
import multiprocessing
import os
import re
import time
from collections import OrderedDict
from itertools import repeat

import networkx as nx
import pandas as pd
import six

from trafficgraphnn.load_data import pad_value_for_feature
from trafficgraphnn.preprocessing.io import (get_preprocessed_filenames,
                                             light_switch_out_files_for_sumo_network,
                                             light_timing_xml_to_phase_df,
                                             write_hdf_for_sumo_network)
from trafficgraphnn.preprocessing.liumethod_new import liu_method_for_net

raw_xml_x_feature_defaults=['occupancy', 'speed', 'green', 'liu_estimated_veh']
raw_xml_y_feature_defaults=['nVehSeen', 'maxJamLengthInVehicles']


_logger = logging.getLogger(__name__)


def run_preprocessing(sumo_network, output_filename=None):
    if output_filename is None:
        output_filename = os.path.join(
            os.path.dirname(sumo_network.netfile),
            'preprocessed_data',
            '{:04}.h5').format(_next_file_number(sumo_network))
    t0 = time.time()
    hdf_filename = write_hdf_for_sumo_network(sumo_network)
    t = time.time() - t0
    _logger.debug('Extracting xml took {} s'.format(t))
    t0 = time.time()
    write_per_lane_tables(output_filename, sumo_network, hdf_filename)
    t - time.time() - t0
    _logger.debug('Writing preprocessed data took {} s'.format(t))
    return output_filename


def write_per_lane_tables(output_filename,
                          sumo_network,
                          raw_xml_filename=None,
                          X_features=raw_xml_x_feature_defaults,
                          Y_features=raw_xml_y_feature_defaults,
                          complib='blosc:lz4', complevel=5):
    """Write an hdf file with per-lane X and Y data arrays"""

    X_df, Y_df = build_X_Y_tables_for_lanes(
        sumo_network, raw_xml_filename=raw_xml_filename, X_features=X_features,
        Y_features=Y_features)

    lanes_with_data = X_df.index.get_level_values(0).unique()
    assert len(lanes_with_data.difference(
               Y_df.index.get_level_values(0)).unique()) == 0

    A_dfs = build_A_tables_for_lanes(sumo_network, lanes_with_data)
    A_df = pd.concat(A_dfs, axis=1)

    if not os.path.isdir(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename))

    with pd.HDFStore(output_filename, 'w', complevel=complevel,
                     complib=complib) as store:
        store.put('X', X_df)
        store.put('Y', Y_df)
        store.put('A', A_df)


def build_A_tables_for_lanes(sumo_network, lanes=None):
    """Returns dict of dataframes for different lane adjacency matrices"""
    if lanes is None:
        lanes = sumo_network.lanes_with_detectors()

    A_dfs = {}
    A_dfs['A_downstream'] = nx.to_pandas_adjacency(
        sumo_network.graph, nodelist=lanes).astype('bool')

    A_dfs['A_upstream'] = nx.to_pandas_adjacency(
        sumo_network.graph.reverse(), nodelist=lanes).astype('bool')

    A_dfs['A_neighbors'] = nx.to_pandas_adjacency(
        sumo_network.get_lane_graph_for_neighboring_lanes(
            include_self_adjacency=False),
        nodelist=lanes).astype('bool')

    A_dfs['A_turn_movements'] = nx.to_pandas_adjacency(
        sumo_network.get_lane_graph_for_turn_movements(),
        nodelist=lanes).astype('bool')

    A_dfs['A_through_movements'] = nx.to_pandas_adjacency(
        sumo_network.get_lane_graph_for_thru_movements(),
        nodelist=lanes).astype('bool')

    return A_dfs


def build_X_Y_tables_for_lanes(sumo_network,
                               lane_subset=None,
                               raw_xml_filename=None,
                               X_features=raw_xml_x_feature_defaults,
                               Y_features=raw_xml_y_feature_defaults,
                               num_workers=None,
                               clip_ending_pad_timesteps=True):
    """Return per-lane dataframe for X and Y with specified feature sets"""
    # default to all lanes
    if lane_subset is None:
        lane_subset = sumo_network.lanes_with_detectors()
    if raw_xml_filename is None:
        raw_xml_filename = os.path.join(os.path.dirname(sumo_network.netfile),
                                        'output', 'raw_xml.hdf')

    if 'green' in X_features:
        green_df = per_lane_green_series_for_sumo_network(sumo_network)
        green_serieses = {lane_id: green_df.loc[:, lane_id].rename('green')
                          for lane_id in lane_subset}
    else:
        green_serieses = {}

    # run liu method if needed
    if len(set(['liu_estimated_m', 'liu_estimated_veh']).intersection(
               X_features)) > 0:
        liu_result_df = liu_method_for_net(sumo_network, raw_xml_filename)
        liu_serieses = {lane_id: __get_liu_series(liu_result_df, lane_id)
                        for lane_id in lane_subset}
    else:
        liu_serieses = {}

    with multiprocessing.Pool(num_workers) as pool:
        dfs = pool.starmap(
            _X_Y_dfs_for_lane,
            zip(repeat(raw_xml_filename),
                lane_subset,
                [sumo_network.graph.nodes[lane]['detectors']
                 for lane in lane_subset],
                repeat(X_features),
                repeat(Y_features),
                [green_serieses[lane] if lane in green_serieses else None
                 for lane in lane_subset],
                [liu_serieses[lane] if lane in liu_serieses else None
                 for lane in lane_subset])
        )
        X_dfs, Y_dfs = zip(*dfs)

        X_df = pd.concat(X_dfs, join='outer', keys=lane_subset,
                         names=['lane', 'begin'])
        Y_df = pd.concat(Y_dfs, join='outer', keys=lane_subset,
                         names=['lane', 'begin'])

        # fill in blanks
        if 'green' in X_features:
            X_df['green'] = X_df['green'].ffill()

        X_df = X_df.fillna(pad_value_for_feature)
        Y_df = Y_df.fillna(pad_value_for_feature)

        if clip_ending_pad_timesteps:
            clip_after = max(map(last_nonpad_timestep, [X_df, Y_df]))
            X_df = X_df.loc[pd.IndexSlice[:, :clip_after], :]
            Y_df = Y_df.loc[pd.IndexSlice[:, :clip_after], :]

    return X_df, Y_df


def _X_Y_dfs_for_lane(filename, lane_id, detector_dict,
                      X_features, Y_features, green_series=None,
                      liu_series=None):
    with pd.HDFStore(filename, 'r') as input_store:
        # e1 (loop) detector data
        e1_detectors = lane_detectors_of_type_sorted_by_position(
            detector_dict, 'e1')
        e1_dfs = OrderedDict((det_id,
                                input_store['raw_xml/{}'.format(det_id)])
                                for det_id in e1_detectors)
        X_lane_dfs_e1 = __get_dfs_feats(e1_dfs, X_features, lane_id)
        Y_lane_dfs_e1 = __get_dfs_feats(e1_dfs, Y_features, lane_id)

        # e2 (lane area) detector data
        e2_detectors = lane_detectors_of_type_sorted_by_position(
            detector_dict, 'e2')
        e2_dfs = OrderedDict((det_id,
                                input_store['raw_xml/{}'.format(det_id)])
                                for det_id in e2_detectors)
        X_lane_dfs_e2 = __get_dfs_feats(e2_dfs, X_features, lane_id)
        Y_lane_dfs_e2 = __get_dfs_feats(e2_dfs, Y_features, lane_id)

        X_lane_dfs = [*X_lane_dfs_e1, *X_lane_dfs_e2]
        Y_lane_dfs = [*Y_lane_dfs_e1, *Y_lane_dfs_e2]

        # add some more X features if needed
        X_lane_dfs.append(green_series) # no-op if these are None
        X_lane_dfs.append(liu_series)

        # append columns (features) together
        X_lane_df = pd.concat(X_lane_dfs, axis=1)
        Y_lane_df = pd.concat(Y_lane_dfs, axis=1)

        return X_lane_df, Y_lane_df


def last_nonpad_timestep(df):
    features = [feat for feat in df.columns if feat in pad_value_for_feature]
    try:
        features.remove('green')
    except ValueError:
        pass

    test_series = pd.Series([pad_value_for_feature[feat] for feat in features],
                            index=features)

    is_pad_row = (df.loc[:, features] == test_series).all(axis=1)
    last_nonpad = is_pad_row.index[~is_pad_row].get_level_values('begin').max()

    return last_nonpad


def __get_dfs_feats(det_dfs, features, lane_id):
    subdfs = OrderedDict((det_id,
                          df[[feat for feat in features if feat in df]])
                         for det_id, df in det_dfs.items())
    out = __det_df_dict_to_dfs_with_prefixed_columns(subdfs, lane_id)
    return out


def __det_df_dict_to_dfs_with_prefixed_columns(df_dict, lane_id):
    dfs = []
    for det_id, df in df_dict.items():
        prefix = _det_id_minus_lane_id(det_id, lane_id) + '/'
        new_colnames = prefix + df.columns
        df = df.rename(columns={col: newcol for col, newcol
                                in zip(df.columns, new_colnames)})
        dfs.append(df)
    return dfs


def __get_liu_series(liu_results, lane_id):
    df = liu_results[lane_id]
    try:
        df = df[lane_id] # if columns are multiindex
    except KeyError:
        pass
    df = df.loc[:,('estimate')]
    df = df.rename('liu_estimated_veh')
    return df


def per_lane_green_series_for_sumo_network(sumo_network):
    light_switch_out_files = light_switch_out_files_for_sumo_network(
        sumo_network)
    assert len(light_switch_out_files) == 1
    light_timing_df = light_timing_xml_to_phase_df(
        light_switch_out_files.pop())
    return light_timing_df


def lane_detectors_of_type_sorted_by_position(lane_dict, det_type):
    if det_type == 'e1':
        det_str = 'e1Detector'
    elif det_type == 'e2':
        det_str = 'e2Detector'
    else:
        raise ValueError('Unknown detector type {}'.format(det_type))

    dets_of_type = [v for v in lane_dict.values() if v['type'] == det_str]
    dets_of_type.sort(key=lambda x: x['pos'])
    dets_of_type = [det['id'] for det in dets_of_type]
    return dets_of_type


def _next_file_number(sumo_network):
    data_dir = os.path.join(os.path.dirname(sumo_network.netfile),
                            'preprocessed_data')

    files = get_preprocessed_filenames(data_dir)
    fnames = [os.path.basename(f) for f in files]
    numbers = [int(re.search(r'\d+', f).group()) for f in fnames]

    return max(numbers, default=0) + 1


def _prune_det_id_in_colnames(colnames, lane_id):
    return [_det_id_minus_lane_id(col, lane_id) for col in colnames]


def _det_id_minus_lane_id(det_id, lane_id):
    return re.sub(f'{lane_id}_', '', det_id)


def _concat_colnames(cols):
    return ['/'.join(col) if not isinstance(col, six.string_types) else col
            for col in cols]
