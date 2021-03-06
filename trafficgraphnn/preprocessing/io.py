import collections
import logging
import multiprocessing
import os
import re
import warnings

import numpy as np
import pandas as pd
import tables

from trafficgraphnn.utils import (E1IterParseWrapper, E2IterParseWrapper,
                                  TLSSwitchIterParseWrapper, _col_dtype_key,
                                  col_type, pairwise_iterate)

_logger = logging.getLogger(__name__)


def write_hdf_for_sumo_network(sumo_network, multiprocess=True):
    output_dir = os.path.join(os.path.dirname(sumo_network.netfile),
                              'output')
    if multiprocess:
        output_hdf = sumo_output_xmls_to_hdf_multiprocess(output_dir)
    else:
        output_hdf = sumo_output_xmls_to_hdf(output_dir)

    light_switch_out_files = light_switch_out_files_for_sumo_network(
        sumo_network)

    assert len(light_switch_out_files) == 1 # more than one xml not supported yet

    tls_output_xml_to_hdf(light_switch_out_files.pop())

    return output_hdf


def light_switch_out_files_for_sumo_network(sumo_network):
    light_switch_out_files = set()
    for edge in sumo_network.graph.edges.data():
        try:
            light_switch_out_files.add(
                os.path.join(os.path.dirname(sumo_network.netfile),
                edge[-1]['tls_output_info']['dest']))
        except KeyError:
            continue
    return light_switch_out_files


def _append_to_store(store, buffer, all_ids):
    converter = {col: _col_dtype_key[col]
                      for col in buffer.keys()
                      if col in _col_dtype_key}
    df = pd.DataFrame.from_dict(buffer)
    df = df.astype(converter)
    df = df.set_index('begin')
    for i in all_ids:
        # sub_df = df.loc[df['id'] == i]
        # sub_df = sub_df.set_index('begin')
        sub_df = df.query(f"id == '{i}'")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', tables.NaturalNameWarning)
            store.append(f'raw_xml/{i}', sub_df)
        assert len(store[f'raw_xml/{i}'].loc[0].shape) == 1, \
            'id %s has len(store[id].loc[0].shape) = %g' % (i, len(store[f'raw_xml/{i}'].loc[0].shape))


def xml_to_df_hdf(parser,
                  store_filename,
                  hdf_file_mode='a',
                  complevel=7,
                  complib='zlib',
                  start_time=0,
                  end_time=np.inf,
                  buffer_size=1e5):
    buffer = collections.defaultdict(list)
    i = 0
    all_ids = set()
    with pd.HDFStore(store_filename, hdf_file_mode, complevel=complevel,
                     complib=complib) as store:
        for _ in parser.iterate_until(start_time):
            pass
        for row in parser.iterate_until(end_time):
            for k, v in row.items():
                buffer[k].append(v)
            all_ids.add(row.get('id'))
            i += 1
            if i >= buffer_size:
                _append_to_store(store, buffer, all_ids)
                buffer = collections.defaultdict(list)
                i = 0
        _append_to_store(store, buffer, all_ids)


def detector_output_xml_to_df(xml_filename):
    basename = os.path.basename(xml_filename)
    if '_e1_' in basename:
        parser = E1IterParseWrapper(xml_filename, True)
    elif '_e2_' in basename:
        parser = E2IterParseWrapper(xml_filename, True)
    else:
        return

    rows = collections.defaultdict(list)
    for row in parser.iterate_until(np.inf):
        for k, v in row.items():
            rows[k].append(v)

    id_set = set(rows['id'])
    # sanity check there is only one detector
    assert len(id_set) == 1

    det_id = id_set.pop()
    converter = {col: _col_dtype_key[col]
                      for col in rows.keys()
                      if col in _col_dtype_key}

    df = (pd.DataFrame.from_dict(rows)
                      .astype(converter)
                      .drop(columns='id')
                      .set_index('begin'))
    return {det_id: df}


def sumo_output_xmls_to_hdf_multiprocess(output_dir,
                                         hdf_filename='raw_xml.hdf',
                                         complevel=5,
                                         complib='blosc:lz4',
                                         num_workers=None,
                                         remove_old_if_exists=True):
    file_list = output_files_in_dir(output_dir)
    output_filename = os.path.join(output_dir, hdf_filename)

    if (remove_old_if_exists and os.path.exists(output_filename)
            and os.path.isfile(output_filename)):
        os.remove(output_filename)
        _logger.debug('Removed file %s for new one', output_filename)

    with multiprocessing.Pool(num_workers) as pool:
        dfs = pool.map(detector_output_xml_to_df, file_list)

    dfs = {k: v for d in dfs if d is not None for k, v in d.items()}

    with pd.HDFStore(output_filename, complevel=complevel,
                     complib=complib) as store:
        for det_id, df in dfs.items():
            store.append('raw_xml/{}'.format(det_id), df)

    return output_filename


def output_files_in_dir(output_dir):
    file_list = [os.path.join(output_dir, f) for f in os.listdir(output_dir)]
    file_list = [f for f in file_list
                 if os.path.isfile(f) and os.path.splitext(f)[-1] == '.xml']
    return file_list


def sumo_output_xmls_to_hdf(output_dir,
                            hdf_filename='raw_xml.hdf',
                            complevel=5,
                            complib='blosc:lz4',
                            remove_old_if_exists=True):
    file_list = output_files_in_dir(output_dir)
    output_filename = os.path.join(output_dir, hdf_filename)

    if (remove_old_if_exists and os.path.exists(output_filename)
            and os.path.isfile(output_filename)):
        os.remove(output_filename)
        _logger.debug('Removed file %s for new one', output_filename)

    for filename in file_list:
        basename = os.path.basename(filename)
        if '_e1_' in basename:
            parser = E1IterParseWrapper(filename, True)
        elif '_e2_' in basename:
            parser = E2IterParseWrapper(filename, True)
        else:
            continue
        xml_to_df_hdf(parser, output_filename, complevel=complevel,
                      complib=complib)
    return output_filename


def tls_output_xml_to_hdf(xml_file,
                          hdf_filename='raw_xml.hdf',
                          complevel=5,
                          complib='blosc:lz4'):

    df = light_timing_xml_to_phase_df(xml_file)
    file_dir = os.path.dirname(xml_file)
    hdf_filename = os.path.join(file_dir, hdf_filename)
    with pd.HDFStore(hdf_filename, complevel=complevel,
                     complib=complib) as store:
        store.append('raw_xml/tls_switch', df, append=False)

    return df


def light_timing_xml_to_phase_df(xml_file):
    parser = TLSSwitchIterParseWrapper(xml_file, True)
    data = [dict(e.attrib) for e in parser.iterate_until(np.inf)]
    df = pd.DataFrame(data)
    df = df.astype({col: col_type(col) for col in df.columns})

    max_time = df.end.max()
    out_df = pd.DataFrame(index=pd.Index(np.arange(0, max_time), name='begin'))

    for lane in df.fromLane.unique():
        subdf = df[df.fromLane == lane].sort_values(['begin', 'end'])
        ints = subdf.groupby(
            (subdf.end.shift()-subdf.begin).lt(0).cumsum()).agg(
                {'begin': 'first', 'end': 'last'})
        intervals = [pd.Interval(i.begin, i.end, 'left')
                     for i in ints.itertuples()]
        out_df[lane] = [any(t in i for i in intervals) for t in out_df.index]

    return out_df


def green_times_from_lane_light_df(lane_df):
    phase_starts = lane_df[(lane_df.shift() != lane_df)]
    green_starts = phase_starts[phase_starts == True].index
    return green_starts


def red_times_from_lane_light_df(lane_df):
    phase_starts = lane_df[(lane_df.shift() != lane_df)]
    red_starts = phase_starts[phase_starts == False].index
    return red_starts


def queueing_intervals_from_lane_light_df(lane_df):
    red_starts = red_times_from_lane_light_df(lane_df)
    queueing_intervals = pairwise_iterate(red_starts)
    return queueing_intervals


def green_phase_start_ends_from_lane_light_df(lane_df):
    green_starts = green_times_from_lane_light_df(lane_df)
    red_starts = red_times_from_lane_light_df(lane_df)

    # if the light starts on red skip that one
    if red_starts[0] < green_starts[0]:
        red_starts = red_starts[1:]
    return zip(green_starts, red_starts)


def get_preprocessed_filenames(directory):
    try:
        return [os.path.join(directory, f)
                for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f))
                and re.match(r'\d+.h5', os.path.basename(f))]
    except FileNotFoundError:
        return []
