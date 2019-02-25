import collections
import os
import re
import warnings

import numpy as np
import pandas as pd
import tables
from tables import NoSuchNodeError

from trafficgraphnn.utils import (E1IterParseWrapper, E2IterParseWrapper,
                                  TLSSwitchIterParseWrapper, _col_dtype_key,
                                  pairwise_exhaustive)


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


def sumo_output_xmls_to_hdf(output_dir,
                            hdf_filename='raw_xml.hdf',
                            complevel=5,
                            complib='blosc:lz4'):
    file_list = [os.path.join(output_dir, f) for f in os.listdir(output_dir)]
    file_list = [f for f in file_list
                 if os.path.isfile(f) and os.path.splitext(f)[-1] == '.xml']
    output_filename = os.path.join(output_dir, hdf_filename)
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


def light_timing_xml_to_df(xml_file):
    parser = TLSSwitchIterParseWrapper(xml_file, True)
    data = [dict(e.attrib) for e in parser.iterate_until(np.inf)]
    df = pd.DataFrame(data)
    df = df.astype({col: _col_dtype_key[col] for col in df.columns})

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
    green_starts = phase_starts[phase_starts == True]
    return green_starts


def red_times_from_lane_light_df(lane_df):
    phase_starts = lane_df[(lane_df.shift() != lane_df)]
    red_starts = phase_starts[phase_starts == False]
    return red_starts


def queueing_intervals_from_lane_light_df(lane_df):
    red_starts = red_times_from_lane_light_df(lane_df)
    queueing_intervals = pairwise_exhaustive(red_starts.index)
    return queueing_intervals


def get_preprocessed_filenames(directory):
    try:
        return [os.path.join(directory, f)
                for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f))
                and re.match(r'sim_\d+.h5', os.path.basename(f))]
    except FileNotFoundError:
        return []


def get_sim_numbers_in_preprocess_store(store, lane_list=None):
    if lane_list is None:
        lane_list = store['A_downstream'].columns

    def sim_numbers_for_lane(lane):
            X_subelements = dir(store.root.__getattr__(lane).X)
            samples = filter(lambda e: re.match(r'_\d{4,5}', e), X_subelements)
            return list(map(lambda e: int(re.search(r'\d{4,5}', e).group()), samples))

    def sim_numbers_for_lane_2(lane):
        query_string = '{}/X/_{:04}'
        i = 1
        try:
            while True:
                X = store.get(query_string.format(lane, i))
                i += 1
        except KeyError:
            pass
        return list(range(1, i))

    try:
        sim_numbers = sim_numbers_for_lane_2(list(lane_list)[0])
    except NoSuchNodeError:
        return []

    # assert all((sim_numbers_for_lane_2(lane) == sim_numbers for lane in lane_list))
    return sim_numbers
