import logging
import os
import tempfile
import time
from collections import defaultdict
from contextlib import ExitStack
from itertools import zip_longest

import numpy as np
import pandas as pd
import six

from trafficgraphnn.utils import (flatten, get_preprocessed_filenames,
                                  get_sim_numbers_in_preprocess_store, iterfy,
                                  string_list_decode)

_logger = logging.getLogger(__name__)

pad_value_for_feature = defaultdict(lambda: 0,
                                    occupancy=0.,
                                    speed=-1.,
                                    liu_estimated=-1.,
                                    green=0.,
                                    nVehSeen=0.,
                                    maxJamLengthInMeters=-1.,
                                   )

pad_value_for_feature.update(
    [('e1_0/occupancy', 0.),
    ('e1_0/speed', -1.),
    ('e1_1/occupancy', 0.),
    ('e1_1/speed', -1.),
    ('e2_0/nVehSeen', 0.),
    ('e2_0/maxJamLengthInMeters', -1.)])

All_A_name_list = ['A_downstream', 'A_upstream', 'A_neighbors',
                   'A_turn_movements', 'A_through_movements']


class Batch(object):
    """Convenience class for holding a batch of sim readers.
    """

    def __init__(self,
                 filenames,
                 sim_indeces,
                 window_size,
                 A_name_list=['A_downstream',
                              'A_upstream',
                              'A_neighbors'],
                 x_feature_subset=['e1_0/occupancy',
                                   'e1_0/speed',
                                   'e1_1/occupancy',
                                   'e1_1/speed',
                                   'liu_estimated',
                                   'green'],
                 y_feature_subset=['e2_0/nVehSeen',
                                   'e2_0/maxJamLengthInMeters']):
        self.A_name_list = A_name_list
        self.x_feature_subset = x_feature_subset
        self.y_feature_subset = y_feature_subset

        self.pad_scalars = get_pad_scalars(self.x_feature_subset, self.y_feature_subset)

        if isinstance(filenames, np.ndarray):
            filenames = filenames.tolist()
        if isinstance(sim_indeces, np.ndarray):
            sim_indeces = sim_indeces.tolist()

        self.readers = windowed_batch_of_generators(
            filenames, sim_indeces, window_size, A_name_list, x_feature_subset,
            y_feature_subset)

        self._batch_sources = [f'{filename}-{sim_number}'
                               for filename, sim_number
                               in zip(filenames, sim_indeces)]

    def iterate(self, try_broadcast_A=False):
        for output in self.readers:
            yield pad_and_stack_batch(output,
                                      self.pad_scalars,
                                      try_broadcast_A=try_broadcast_A)

    @classmethod
    def from_file(cls,
                  filename,
                  window_size,
                  sim_subset=None,
                  A_name_list=['A_downstream',
                               'A_upstream',
                               'A_neighbors'],
                  x_feature_subset=['e1_0/occupancy',
                                    'e1_0/speed',
                                    'e1_1/occupancy',
                                    'e1_1/speed',
                                    'liu_estimated',
                                    'green'],
                  y_feature_subset=['e2_0/nVehSeen',
                                    'e2_0/maxJamLengthInMeters']):
        if sim_subset is None:
            sim_subset = get_sim_numbers_in_file(filename)
        return cls([filename] * len(sim_subset), sim_subset, window_size,
                   A_name_list=A_name_list, x_feature_subset=x_feature_subset,
                   y_feature_subset=y_feature_subset)


def batches_from_directories(directories,
                             batch_size,
                             window_size,
                             shuffle=True,
                             A_name_list=['A_downstream',
                                          'A_upstream',
                                          'A_neighbors'],
                             x_feature_subset=['e1_0/occupancy',
                                               'e1_0/speed',
                                               'e1_1/occupancy',
                                               'e1_1/speed',
                                               'liu_estimated',
                                               'green'],
                             y_feature_subset=['e2_0/nVehSeen',
                                               'e2_0/maxJamLengthInMeters']):

    file_and_sims = get_file_and_sim_indeces_in_dirs(directories)

    if shuffle:
        np.random.shuffle(file_and_sims)

    filenames, sim_indeces = list(map(list, zip(*file_and_sims)))

    batched_filenames = [filenames[i:i + batch_size]
                         for i in range(0, len(filenames), batch_size)]
    batched_sim_indeces = [sim_indeces[i:i + batch_size]
                         for i in range(0, len(sim_indeces), batch_size)]

    for f_batch, si_batch in zip(batched_filenames, batched_sim_indeces):
        batch = Batch(f_batch, si_batch, window_size, A_name_list=A_name_list,
                      x_feature_subset=x_feature_subset,
                      y_feature_subset=y_feature_subset)
        yield batch


def get_file_and_sim_indeces_in_dirs(directories):
    tstart = time.time()
    directories = iterfy(directories)
    filenames = flatten([get_preprocessed_filenames(directory)
                         for directory in directories])
    file_and_sims = []

    for filename in filenames:
        for sim_number in get_sim_numbers_in_file(filename):
            file_and_sims.append((filename, sim_number))

    t = time.time() - tstart
    _logger.debug('getting file and sim indeces took %s s', t)
    return file_and_sims


def get_pad_scalars(x_feature_subset, y_feature_subset):
    A_padding_scalar = False
    x_padding_scalars = [get_pad_value_for_feature(feat)
                         for feat in x_feature_subset]
    y_padding_scalars = [get_pad_value_for_feature(feat)
                         for feat in y_feature_subset]
    return A_padding_scalar, x_padding_scalars, y_padding_scalars


def max_num_lanes_in_batch(output_list):
    """Computes the dimensions to pad all outputs to. This will depend on the
    maximum number of lanes of all simulations in the batch.

    """
    A_matrix_size_per_gen = [np.shape(gen[0]) for gen in output_list
                             if gen is not None]

    # sanity check: make sure adjacency matrices are square
    assert all(A_shape[-1] == A_shape[-2] for A_shape in A_matrix_size_per_gen)
    max_num_lanes = max([A_shape[-2] for A_shape in A_matrix_size_per_gen],
                        default=0)
    return max_num_lanes


def get_max_A_depth(output_list):
    """Returns the max depth (number of edge types) in the batch"""
    A_matrix_size_per_gen = [np.shape(gen[0]) for gen in output_list
                             if gen is not None]

    return max([A_shape[-3] for A_shape in A_matrix_size_per_gen], default=0)


def can_broadcast_A_matrices_over_time(output_list):
    # Do the A matrices vary across timesteps or, if they are all equal, can we
    # have a time dimension of 1?
    A_matrices_by_gen = [[variables[0] for variables in output]
                         for output in output_list]
    all_same_As = all(np.array_equal(batch_A[0], A)
                      for batch_A in A_matrices_by_gen for A in batch_A)
    return all_same_As


def pad_and_stack_batch(outputs, pad_scalars, try_broadcast_A=False):
    num_timesteps_per_gen = [len(gen[-1]) for gen in outputs if gen is not None]

    max_timesteps = max(num_timesteps_per_gen, default=0)
    max_lanes = max_num_lanes_in_batch(outputs)
    max_A_depth = get_max_A_depth(outputs)

    len_x = len(pad_scalars[1])
    len_y = len(pad_scalars[2])

    blank_A = np.zeros((0, 0, 0))
    blank_x = np.zeros((0, 0, len_x))
    blank_y = np.zeros((0, 0, len_y))
    blank_value = [blank_A, blank_x, blank_y]

    # pad everything
    A_matrices = []
    X_per_gen = []
    Y_per_gen = []
    for g in range(len(outputs)):
        if outputs[g] is None:
            A, X, Y = blank_value
        else:
            A = outputs[g][0]
            X = outputs[g][1].astype(np.float32)
            Y = outputs[g][2].astype(np.float32)

        if np.ndim(A) == 2: # no depth or time dimensions
            A = np.reshape(A, [1, 1, *A.shape])
        elif np.ndim(A) == 3: # no time dimension
            A = np.expand_dims(A, 0)

        pad_A_depth = max_A_depth - A.shape[-3]
        pad_timesteps = max_timesteps - X.shape[0]
        pad_lanes = max_lanes - X.shape[1]

        A_pad_dims = ((0, 0), (0, pad_A_depth), (0, pad_lanes), (0, pad_lanes))
        A = np.pad(A, A_pad_dims, 'constant', constant_values=pad_scalars[0])
        A_shape = A.shape
        if A_shape[0] == 1:
            A = np.broadcast_to(A, [max_timesteps, *A_shape[1:]])
        else:
            A = np.pad(A, ((0, pad_timesteps), (0, 0), (0, 0), (0, 0)),
                       'constant', constant_values=pad_scalars[0])

        data_pad_dims = ((0, pad_timesteps), (0, pad_lanes), (0, 0))
        X = _split_pad_concat(X, data_pad_dims, pad_scalars[1])
        Y = _split_pad_concat(Y, data_pad_dims, pad_scalars[2])

        A_matrices.append(A)
        X_per_gen.append(X)
        Y_per_gen.append(Y)

    if try_broadcast_A and can_broadcast_A_matrices_over_time(outputs):
        A = np.expand_dims(
            np.stack([A_t[0] for A_t in A_matrices]), 1)
    else:
        A = np.stack(A_matrices)

    X = np.stack(X_per_gen)
    Y = np.stack(Y_per_gen)

    return A, X, Y


def _split_pad_concat(array, pad_dims, scalars):
    assert array.shape[-1] == len(scalars)
    split = np.split(array, array.shape[-1], -1)
    padded = [np.pad(part, pad_dims, 'constant', constant_values=scalar)
              for part, scalar in zip(split, scalars)]
    concatted = np.concatenate(padded, -1)
    return concatted


def _slice_off_empty_timesteps(nested_list, empty_timesteps):
    return [[ts for ts, empty in zip(list_for_gen, empty_timesteps)
            if not empty] for list_for_gen in nested_list]


def windowed_batch_of_generators(
    filenames,
    sim_indeces,
    window_size,
    A_name_list=['A_downstream',
                 'A_upstream',
                 'A_neighbors'],
    x_feature_subset=['e1_0/occupancy',
                      'e1_0/speed',
                      'e1_1/occupancy',
                      'e1_1/speed',
                      'liu_estimated',
                      'green'],
    y_feature_subset=['e2_0/nVehSeen',
                      'e2_0/maxJamLengthInMeters']):

    assert len(filenames) == len(sim_indeces)
    generators = [generator_from_file(f, si,
                                      chunk_size=window_size,
                                      repeat_A_over_time=True,
                                      A_name_list=A_name_list,
                                      x_feature_subset=x_feature_subset,
                                      y_feature_subset=y_feature_subset)
                  for f, si in zip(filenames, sim_indeces)]

    reader = zip_longest(*generators)
    return reader


def windowed_unpadded_batch_of_generators(filenames,
    sim_indeces,
    window_size,
    batch_size_to_pad_to=None,
    A_name_list=['A_downstream',
                 'A_upstream',
                 'A_neighbors'],
    x_feature_subset=['e1_0/occupancy',
                      'e1_0/speed',
                      'e1_1/occupancy',
                      'e1_1/speed',
                      'liu_estimated',
                      'green'],
    y_feature_subset=['e2_0/nVehSeen',
                      'e2_0/maxJamLengthInMeters'],
    generator_buffer_size=10):
    assert len(filenames) == len(sim_indeces)
    if batch_size_to_pad_to is not None:
        num_dummy_generators = batch_size_to_pad_to - len(filenames)

    fill_A = np.zeros((0, 0, 0, 0)) # time x depth x lanes x lanes
    fill_X = np.zeros((0, 0, len(x_feature_subset))) # time x lane x channel
    fill_Y = np.zeros((0, 0, len(y_feature_subset))) # time x lane x channel

    default_val = (fill_A, fill_X, fill_Y)

    generators = [generator_from_file(f, si,
                                      chunk_size=window_size,
                                      repeat_A_over_time=True,
                                      A_name_list=A_name_list,
                                      x_feature_subset=x_feature_subset,
                                      y_feature_subset=y_feature_subset,
                                      buffer_size=generator_buffer_size)
                  for f, si in zip(filenames, sim_indeces)]

    generators.extend([iter(()) for _ in range(num_dummy_generators)])

    batch = zip_longest(*generators, fillvalue=default_val)
    for out in batch:
        yield from out


def read_from_file(
    filename,
    simulation_number,
    repeat_A_over_time=True,
    A_name_list=['A_downstream',
                 'A_upstream',
                 'A_neighbors'],
    x_feature_subset=['e1_0/occupancy',
                      'e1_0/speed',
                      'e1_1/occupancy',
                      'e1_1/speed',
                      'liu_estimated',
                      'green'],
    y_feature_subset=['e2_0/nVehSeen',
                      'e2_0/maxJamLengthInMeters'],
    return_X_Y_as_dfs=False):

    # Input handling if we came from TF
    if isinstance(filename, six.binary_type):
        filename = filename.decode()
    if isinstance(simulation_number, six.string_types + (six.binary_type,)):
        simulation_number = int(simulation_number)

    A_name_list, x_feature_subset, y_feature_subset = map(
        string_list_decode,
        [A_name_list, x_feature_subset, y_feature_subset])
    assert all([A_name in All_A_name_list for A_name in A_name_list])

    t0 = time.time()
    with pd.HDFStore(filename, 'r') as store:
        A_list = []
        for A_name in A_name_list:
            try:
                A = store[A_name]
                A_list.append(A)
            except KeyError:
                A_list.append(np.zeros_like(A_list[0]))

        lane_list = A_list[0].index
        assert all((all(lane_list == A.index) for A in A_list))
        A = np.stack([A.values for A in A_list])

        X_dfs = [store['{}/X/_{:04}'.format(lane, simulation_number)]
                for lane in lane_list]
        X_dfs = [df.loc[:, x_feature_subset] for df in X_dfs]
        X_df = pd.concat(X_dfs, keys=lane_list, names=['lane', 'time'])
        X_df = X_df.swaplevel().sort_index()
        X_df = X_df.fillna(pad_value_for_feature).astype(np.float32)

        Y_dfs = [store['{}/Y/_{:04}'.format(lane, simulation_number)]
                for lane in lane_list]
        Y_dfs = [df.loc[:, y_feature_subset] for df in Y_dfs]
        Y_df = pd.concat(Y_dfs, keys=lane_list, names=['lane', 'time'])
        Y_df = Y_df.swaplevel().sort_index()
        Y_df = Y_df.fillna(pad_value_for_feature).astype(np.float32)

    t = time.time() - t0
    _logger.debug('Loading data from disk took %s s', t)

    num_lanes = len(lane_list)
    len_x = len(x_feature_subset)
    len_y = len(y_feature_subset)

    if not return_X_Y_as_dfs:
        X = X_df.values.reshape(-1, num_lanes, len_x)
        Y = Y_df.values.reshape(-1, num_lanes, len_y)
        num_timesteps = X.shape[0]
    else:
        X = X_df
        Y = Y_df
        num_timesteps = len(X.index.get_level_values(0).unique())

    A = np.expand_dims(A, 0)
    if repeat_A_over_time:
        A = np.repeat(A, num_timesteps, axis=0)

    return A, X, Y


def generator_prefetch_all_from_file(
    filename,
    simulation_number,
    chunk_size=None,
    repeat_A_over_time=True,
    A_name_list=['A_downstream',
                 'A_upstream',
                 'A_neighbors'],
    x_feature_subset=['e1_0/occupancy',
                      'e1_0/speed',
                      'e1_1/occupancy',
                      'e1_1/speed',
                      'liu_estimated',
                      'green'],
    y_feature_subset=['e2_0/nVehSeen',
                      'e2_0/maxJamLengthInMeters']):

    A, X_df, Y_df = read_from_file(filename,
                                   simulation_number,
                                   repeat_A_over_time,
                                   A_name_list,
                                   x_feature_subset,
                                   y_feature_subset)
    try:
        slice_begin = 0
        slice_end = chunk_size
        num_lanes = A.shape[-1]
        len_x = len(x_feature_subset)
        len_y = len(y_feature_subset)

        while True:
            time_slicer = slice(slice_begin, slice_end)

            X_df_slice = X_df.loc[time_slicer]
            Y_df_slice = Y_df.loc[time_slicer]

            t = X_df_slice.index.get_level_values(0).unique().values
            if t.size == 0:
                return

            X_slice = X_df_slice.values.reshape(-1, num_lanes, len_x)
            Y_slice = Y_df_slice.values.reshape(-1, num_lanes, len_y)

            if repeat_A_over_time:
                A_slice = A[time_slicer]
            else:
                A_slice = A

            yield A_slice, X_slice, Y_slice, t

            slice_begin += chunk_size
            slice_end += chunk_size
    except TypeError:
        return


class _Buffer(object):
    def __init__(self, source_df, start_t, window_size, num_windows):
        self.dfs = []
        max_time = start_t + window_size * num_windows - 1
        unwindowed_df = source_df.loc[start_t:max_time].copy()
        window_end_t = start_t + window_size - 1
        for _ in range(num_windows):
            slicer = slice(start_t, window_end_t)
            df = unwindowed_df.loc[slicer]
            self.dfs.append(df)
            start_t += window_size
            window_end_t += window_size

    def __iter__(self):
        yield from self.dfs


def generator_from_file(
    filename,
    simulation_number,
    chunk_size=None,
    repeat_A_over_time=True,
    A_name_list=['A_downstream',
                 'A_upstream',
                 'A_neighbors'],
    x_feature_subset=['e1_0/occupancy',
                      'e1_0/speed',
                      'e1_1/occupancy',
                      'e1_1/speed',
                      'liu_estimated',
                      'green'],
    y_feature_subset=['e2_0/nVehSeen',
                      'e2_0/maxJamLengthInMeters'],
    buffer_size=10):

    # Input handling if we came from TF
    if isinstance(filename, six.binary_type):
        filename = filename.decode()
    if isinstance(simulation_number, six.string_types + (six.binary_type,)):
        simulation_number = int(simulation_number)

    A_name_list, x_feature_subset, y_feature_subset = map(
        string_list_decode,
        [A_name_list, x_feature_subset, y_feature_subset])
    assert all([A_name in All_A_name_list for A_name in A_name_list])

    filedir = os.path.dirname(filename)

    with ExitStack() as stack:
        temp_dir = stack.enter_context(tempfile.TemporaryDirectory(dir=filedir))
        temp_store_name = os.path.join(temp_dir, 'temp.h5')
        temp_store = stack.enter_context(pd.HDFStore(temp_store_name,
                                                     complevel=5,
                                                     complib='blosc:lz4'))
        with pd.HDFStore(filename, 'r') as store:
            A_list = []
            for A_name in A_name_list:
                try:
                    A = store[A_name]
                    A_list.append(A)
                except KeyError:
                    A_list.append(np.zeros_like(A_list[0]))

            lane_list = A_list[0].index
            assert all((all(lane_list == A.index) for A in A_list))
            A = np.stack([A.values for A in A_list])

            X_df_strings = ['{}/X/_{:04}'.format(lane, simulation_number)
                            for lane in lane_list]
            _read_concat_write(store, X_df_strings, x_feature_subset,
                              temp_store, 'X', lane_list)

            Y_df_strings = ['{}/Y/_{:04}'.format(lane, simulation_number)
                            for lane in lane_list]
            _read_concat_write(store, Y_df_strings, y_feature_subset,
                              temp_store, 'Y', lane_list)

        slice_begin = 0
        slice_end = chunk_size - 1
        num_lanes = A.shape[-1]
        len_x = len(x_feature_subset)
        len_y = len(y_feature_subset)

        try:
            while True:
                X_buffer = _Buffer(temp_store['X'], slice_begin, chunk_size,
                                   buffer_size)
                Y_buffer = _Buffer(temp_store['Y'], slice_begin, chunk_size,
                                   buffer_size)
                assert len(X_buffer.dfs) == len(Y_buffer.dfs)

                for X_df, Y_df in zip(X_buffer, Y_buffer):

                    X_t = X_df.index.get_level_values(0).unique().values
                    Y_t = Y_df.index.get_level_values(0).unique().values
                    assert np.array_equal(X_t, Y_t)

                    num_timesteps = X_t.size
                    if num_timesteps == 0:
                        return

                    assert slice_begin == X_t[0]
                    try:
                        assert slice_end == X_t[-1]
                    except AssertionError:
                        assert slice_begin + num_timesteps - 1 == X_t[-1]

                    X_slice = X_df.values.reshape(-1, num_lanes, len_x)
                    Y_slice = Y_df.values.reshape(-1, num_lanes, len_y)

                    if repeat_A_over_time:
                        A_slice = np.broadcast_to(A, [num_timesteps, *A.shape])
                    else:
                        A_slice = A

                    yield A_slice, X_slice, Y_slice

                    slice_begin += chunk_size
                    slice_end += chunk_size
        except TypeError:
            return


def _read_concat_write(source_store, read_strings, col_subset, write_store,
                       write_key, lane_list):
    dfs = [source_store[k].loc[:, col_subset] for k in read_strings]
    df = _merge_fill(dfs, lane_list)
    write_store.append(write_key, df, complevel=5, complib='blosc:lz4')


def _merge_fill(dfs, lane_list):
    df = pd.concat(dfs, keys=lane_list, names=['lane', 'time'])
    df = df.swaplevel().sort_index()
    df = df.fillna(pad_value_for_feature).astype(np.float32)
    return df


def _colwise_iterator(listoflist_series, return_time=True):
    col_iterators = [[col.items() for col in lane_cols] for lane_cols in listoflist_series]
    if return_time:
        def step_iters():
            step = [[next(it) for it in lane_its] for lane_its in col_iterators]
            time = step[0][0][0]
            step = tuple([time, tuple([np.stack([row[1] for row in per_lane]) for per_lane in step])])
            return step
    else:
        def step_iters():
            return tuple(np.stack([next(it)[1] for it in lane_its]) for lane_its in col_iterators)
    try:
        while True:
            col_out = step_iters()
            yield col_out
    except StopIteration:
        return


def _df_iterator(dfs, num_feats):
    iterators = [df.iterrows() for df in dfs]
    try:
        while True:
            rows = [six.next(it)[1].values for it in iterators]
            yield np.split(np.transpose(rows), num_feats)
    except StopIteration:
        return


def _timestep_iterator(dfs):
    iterators = [df.iterrows() for df in dfs]
    try:
        while True:
            yield np.stack([six.next(it)[1].values for it in iterators], 0)
    except StopIteration:
        return

def get_sim_numbers_in_file(filename):
    with pd.HDFStore(filename, 'r') as store:
        return get_sim_numbers_in_preprocess_store(store)

def get_pad_value_for_feature(feature):
    return pad_value_for_feature[feature]
