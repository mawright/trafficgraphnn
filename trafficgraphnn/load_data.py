from collections import defaultdict
from itertools import takewhile, zip_longest

import numpy as np
import pandas as pd
import six

from trafficgraphnn.utils import (flatten, get_preprocessed_filenames,
                                  get_sim_numbers_in_preprocess_store, grouper,
                                  iterfy, paditerable)

pad_value_for_feature = defaultdict(lambda: 0,
                                    occupancy=0.,
                                    speed=-1.,
                                    liu_estimated=-1.,
                                    green=0.,
                                    nVehSeen=0.,
                                    maxJamLengthInMeters=np.nan,
                                   )

pad_value_for_feature.update(
    [('e1_0/occupancy', 0),
    ('e1_0/speed', -1.),
    ('e1_1/occupancy', 0.),
    ('e1_1/speed', -1.),
    ('e2_0/nVehSeen', 0.),
    ('e2_0/maxJamLengthInMeters', np.nan)])

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
    directories = iterfy(directories)
    filenames = flatten([get_preprocessed_filenames(directory)
                         for directory in directories])
    file_and_sims = []

    for filename in filenames:
        for sim_number in get_sim_numbers_in_file(filename):
            file_and_sims.append((filename, sim_number))
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
                                      A_name_list=A_name_list,
                                      x_feature_subset=x_feature_subset,
                                      y_feature_subset=y_feature_subset)
                  for f, si in zip(filenames, sim_indeces)]

    # reader = batch_generators(generators)
    reader = zip_longest(*generators)
    return reader


def batch_generators(generators):
    batch = zip(*generators)
    reader = takewhile(lambda x: any(x), batch)
    return reader


def generator_from_file(
    filename,
    simulation_number,
    chunk_size=None,
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

    if isinstance(simulation_number, six.string_types + (bytes,)):
        simulation_number = int(simulation_number)
    assert all([A_name in All_A_name_list for A_name in A_name_list])
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
        try:
            slice_begin = 0
            slice_end = chunk_size
            while True:
                time_slicer = slice(slice_begin, slice_end)
                t = store['{}/X/_{:04}'.format(lane_list[0], simulation_number)
                          ].iloc[time_slicer, 0].index

                if t.empty:
                    return

                X_dfs = [store[
                    '{}/X/_{:04}'.format(lane, simulation_number)].loc[
                        t, x_feature_subset]
                    for lane in lane_list]
                Y_dfs = [store[
                    '{}/Y/_{:04}'.format(lane, simulation_number)].loc[
                        t, y_feature_subset]
                    for lane in lane_list]

                X = np.stack([df.values for df in X_dfs], 1)
                Y = np.stack([df.values for df in Y_dfs], 1)

                yield A, X, Y, t

                slice_begin += chunk_size
                slice_end += chunk_size
        except TypeError:
            return


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
