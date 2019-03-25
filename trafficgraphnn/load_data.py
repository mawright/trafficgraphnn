import logging
from collections import defaultdict
from itertools import zip_longest

import numpy as np
import pandas as pd
import six

from trafficgraphnn.preprocessing.io import get_preprocessed_filenames, queueing_intervals_from_lane_light_df
from trafficgraphnn.utils import flatten, iterfy, string_list_decode

_logger = logging.getLogger(__name__)

pad_value_for_feature = defaultdict(lambda: 0,
                                    occupancy=0.,
                                    speed=-1.,
                                    liu_estimated_m=-1.,
                                    liu_estimated_veh=-1.,
                                    green=0.,
                                    nVehSeen=0.,
                                    maxJamLengthInMeters=-1.,
                                    maxJamLengthInVehicles=-1.,
                                   )

pad_value_for_feature.update(
    [('e1_0/occupancy', 0.),
    ('e1_0/speed', -1.),
    ('e1_1/occupancy', 0.),
    ('e1_1/speed', -1.),
    ('e2_0/nVehSeen', 0.),
    ('e2_0/maxJamLengthInMeters', -1.),
    ('e2_0/maxJamLengthInVehicles', -1.),
])

x_feature_subset_default = ['e1_0/occupancy',
                            'e1_0/speed',
                            'e1_1/occupancy',
                            'e1_1/speed',
                            'liu_estimated_veh',
                            'green']

y_feature_subset_default = ['e2_0/nVehSeen',
                            'e2_0/maxJamLengthInVehicles']

All_A_name_list = ['A_downstream', 'A_upstream', 'A_neighbors',
                   'A_turn_movements', 'A_through_movements']

per_cycle_features_default = ['maxJamLengthInMeters',
                              'maxJamLengthInVehicles',
                              'e2_0/maxJamLengthInMeters',
                              'e2_0/maxJamLengthInVehicles',
                              'liu_estimated_veh',
                              'liu_estimated_m']


class Batch(object):
    """Convenience class for holding a batch of sim readers.
    """

    def __init__(self,
                 filenames,
                 window_size,
                 A_name_list=['A_downstream',
                              'A_upstream',
                              'A_neighbors'],
                 x_feature_subset=x_feature_subset_default,
                 y_feature_subset=y_feature_subset_default):
        self.filenames = filenames
        self.A_name_list = A_name_list
        self.x_feature_subset = x_feature_subset
        self.y_feature_subset = y_feature_subset

        self.pad_scalars = get_pad_scalars(self.x_feature_subset, self.y_feature_subset)

        if isinstance(filenames, np.ndarray):
            filenames = filenames.tolist()

        self.readers = windowed_batch_of_generators(
            filenames, window_size, A_name_list, x_feature_subset,
            y_feature_subset)

    def iterate(self, try_broadcast_A=False):
        for output in self.readers:
            yield pad_and_stack_batch(output[:3],
                                      self.pad_scalars,
                                      try_broadcast_A=try_broadcast_A)


def batches_from_directories(directories,
                             batch_size,
                             window_size,
                             shuffle=True,
                             A_name_list=['A_downstream',
                                          'A_upstream',
                                          'A_neighbors'],
                             x_feature_subset=x_feature_subset_default,
                             y_feature_subset=y_feature_subset_default):

    filenames = get_file_names_in_dirs(directories)

    if shuffle:
        np.random.shuffle(filenames)

    batched_filenames = [filenames[i:i + batch_size]
                         for i in range(0, len(filenames), batch_size)]

    for f_batch in batched_filenames:
        batch = Batch(f_batch, window_size, A_name_list=A_name_list,
                      x_feature_subset=x_feature_subset,
                      y_feature_subset=y_feature_subset)
        yield batch


def get_file_names_in_dirs(directories):
    directories = iterfy(directories)
    filenames = list(flatten([get_preprocessed_filenames(directory)
                              for directory in directories]))
    return filenames


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
    window_size,
    A_name_list=['A_downstream',
                 'A_upstream',
                 'A_neighbors'],
    x_feature_subset=x_feature_subset_default,
    y_feature_subset=y_feature_subset_default):

    generators = [generator_prefetch_all_from_file(
        f,
        chunk_size=window_size,
        repeat_A_over_time=True,
        A_name_list=A_name_list,
        x_feature_subset=x_feature_subset,
        y_feature_subset=y_feature_subset) for f in filenames]

    reader = zip_longest(*generators)
    return reader


def windowed_unpadded_batch_of_generators(
    filenames,
    window_size,
    batch_size_to_pad_to=None,
    A_name_list=['A_downstream',
                 'A_upstream',
                 'A_neighbors'],
    x_feature_subset=x_feature_subset_default,
    y_feature_subset=y_feature_subset_default,
    per_cycle_features=per_cycle_features_default,
    prefetch_all=True):
    if batch_size_to_pad_to is not None:
        num_dummy_generators = batch_size_to_pad_to - len(filenames)

    fill_A = np.zeros((0, 0, 0, 0)) # time x depth x lanes x lanes
    fill_X = np.zeros((0, 0, len(x_feature_subset))) # time x lane x channel
    fill_Y = np.zeros((0, 0, len(y_feature_subset))) # time x lane x channel
    fill_t = np.zeros((0)) # time
    fill_lane = [] # lane

    default_val = (fill_A, fill_X, fill_Y, fill_t, fill_lane)

    if not prefetch_all:
        raise NotImplementedError('Non-prefetching not supported currently.')

    generators = [generator_prefetch_all_from_file(
        f,
        chunk_size=window_size,
        repeat_A_over_time=True,
        A_name_list=A_name_list,
        x_feature_subset=x_feature_subset,
        y_feature_subset=y_feature_subset,
        per_cycle_features=per_cycle_features)
                  for f in filenames]

    generators.extend([iter(()) for _ in range(num_dummy_generators)])

    batch = zip_longest(*generators, fillvalue=default_val)
    for out in batch:
        yield from out


def read_from_file(
    filename,
    repeat_A_over_time=True,
    A_name_list=['A_downstream',
                 'A_upstream',
                 'A_neighbors'],
    x_feature_subset=x_feature_subset_default,
    y_feature_subset=y_feature_subset_default,
    per_cycle_features=per_cycle_features_default,
    when_per_cycle='end', # begin = on green, # end = on red
    return_X_Y_as_dfs=False):

    # Input handling if we came from TF
    if isinstance(filename, six.binary_type):
        filename = filename.decode()

    if len(per_cycle_features) > 0 and when_per_cycle not in ['end', 'begin']:
        raise ValueError(
            'Argument `when_per_cycle` must be one of `end`, `begin`')

    A_name_list, x_feature_subset, y_feature_subset = map(
        string_list_decode,
        [A_name_list, x_feature_subset, y_feature_subset])
    assert all([A_name in All_A_name_list for A_name in A_name_list])

    with pd.HDFStore(filename, 'r') as store:
        A_df = store['A']
        lane_list = A_df.index
        num_lanes = len(lane_list)
        A = np.stack([A_df[A_name] if A_name in All_A_name_list
                      else np.zeros((num_lanes, num_lanes), dtype='bool')
                      for A_name in A_name_list])

        X_df = store['X'].loc[:,x_feature_subset]
        X_df = X_df.fillna(pad_value_for_feature).astype(np.float32)

        Y_df = store['Y'].loc[:,y_feature_subset]
        Y_df = Y_df.fillna(pad_value_for_feature).astype(np.float32)

        # masking out Y features only predicted per cycle
        if 'green' in X_df and len(per_cycle_features) > 0:
            if when_per_cycle == 'end':
                index = 1
            elif when_per_cycle == 'begin':
                index = 0
            cycle_intervals = [item for _, series
                               in X_df['green'].groupby('lane') for item in
                               queueing_intervals_from_lane_light_df(series)]

            feats_to_mask = [feat for feat in per_cycle_features
                             if feat in Y_df]

            for feat in feats_to_mask:
                series = Y_df[feat]
                keys = [cycle[index] for cycle in cycle_intervals]
                values = [series.loc[cycle[0]:cycle[1]].max()
                          for cycle in cycle_intervals]
                # return keys, values
                Y_df[feat] = get_pad_value_for_feature(feat)
                Y_df.loc[keys, feat] = values

    len_x = len(x_feature_subset)
    len_y = len(y_feature_subset)

    if not return_X_Y_as_dfs:
        X = X_df.values.reshape(num_lanes, -1, len_x).transpose([1, 0, 2])
        Y = Y_df.values.reshape(num_lanes, -1, len_y).transpose([1, 0, 2])
        num_timesteps = X.shape[0]
    else:
        X = X_df
        Y = Y_df
        num_timesteps = len(X.index.get_level_values('begin').unique())

    A = np.expand_dims(A, 0)
    if repeat_A_over_time:
        A = np.repeat(A, num_timesteps, axis=0)

    return A, X, Y


def generator_prefetch_all_from_file(
    filename,
    chunk_size=None,
    repeat_A_over_time=True,
    A_name_list=['A_downstream',
                 'A_upstream',
                 'A_neighbors'],
    x_feature_subset=x_feature_subset_default,
    y_feature_subset=y_feature_subset_default,
    per_cycle_features=per_cycle_features_default):

    A, X_df, Y_df = read_from_file(filename,
                                   repeat_A_over_time,
                                   A_name_list,
                                   x_feature_subset,
                                   y_feature_subset,
                                   per_cycle_features,
                                   return_X_Y_as_dfs=True)
    try:
        t_begin = 0
        t_end = chunk_size - 1
        num_lanes = A.shape[-1]

        if X_df.index.names[0] == 'lane':
            def get_slice(df, time_start, time_end):
                subdf = df.loc[pd.IndexSlice[:, time_start:time_end], :]
                lanes = subdf.index.get_level_values(0).unique()
                timesteps = subdf.index.get_level_values(1).unique()
                num_timesteps = len(timesteps)
                num_feats = subdf.shape[-1]
                return (subdf.values
                             .reshape((num_lanes, num_timesteps, num_feats))
                             .transpose((1, 0, 2)),
                        timesteps,
                        lanes)
        elif X_df.index.names[1] == 'lane':
            def get_slice(df, time_start, time_end):
                subdf = df.loc[pd.IndexSlice[:, time_start:time_end], :]
                lanes = subdf.index.get_level_values(1).unique()
                timesteps = subdf.index.get_level_values(0).unique()
                num_timesteps = len(timesteps)
                num_feats = subdf.shape[-1]
                return (subdf.values.reshape((num_timesteps, num_lanes,
                                             num_feats)),
                        timesteps,
                        lanes)

        while True:
            X_slice, X_timesteps, X_lanes = get_slice(X_df, t_begin, t_end)
            Y_slice, Y_timesteps, Y_lanes = get_slice(Y_df, t_begin, t_end)

            assert (X_timesteps == Y_timesteps).all()
            assert (X_lanes == Y_lanes).all()
            timesteps = X_timesteps
            lanes = X_lanes

            if X_slice.size == 0:
                assert Y_slice.size == 0
                return

            if repeat_A_over_time:
                A_slice = A[t_begin:t_begin+chunk_size]
            else:
                A_slice = A

            yield A_slice, X_slice, Y_slice, timesteps, lanes

            t_begin += chunk_size
            t_end += chunk_size
    except TypeError:
        return


def get_pad_value_for_feature(feature):
    return pad_value_for_feature[feature]
