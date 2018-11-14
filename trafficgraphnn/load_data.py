from collections import defaultdict
from itertools import takewhile

import numpy as np
import pandas as pd
import six

from trafficgraphnn.utils import (flatten, get_preprocessed_filenames,
                                  get_sim_numbers_in_preprocess_store, grouper,
                                  paditerable)

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

    def iterate(self):
        for output in self.readers:
            yield pad_and_stack_batch(output, self.pad_scalars)


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
    A_matrix_size_per_gen = [np.shape(gen[0][0]) for gen in output_list]

    # sanity check: make sure adjacency matrices are square
    assert all(A_shape[-1] == A_shape[-2] for A_shape in A_matrix_size_per_gen)
    max_num_lanes = max(A_shape[-2] for A_shape in A_matrix_size_per_gen)
    return max_num_lanes


def get_max_A_depth(output_list):
    """Returns the max depth (number of edge types) in the batch"""
    A_matrix_size_per_gen = [np.shape(gen[0][0]) for gen in output_list]

    return max(A_shape[0] for A_shape in A_matrix_size_per_gen)


def can_broadcast_A_matrices_over_time(output_list):
    # Do the A matrices vary across timesteps or, if they are all equal, can we
    # have a time dimension of 1?
    A_matrices_by_gen = [[variables[0] for variables in output]
                         for output in output_list]
    all_same_As = all(np.array_equal(batch_A[0], A)
                      for batch_A in A_matrices_by_gen for A in batch_A)
    return all_same_As


def pad_and_stack_batch(outputs, pad_scalars):
    empty_timesteps_per_gen = [[timestep is None for timestep in gen] for
                               gen in outputs]

    # these timesteps can be sliced off
    completely_empty_timesteps = np.asarray(empty_timesteps_per_gen).all(0)

    outputs = _slice_off_empty_timesteps(outputs, completely_empty_timesteps)

    max_lanes = max_num_lanes_in_batch(outputs)
    max_A_depth = get_max_A_depth(outputs)

    blank_A = np.full([max_A_depth, max_lanes, max_lanes], pad_scalars[0])
    blank_x = [np.full([max_lanes], scalar) for scalar in pad_scalars[1]]
    blank_y = [np.full([max_lanes], scalar) for scalar in pad_scalars[2]]
    blank_value = [blank_A, blank_x, blank_y]

    # pad everything
    for g in range(len(outputs)):
        for t in range(len(outputs[g])):
            if outputs[g][t] is None: # pad the entire timestep with the pad value
                outputs[g][t] = blank_value
            else: # pad the data to the max number of lanes
                A_shape = np.shape(outputs[g][t][0])
                outputs[g][t][0] = np.pad(outputs[g][t][0],
                                          [[0, A_shape[0] - max_A_depth],
                                           [0, A_shape[1] - max_lanes],
                                           [0, A_shape[2] - max_lanes]],
                                          'constant',
                                          constant_values=pad_scalars[0])
                outputs[g][t][1] = [
                    np.pad(x_feat, max_lanes - len(x_feat), 'constant',
                           constant_values=scalar)
                    for x_feat, scalar in zip(outputs[g][t][1], pad_scalars[1])
                ]
                outputs[g][t][2] = [
                    np.pad(y_feat, max_lanes - len(y_feat), 'constant',
                           constant_values=scalar)
                    for y_feat, scalar in zip(outputs[g][t][2], pad_scalars[2])
                ]

    A_matrices = [[variables[0] for variables in output] for output in outputs]
    X_per_gen = [[variables[1] for variables in output] for output in outputs]
    Y_per_gen = [[variables[2] for variables in output] for output in outputs]

    if can_broadcast_A_matrices_over_time(outputs):
        A = np.expand_dims(
            np.stack([A_t[0] for A_t in A_matrices]), 1)
    else:
        A = np.stack([np.stack([A for A in A_t]) for A_t in A_matrices])

    X = np.stack([[np.stack(X_tg, -1) for X_tg in X_g] for X_g in X_per_gen])
    Y = np.stack([[np.stack(Y_tg, -1) for Y_tg in Y_g] for Y_g in Y_per_gen])

    return A, X, Y

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
                                      A_name_list=A_name_list,
                                      x_feature_subset=x_feature_subset,
                                      y_feature_subset=y_feature_subset)
                  for f, si in zip(filenames, sim_indeces)]

    reader = window_and_batch_generators(generators, window_size)
    return reader


def window_and_batch_generators(generators, window_size):
    groupers = [grouper(paditerable(gen), window_size) for gen in generators]
    batch = zip(*groupers)
    reader = takewhile(lambda x: any(flatten(x)), batch)
    return reader


def generator_from_file(
    filename,
    simulation_number,
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
            while True:
                col_select_X = [[store[
                    '{}/X/_{:04}'.format(lane, simulation_number)].loc[:, feat]
                                 for lane in lane_list]
                                for feat in x_feature_subset]
                col_select_Y = [[store[
                    '{}/Y/_{:04}'.format(lane, simulation_number)].loc[:, feat]
                                 for lane in lane_list]
                                for feat in y_feature_subset]

                X_iter = _colwise_iterator(col_select_X, True)
                Y_iter = _colwise_iterator(col_select_Y, False)

                for tx, y, in zip(X_iter, Y_iter):
                    t, x = tx
                    yield [A, x, y, t]
        except KeyError:
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
