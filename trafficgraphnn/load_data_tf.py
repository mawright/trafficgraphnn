import os

import numpy as np
import pandas as pd
import tensorflow as tf

from trafficgraphnn.utils import get_preprocessed_filenames

All_A_name_list = ['A_downstream', 'A_upstream', 'A_neighbors',
                   'A_turn_movements', 'A_through_movements']


def generator_from_file(filename,
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
        A = np.expand_dims(np.stack([A.values for A in A_list]), 1)
        sim_index = 1
        try:
            while True:
                X = tuple(
                    [np.stack([store['{}/X/_{:04}'.format(lane, sim_index)].loc[
                        :, feat].values
                               for lane in lane_list], axis=1)
                     for feat in x_feature_subset]
                    )
                Y = tuple(
                    [np.stack([store['{}/Y/_{:04}'.format(lane, sim_index)].loc[
                        :, feat].values
                               for lane in lane_list], axis=1)
                     for feat in y_feature_subset]
                    )
                yield A, X, Y
                sim_index += 1
        except KeyError:
            return


def get_pad_value_for_feature(feature):
    if 'occupancy' in feature:
        return 0.
    elif 'speed' in feature:
        return -1.
    elif feature == 'liu_estimated':
        return -1.
    elif feature == 'green':
        return 0.
    elif 'nVehSeen' in feature:
        return 0.
    elif 'maxJamLengthInMeters' in feature:
        return np.nan
    else:
        return 0.

def load_data(
    directory, parallel_interleave_cycle_length=None, batch_size=32,
    x_features=['e1_0/occupancy',
                'e1_0/speed',
                'e1_1/occupancy',
                'e1_1/speed',
                'liu_estimated',
                'green'],
    y_features=['e2_0/nVehSeen',
                'e2_0/maxJamLengthInMeters'],
    A_name_list=['A_downstream',
                 'A_upstream',
                 'A_neighbors']):
    if parallel_interleave_cycle_length is None:
        parallel_interleave_cycle_length = os.cpu_count()
    data_files = get_preprocessed_filenames(directory)

    dataset = tf.data.Dataset.from_tensor_slices(data_files) # dataset of file names
    dataset = dataset.shuffle(buffer_size=1000) # shuffled dataset of file names

    gen = lambda filename: tf.data.Dataset.from_generator(
        lambda x: generator_from_file(x, A_name_list, x_features, y_features),
        (tf.bool,
         (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
         (tf.float32, tf.float32)),
        args=(filename,))

    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            gen, cycle_length=parallel_interleave_cycle_length))

    a_padding_dims = tuple([-1, -1, -1, -1])
    x_padding_dims = tuple([tuple([-1, -1])] * len(x_features))
    y_padding_dims = tuple([tuple([-1, -1])] * len(y_features))
    x_padding_values = tuple((get_pad_value_for_feature(feat)
                              for feat in x_features))
    y_padding_values = tuple((get_pad_value_for_feature(feat)
                              for feat in y_features))

    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=(a_padding_dims, x_padding_dims, y_padding_dims),
        padding_values=(False, x_padding_values, y_padding_values))

    def stack_fn(A, X, Y):
        X = tf.stack(X, axis=-1)
        Y = tf.stack(Y, axis=-1)
        return A, X, Y

    dataset = dataset.map(stack_fn)
    dataset = dataset.prefetch(1)

    return dataset
