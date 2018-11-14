import os

import tensorflow as tf

from trafficgraphnn.load_data import (get_sim_numbers_in_file,
                                      generator_from_file,
                                      get_pad_value_for_feature)
from trafficgraphnn.utils import (get_preprocessed_filenames,
                                  get_sim_numbers_in_preprocess_store, iterfy)


def load_data(
    directories, batch_size=32, window_size=400,
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
    if batch_size is None:
        batch_size = os.cpu_count()

    directories = iterfy(directories)
    data_files = [f for d in directories for f in get_preprocessed_filenames(d)]

    sim_number_file_inventory = {
        filename: get_sim_numbers_in_file(filename)
        for filename in data_files}
    file_and_sim_tuples = [(k, str(i)) for k, v in sim_number_file_inventory.items()
                                       for i in v]

    dataset = tf.data.Dataset.from_tensor_slices(file_and_sim_tuples)
    dataset = dataset.shuffle(buffer_size=1000) # shuffled dataset of file names

    gen = lambda file_and_sim: tf.data.Dataset.from_generator(
        lambda x: generator_from_file(x, A_name_list, x_features, y_features),
        (tf.bool,
         (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
         (tf.float32, tf.float32)),
        args=(file_and_sim[0], file_and_sim[1])).batch(window_size)

    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            gen, cycle_length=batch_size))

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

    return TFDataset(dataset)


class TFDataset(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                                        dataset.output_shapes)

        A, X, Y = self.iterator.get_next()
        self.A = A
        self.X = X
        self.Y = Y
