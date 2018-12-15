import math
from collections import namedtuple

import tensorflow as tf

from trafficgraphnn.load_data import get_file_and_sim_indeces_in_dirs, read_from_file, pad_value_for_feature

TFBatch = namedtuple('TFBatch', ['dataset', 'initializer'])

def make_datasets(directories,
                  batch_size,
                  window_size=None,
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
    file_and_sims = [(f, str(si)) for f, si in file_and_sims]

    dataset = tf.data.Dataset.from_tensor_slices(file_and_sims)
    if shuffle:
        dataset = dataset.shuffle(int(1e5))

    def split_squeeze(x):
        x, y = tf.split(x, 2, axis=-1)
        x = tf.squeeze(x, axis=-1)
        y = tf.squeeze(y, axis=-1)
        return x, y

    dataset = dataset.map(split_squeeze, 2)

    load_func = lambda filename, sim_number: tf.py_func(
        read_from_file, [filename, sim_number, True, ['A_downstream']],
        Tout=[tf.bool, tf.float32, tf.float32],
        stateful=False
    )

    dataset = dataset.map(load_func, 2)

    def split_for_pad(A, X, Y):
        X = tf.split(X, len(x_feature_subset), axis=-1)
        Y = tf.split(Y, len(y_feature_subset), axis=-1)
        squeeze = lambda t: tf.squeeze(t, -1)
        return (A,
                dict(zip(x_feature_subset, map(squeeze, X))),
                dict(zip(y_feature_subset, map(squeeze, Y))))

    dataset = dataset.map(split_for_pad, 2)

    xpad =  {x: pad_value_for_feature[x] for x in x_feature_subset}
    ypad =  {y: pad_value_for_feature[y] for y in y_feature_subset}

    dataset = dataset.padded_batch(
        batch_size,
        ((-1, -1, -1, -1), {x: (-1, -1) for x in x_feature_subset}, {y: (-1, -1) for y in y_feature_subset}),
        (False, xpad, ypad)
    )

    def stack_post_pad(A, X, Y):
        X_stacked = tf.stack([X[x] for x in x_feature_subset], -1)
        Y_stacked = tf.stack([Y[y] for y in y_feature_subset], -1)
        return A, X_stacked, Y_stacked

    dataset = dataset.map(stack_post_pad, 2)

    num_batches = math.ceil(len(file_and_sims) / batch_size)
    datasets = [dataset.skip(i).take(1) for i in range(num_batches)]

    def window(A, X, Y, window_size):
        timesteps = A.shape[1]
        for t0 in range(0, timesteps, window_size):
            t1 = t0 + window_size
            yield A[:,t0:t1], X[:,t0:t1], Y[:,t0:t1]

    window_op = lambda A, X, Y: tf.data.Dataset.from_generator(
        lambda A, X, Y: window(A, X, Y, window_size),
        datasets[0].output_types,
        datasets[0].output_shapes,
        args=(A, X, Y)
    )

    datasets = [ds.flat_map(window_op) for ds in datasets]
    datasets = [ds.prefetch(2) for ds in datasets]

    return datasets


class TFBatcher(object):
    def __init__(self,
                 train_directories,
                 batch_size,
                 window_size,
                 val_directories=None,
                 do_drop_remainder_batch=True,
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

        training_datasets = make_datasets(train_directories,
                                          batch_size,
                                          window_size,
                                          shuffle,
                                          A_name_list,
                                          x_feature_subset,
                                          y_feature_subset)

        self.iterator = tf.data.Iterator.from_structure(
            training_datasets[0].output_types,
            training_datasets[0].output_shapes)

        self.train_batches = self.make_init_ops_and_batch(training_datasets)

        self.tensor = self.iterator.get_next()
        # name the tensors
        self.A = tf.identity(self.tensor[0], name='A')
        self.X = tf.identity(self.tensor[1], name='X')
        self.Y = tf.identity(self.tensor[2], name='Y')

        if val_directories is not None:
            val_datasets = make_datasets(val_directories,
                                         batch_size,
                                         window_size,
                                         False,
                                         A_name_list,
                                         x_feature_subset,
                                         y_feature_subset)
            self.val_batches = self.make_init_ops_and_batch(val_datasets)

    def make_init_ops_and_batch(self, datasets):
        init_ops = [self.iterator.make_initializer(ds)
                    for ds in datasets]
        batches = [TFBatch(ds, init) for ds, init in zip(datasets, init_ops)]
        return batches

    @property
    def num_train_batches(self):
        return len(self.train_batches)

    @property
    def num_val_batches(self):
        return len(self.val_batches)

    @property
    def num_batches(self):
        return self.num_train_batches
