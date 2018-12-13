import math
from collections import namedtuple

import tensorflow as tf

from trafficgraphnn.load_data import Batch, get_file_and_sim_indeces_in_dirs

TFBatch = namedtuple('TFBatch', ['dataset', 'initializer'])

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

        def make_datasets(directories):
            file_and_sims = get_file_and_sim_indeces_in_dirs(directories)
            file_and_sims = [(f, str(si)) for f, si in file_and_sims]

            dataset = tf.data.Dataset.from_tensor_slices(file_and_sims)
            if shuffle:
                dataset = dataset.shuffle(int(1e5))
            dataset = dataset.batch(batch_size,
                                    drop_remainder=do_drop_remainder_batch)

            def split_squeeze(x):
                x, y = tf.split(x, 2, axis=1)
                x = tf.squeeze(x, axis=1)
                y = tf.squeeze(y, axis=1)
                return x, y
            dataset = dataset.map(split_squeeze)

            num_batches = math.ceil(len(file_and_sims) / batch_size)
            datasets = [dataset.skip(i).take(1) for i in range(num_batches)]

            def genfunc(filenames, sim_indeces):
                batch = Batch(filenames, sim_indeces, window_size,
                              A_name_list=A_name_list,
                              x_feature_subset=x_feature_subset,
                              y_feature_subset=y_feature_subset)
                return batch.iterate()

            gen = lambda f, si: tf.data.Dataset.from_generator(
                genfunc,
                (tf.bool, tf.float32, tf.float32),
                output_shapes=(
                    # A: batch x time x depth x lane x lane
                    (None, None, None, None, None),
                    # X: batch x time x lane x feat
                    (None, None, None, len(x_feature_subset)),
                    # Y: batch x time x lane x feat
                    (None, None, None, len(y_feature_subset))
                ), args=(f, si))

            datasets = [ds.flat_map(gen) for ds in datasets]
            def set_names(A, X, Y):
                return {'A': A, 'X': X, 'Y': Y}
            datasets = [ds.map(set_names, num_parallel_calls=10)
                        for ds in datasets]
            datasets = [ds.prefetch(None) for ds in datasets]
            return datasets

        training_datasets = make_datasets(train_directories)

        self.iterator = tf.data.Iterator.from_structure(
            training_datasets[0].output_types,
            training_datasets[0].output_shapes)

        self.train_batches = self.make_init_ops_and_batch(training_datasets)

        self.tensor = self.iterator.get_next()
        self.A = tf.identity(self.tensor['A'], name='A')
        self.X = tf.identity(self.tensor['X'], name='X')
        self.Y = tf.identity(self.tensor['Y'], name='Y')

        if val_directories is not None:
            val_datasets = make_datasets(val_directories)
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
