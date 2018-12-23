import logging
import math
import time

import tensorflow as tf

from trafficgraphnn.load_data import (get_file_and_sim_indeces_in_dirs,
                                      pad_value_for_feature,
                                      windowed_unpadded_batch_of_generators)

_logger = logging.getLogger(__name__)


class TFBatch(object):
    def __init__(self, dataset, initializer):
        self.dataset = dataset
        self.initializer = initializer

    def initialize(self, session, *args, **kwargs):
        session.run(self.initializer)

    def set_self_as_active(self, **kwargs):
        pass


class TFHandleBatch(object):
    def __init__(self, dataset, session, parent):
        self.dataset = dataset
        self.iterator = dataset.make_initializable_iterator()
        self.handle = session.run(self.iterator.string_handle())
        self.parent = parent

    @property
    def initializer(self):
        return self.iterator.initializer

    def initialize(self, session, feed_dict=None):
        session.run(self.initializer)
        if feed_dict is not None:
            self.set_self_as_active(feed_dict)

    def set_self_as_active(self, feed_dict):
        feed_dict[self.parent.handle] = self.handle


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
                                    'e2_0/maxJamLengthInMeters'],
                  buffer_size=10):

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

    dataset = dataset.apply(tf.data.experimental.map_and_batch(split_squeeze,
                                                               batch_size,
                                                               2))

    num_batches = math.ceil(len(file_and_sims) / batch_size)
    datasets = [dataset.skip(i).take(1) for i in range(num_batches)]


    def windowed_padded_batch_func(filenames, sim_number):
        return windowed_unpadded_batch_of_generators(filenames, sim_number,
                                                     window_size,
                                                     batch_size,
                                                     A_name_list,
                                                     x_feature_subset,
                                                     y_feature_subset,
                                                     buffer_size)

    gen_op = lambda filename, sim_number: tf.data.Dataset.from_generator(
        windowed_padded_batch_func,
        output_types=(tf.bool, tf.float32, tf.float32),
        output_shapes=((None, None, None, None),
                       (None, None, 6),
                       (None, None, 2)),
        args=(filename, sim_number)
    )

    datasets = [ds.flat_map(gen_op) for ds in datasets]

    def split_for_pad(A, X, Y):
        X = tf.split(X, len(x_feature_subset), axis=-1)
        Y = tf.split(Y, len(y_feature_subset), axis=-1)
        squeeze = lambda t: tf.squeeze(t, -1)
        return (A,
                dict(zip(x_feature_subset, map(squeeze, X))),
                dict(zip(y_feature_subset, map(squeeze, Y))))

    datasets = [ds.map(split_for_pad, 2) for ds in datasets]

    xpad =  {x: pad_value_for_feature[x] for x in x_feature_subset}
    ypad =  {y: pad_value_for_feature[y] for y in y_feature_subset}

    datasets = [ds.padded_batch(
        batch_size,
        ((-1, -1, -1, -1), {x: (-1, -1) for x in x_feature_subset},
                           {y: (-1, -1) for y in y_feature_subset}),
        (False, xpad, ypad)
    ) for ds in datasets]

    def stack_post_pad(A, X, Y):
        X_stacked = tf.stack([X[x] for x in x_feature_subset], -1)
        Y_stacked = tf.stack([Y[y] for y in y_feature_subset], -1)
        return A, X_stacked, Y_stacked

    datasets = [ds.map(stack_post_pad, 2) for ds in datasets]

    datasets = [ds.prefetch(1) for ds in datasets]

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
                                   'e2_0/maxJamLengthInMeters'],
                 buffer_size=10):

        t0 = time.time()
        self._train_datasets = make_datasets(train_directories,
                                             batch_size,
                                             window_size,
                                             shuffle,
                                             A_name_list,
                                             x_feature_subset,
                                             y_feature_subset,
                                             buffer_size)
        if val_directories is not None:
            self._val_datasets = make_datasets(val_directories,
                                               batch_size,
                                               window_size,
                                               False,
                                               A_name_list,
                                               x_feature_subset,
                                               y_feature_subset,
                                               buffer_size)
        else:
            self._val_datasets = None
        t = time.time() - t0
        _logger.debug('Made TFBatcher object in %s s', t)

    def make_init_ops_and_batch(self, datasets):
        init_ops = [self.iterator.make_initializer(ds)
                    for ds in datasets]
        batches = [TFBatch(ds, init) for ds, init in zip(datasets, init_ops)]
        return batches

    def init_initializable_iterator(self):
        self.iterator = tf.data.Iterator.from_structure(
            self._train_datasets[0].output_types,
            self._train_datasets[0].output_shapes)

        self.train_batches = self.make_init_ops_and_batch(self._train_datasets)
        if self._val_datasets is not None:
            self.val_batches = self.make_init_ops_and_batch(self._val_datasets)

        self._init_outputs()

    def init_feedable_iterator(self, session):
        self.handle = tf.placeholder(tf.string, [])

        self.iterator = tf.data.Iterator.from_string_handle(
            self.handle,
            self._train_datasets[0].output_types,
            self._train_datasets[0].output_shapes)

        self.train_batches = self.make_per_dataset_iterators_and_handles(
            self._train_datasets, session)
        if self._val_datasets is not None:
            self.val_batches = self.make_per_dataset_iterators_and_handles(
                self._val_datasets, session)

        self._init_outputs()

    def _init_outputs(self):
        self.tensor = self.iterator.get_next()
        # name the tensors
        self.A = tf.identity(self.tensor[0], name='A')
        self.X = tf.identity(self.tensor[1], name='X')
        self.Y = tf.identity(self.tensor[2], name='Y')

    def make_per_dataset_iterators_and_handles(self, datasets, session):
        batches = [TFHandleBatch(ds, session, self) for ds in datasets]
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
