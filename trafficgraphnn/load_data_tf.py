import logging
import math
import os
import time

import tensorflow as tf

from trafficgraphnn.load_data import (on_green_feats_default,
                                      pad_value_for_feature,
                                      windowed_unpadded_batch_of_generators,
                                      x_feature_subset_default,
                                      y_feature_subset_default)
from trafficgraphnn.preprocessing.io import get_preprocessed_filenames
from trafficgraphnn.utils import iterfy

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


def make_datasets(filenames,
                  batch_size,
                  window_size=None,
                  shuffle=True,
                  A_name_list=['A_downstream',
                               'A_upstream',
                               'A_neighbors'],
                  x_feature_subset=x_feature_subset_default,
                  y_feature_subset=y_feature_subset_default,
                  y_on_green_mask_feats=on_green_feats_default,
                  average_interval=None):

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle:
        dataset = dataset.shuffle(int(1e5))

    dataset = dataset.batch(batch_size)

    num_batches = math.ceil(len(filenames) / batch_size)
    datasets = [dataset.skip(i).take(1) for i in range(num_batches)]

    def windowed_batch_func(files):
        return windowed_unpadded_batch_of_generators(files,
                                                     window_size,
                                                     batch_size,
                                                     A_name_list,
                                                     x_feature_subset,
                                                     y_feature_subset,
                                                     y_on_green_mask_feats,
                                                     prefetch_all=True)

    gen_op = lambda filename: tf.data.Dataset.from_generator(
        windowed_batch_func,
        output_types=(tf.bool, tf.float32, tf.float32),
        output_shapes=((None, None, None, None),
                       (None, None, len(x_feature_subset)),
                       (None, None, len(y_feature_subset))),
        args=(filename,)
    )

    datasets = [ds.flat_map(gen_op) for ds in datasets]

    def split_for_pad(A, X, Y):
        X = tf.unstack(X, axis=-1)
        Y = tf.unstack(Y, axis=-1)
        return (A,
                dict(zip(x_feature_subset, X)),
                dict(zip(y_feature_subset, Y)))

    datasets = [ds.map(split_for_pad, 2) for ds in datasets]

    xpad =  {x: pad_value_for_feature[x] for x in x_feature_subset}
    ypad =  {y: pad_value_for_feature[y] for y in y_feature_subset}

    if average_interval is not None and average_interval > 1:
        def average_over_interval(A, X, Y, average_interval):
            shape = tf.shape(X[x_feature_subset[0]])
            num_timesteps = shape[0]
            divided = num_timesteps / average_interval
            num_intervals = tf.ceil(divided)
            deficit = tf.cast(num_intervals * average_interval, tf.int32) - num_timesteps
            padding = [[0, deficit], [0, 0]]

            def pad_reshape_average(tensor_dict):
                outputs = {}
                for feature, tensor in tensor_dict.items():
                    PAD_VALUE = pad_value_for_feature[feature]
                    padded = tf.pad(tensor, padding, constant_values=PAD_VALUE)

                    reshaped = tf.reshape(
                        padded, [num_intervals, average_interval, shape[-1]])

                    if feature in y_on_green_mask_feats:
                        averaged = tf.reduce_max(reshaped, 1)
                    else:
                        averaged = tf.reduce_mean(reshaped, 1)
                    outputs[feature] = averaged

                return outputs

            new_X = pad_reshape_average(X)
            new_Y = pad_reshape_average(Y)

            get_As = tf.range(0, num_timesteps, average_interval)
            new_A = tf.gather(A, get_As)

            return new_A, new_X, new_Y
        datasets = [ds.map(lambda A, X, Y:
                           average_over_interval(A, X, Y, average_interval))
                    for ds in datasets]

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
                 filenames_or_dirs,
                 batch_size,
                 window_size,
                 average_interval=None,
                 val_proportion=.2,
                 shuffle=True,
                 A_name_list=['A_downstream',
                              'A_upstream',
                              'A_neighbors'],
                 x_feature_subset=x_feature_subset_default,
                 y_feature_subset=y_feature_subset_default,
                 y_on_green_mask_feats=on_green_feats_default):

        filenames_or_dirs = iterfy(filenames_or_dirs)
        filenames = []
        for entry in filenames_or_dirs:
            if os.path.isfile(entry):
                filenames.append(entry)
            elif os.path.isdir(entry):
                filenames.extend(get_preprocessed_filenames(entry))

        num_training = int(len(filenames) * val_proportion)

        train_files = filenames[:-num_training]
        val_files = filenames[-num_training:]

        t0 = time.time()
        self._train_datasets = make_datasets(train_files,
                                             batch_size,
                                             window_size,
                                             shuffle,
                                             A_name_list,
                                             x_feature_subset,
                                             y_feature_subset,
                                             y_on_green_mask_feats,
                                             average_interval)
        if len(val_files) > 0:
            self._val_datasets = make_datasets(val_files,
                                               batch_size,
                                               window_size,
                                               False,
                                               A_name_list,
                                               x_feature_subset,
                                               y_feature_subset,
                                               y_on_green_mask_feats,
                                               average_interval)
        else:
            self._val_datasets = None
        t = time.time() - t0
        _logger.debug('Made TFBatcher object in %s s', t)

        self.batch_size = batch_size
        self.window_size = window_size

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

        self.Y_slices = tf.unstack(self.Y, axis=-1)

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
