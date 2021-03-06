import logging
import os

import numpy as np
import tensorflow as tf

from keras import backend as K
from trafficgraphnn.load_data import (pad_value_for_feature,
                                      per_cycle_features_default,
                                      read_from_file,
                                      windowed_unpadded_batch_of_generators,
                                      x_feature_subset_default,
                                      y_feature_subset_default)
from trafficgraphnn.preprocessing.io import get_preprocessed_filenames
from trafficgraphnn.utils import get_num_cpus, iterfy

_logger = logging.getLogger(__name__)


class TFBatch(object):
    def __init__(self, filenames, filename_ph, initializer):
        self.filenames = filenames
        self.filename_ph = filename_ph
        self.initializer = initializer

    def initialize(self, session=None):
        if session is None:
            session = K.get_session()
        session.run(self.initializer, {self.filename_ph: self.filenames})


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


def make_dataset_fast(filename_ph,
                      batch_size,
                      window_size,
                      A_name_list=['A_downstream',
                                  'A_upstream',
                                  'A_neighbors'],
                      x_feature_subset=x_feature_subset_default,
                      y_feature_subset=y_feature_subset_default,
                      per_cycle_features=per_cycle_features_default,
                      average_interval=None,
                      num_parallel_calls=None,
                      max_time=None,
                      gpu_prefetch=True):

    if num_parallel_calls is None:
        num_parallel_calls = get_num_cpus()

    dataset = tf.data.Dataset.from_tensor_slices(filename_ph)

    def _read(filename):
        out = read_from_file(filename,
                             A_name_list=A_name_list,
                             x_feature_subset=x_feature_subset,
                             y_feature_subset=y_feature_subset,
                             max_time=max_time,
                             return_t_and_lanenames=True)
        return (*out, filename)

    dataset = dataset.map(lambda filename: tf.py_func(
        _read, [filename],
        [tf.bool, tf.float32, tf.float32, tf.float32, tf.string, tf.string]))

    def split_for_pad(A, X, Y, t, lanes, filename):
        X = tf.unstack(X, len(x_feature_subset), axis=-1)
        Y = tf.unstack(Y, len(y_feature_subset), axis=-1)
        return (A,
                dict(zip(x_feature_subset, X)),
                dict(zip(y_feature_subset, Y)),
                t, lanes, filename)

    dataset = dataset.map(split_for_pad, num_parallel_calls)

    xpad =  {x: pad_value_for_feature[x] for x in x_feature_subset}
    ypad =  {y: pad_value_for_feature[y] for y in y_feature_subset}

    if average_interval is not None and average_interval > 1:
        dataset = dataset.map(
            lambda A, X, Y, t, lanes, filenames:
            average_over_interval(A, X, Y, average_interval, t, lanes,
                                  filenames,
                                  x_feature_subset=x_feature_subset,
                                  y_feature_subset=y_feature_subset,
                                  per_cycle_features=per_cycle_features),
            num_parallel_calls=num_parallel_calls)

    dataset = dataset.padded_batch(
        batch_size,
        ((-1, -1, -1, -1), {x: (-1, -1) for x in x_feature_subset},
                           {y: (-1, -1) for y in y_feature_subset},
                           (-1,), (-1,), ()),
        (False, xpad, ypad, 0., '', ''))

    stack = lambda A, X, Y, t, lanes, filenames: stack_post_pad(
        A, X, Y, t, lanes, filenames,
        x_feature_subset=x_feature_subset, y_feature_subset=y_feature_subset)

    dataset = dataset.map(stack, num_parallel_calls=num_parallel_calls)

    dataset = dataset.prefetch(1)
    if gpu_prefetch:
        dataset = dataset.apply(tf.data.experimental.prefetch_to_device(
                                '/GPU:0'))

    return dataset


def make_dataset(filename_ph,
                 batch_size,
                 window_size,
                 A_name_list=['A_downstream',
                              'A_upstream',
                              'A_neighbors'],
                 x_feature_subset=x_feature_subset_default,
                 y_feature_subset=y_feature_subset_default,
                 per_cycle_features=per_cycle_features_default,
                 average_interval=None,
                 num_parallel_calls=None):

    if num_parallel_calls is None:
        num_parallel_calls = get_num_cpus()

    dataset = tf.data.Dataset.from_tensors(filename_ph)

    def windowed_batch_func(files):
        batch = windowed_unpadded_batch_of_generators(files,
                                                      window_size,
                                                      batch_size,
                                                      A_name_list,
                                                      x_feature_subset,
                                                      y_feature_subset,
                                                      per_cycle_features,
                                                      prefetch_all=True)
        while True:
            try:
                out = next(batch)
                yield (*out, files)
            except StopIteration:
                return

    gen_op = lambda filename: tf.data.Dataset.from_generator(
        windowed_batch_func,
        output_types=(tf.bool, tf.float32, tf.float32, tf.float32, tf.string,
                      tf.string),
        output_shapes=((None, None, None, None),
                       (None, None, len(x_feature_subset)),
                       (None, None, len(y_feature_subset)),
                       None,
                       None,
                       None),
        args=(filename,))

    dataset = dataset.flat_map(gen_op)

    dataset = dataset.map(
        lambda A, X, Y, t, lanes, filenames: split_for_pad(
            A, X, Y, t, lanes, filenames,
            x_feature_subset=x_feature_subset,
            y_feature_subset=y_feature_subset),
        num_parallel_calls)

    xpad =  {x: pad_value_for_feature[x] for x in x_feature_subset}
    ypad =  {y: pad_value_for_feature[y] for y in y_feature_subset}

    if average_interval is not None and average_interval > 1:
        dataset = dataset.map(
            lambda A, X, Y, t, lanes, filenames:
            average_over_interval(A, X, Y, average_interval, t, lanes,
                                  filenames,
                                  x_feature_subset=x_feature_subset,
                                  y_feature_subset=y_feature_subset,
                                  per_cycle_features=per_cycle_features),
            num_parallel_calls=num_parallel_calls)

    dataset = dataset.padded_batch(
        batch_size,
        ((-1, -1, -1, -1), {x: (-1, -1) for x in x_feature_subset},
                           {y: (-1, -1) for y in y_feature_subset},
                           (-1,), (-1,), (-1,)),
        (False, xpad, ypad, 0., '', ''))

    stack = lambda A, X, Y, t, lanes, filenames: stack_post_pad(
        A, X, Y, t, lanes, filenames,
        x_feature_subset=x_feature_subset, y_feature_subset=y_feature_subset)

    dataset = dataset.map(stack, num_parallel_calls=num_parallel_calls)

    dataset = dataset.prefetch(1)

    return dataset


def average_over_interval(A, X, Y, average_interval, t, lanes, filename,
                          x_feature_subset, y_feature_subset,
                          per_cycle_features):
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

            if feature in per_cycle_features:
                averaged = tf.reduce_max(reshaped, 1)
            else:
                averaged = tf.reduce_mean(reshaped, 1)
            outputs[feature] = averaged

        return outputs

    new_X = pad_reshape_average(X)
    new_Y = pad_reshape_average(Y)

    get_slice = tf.range(
        0,
        tf.cast(num_intervals * average_interval, tf.int32),
        average_interval, dtype=tf.int32)

    new_A = tf.gather(A, get_slice)
    new_t = tf.gather(t, get_slice)

    return new_A, new_X, new_Y, new_t, lanes, filename


def split_for_pad(A, X, Y, t, lanes, filenames,
                  x_feature_subset, y_feature_subset):
    X = tf.unstack(X, len(x_feature_subset), axis=-1)
    Y = tf.unstack(Y, len(y_feature_subset), axis=-1)
    return (A,
            dict(zip(x_feature_subset, X)),
            dict(zip(y_feature_subset, Y)),
            t, lanes, filenames)


def stack_post_pad(A, X, Y, t, lanes, filenames,
                   x_feature_subset, y_feature_subset):
    X_stacked = tf.stack([X[x] for x in x_feature_subset], -1)
    Y_stacked = tf.stack([Y[y] for y in y_feature_subset], -1)
    return A, X_stacked, Y_stacked, t, lanes, filenames


class TFBatcher(object):
    def __init__(self,
                 filenames_or_dirs,
                 batch_size,
                 window_size,
                 average_interval=None,
                 val_proportion=.1,
                 test_proportion=.1,
                 sub_batching=False,
                 shuffle=True,
                 A_name_list=['A_downstream',
                              'A_upstream',
                              'A_neighbors'],
                 x_feature_subset=x_feature_subset_default,
                 y_feature_subset=y_feature_subset_default,
                 per_cycle_features=per_cycle_features_default,
                 flatten_A=False,
                 max_time=None,
                 gpu_prefetch=True):

        filenames_or_dirs = iterfy(filenames_or_dirs)
        filenames = []
        for entry in filenames_or_dirs:
            if os.path.isfile(entry):
                filenames.append(entry)
            elif os.path.isdir(entry):
                filenames.extend(get_preprocessed_filenames(entry))
        filenames.sort()
        self.batch_size = batch_size
        self.window_size = window_size
        self.average_interval = average_interval
        self.val_proportion = val_proportion
        self.sub_batching = sub_batching
        self.shuffle_on_epoch = shuffle
        self.A_name_list = A_name_list
        self.x_feature_subset = x_feature_subset
        self.y_feature_subset = y_feature_subset
        self.per_cycle_features = per_cycle_features
        self.flat_A = flatten_A

        num_validation = int(len(filenames) * val_proportion)
        num_test = int(len(filenames) * test_proportion)

        self.train_files = filenames[:-(num_validation+num_test)]
        self.val_files = filenames[-(num_validation+num_test):-num_test]
        self.test_files = filenames[-num_test:]

        self.filename_ph = tf.placeholder(tf.string, [None], 'filenames')

        if sub_batching:
            self._tf_dataset = make_dataset(self.filename_ph,
                                            batch_size,
                                            window_size,
                                            A_name_list,
                                            x_feature_subset,
                                            y_feature_subset,
                                            per_cycle_features,
                                            average_interval)
        else:
            self._tf_dataset = make_dataset_fast(self.filename_ph,
                                                 batch_size,
                                                 window_size,
                                                 A_name_list,
                                                 x_feature_subset,
                                                 y_feature_subset,
                                                 per_cycle_features,
                                                 average_interval,
                                                 max_time=max_time,
                                                 gpu_prefetch=gpu_prefetch)

        self.init_initializable_iterator()
        self._make_batches()

    def _make_batches(self):
        self.train_file_batches = self._split_batches(self.train_files)
        self.val_file_batches = self._split_batches(self.val_files)
        self.test_file_batches = self._split_batches(self.test_files)

        self.val_batches = [TFBatch(files, self.filename_ph, self.init_op)
                            for files in self.val_file_batches]
        self.test_batches = [TFBatch(files, self.filename_ph, self.init_op)
                                for files in self.test_file_batches]

    def _split_batches(self, filename_list):
        if self.sub_batching:
            return [filename_list[i:i+self.batch_size] for i
                    in range(0, len(filename_list), self.batch_size)]
        else:
            return [filename_list]

    def shuffle(self):
        np.random.shuffle(self.train_files)

    def init_epoch(self):
        if self.shuffle_on_epoch:
            self.shuffle()
        self.train_file_batches = self._split_batches(self.train_files)
        self.train_batches = [TFBatch(files, self.filename_ph, self.init_op)
                              for files in self.train_file_batches]

    def init_initializable_iterator(self):
        self.iterator = self._tf_dataset.make_initializable_iterator()
        self.init_op = self.iterator.initializer

        self._init_outputs()

    def init_feedable_iterator(self, session):
        self.handle = tf.placeholder(tf.string, [])

        self.iterator = tf.data.Iterator.from_string_handle(
            self.handle,
            self._tf_dataset.output_types,
            self._tf_dataset.output_shapes)

        self._init_outputs()

    def _init_outputs(self):
        self.tensor = self.iterator.get_next()
        # name the tensors
        self.A = tf.identity(self.tensor[0], name='A')
        self.X = tf.identity(self.tensor[1], name='X')
        self.Y = tf.identity(self.tensor[2], name='Y')
        self.t = tf.identity(self.tensor[3], name='t')
        self.lanes = tf.identity(self.tensor[4], name='lanes')
        self.filename_tensor = tf.identity(self.tensor[5], name='filenames')

        self.Y_slices = tf.unstack(self.Y, axis=-1)

        if self.flat_A:
            self._flatten_A()

    def _flatten_A(self):
        self.A = tf.reduce_any(self.A, 2, keepdims=True, name='A_flat')

    @property
    def num_train_batches(self):
        if self.sub_batching:
            return len(self.train_file_batches)
        else:
            return np.ceil(len(self.train_file_batches[0]) / self.batch_size)
