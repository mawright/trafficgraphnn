import datetime
import json
import logging
import os
import re
import time
import warnings
from collections import namedtuple

import numpy as np
import pandas as pd
import tables
import tensorflow as tf

import keras.backend as K
from keras.callbacks import (BaseLogger, CallbackList, CSVLogger, History,
                             ModelCheckpoint, ProgbarLogger, ReduceLROnPlateau,
                             TensorBoard, TerminateOnNaN)
from trafficgraphnn.utils import _col_dtype_key, iterfy

_logger = logging.getLogger(__name__)


def make_callbacks(model, model_save_dir, do_validation=False):
    timestamp = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
    callback_list = CallbackList()
    callback_list.append(BaseLogger())
    callback_list.append(TerminateOnNaN())
    callback_list.append(CSVLogger(os.path.join(model_save_dir,
                                                timestamp, 'log.csv')))
    callback_list.append(
        TensorBoard(log_dir=os.path.join(
            model_save_dir, 'logs', timestamp), update_freq=1000))
    history = History()
    callback_list.append(history)
    model.history = history
    if do_validation:
        display_metrics = ['val_' + n for n in model.metrics_names]
        callback_list.append(ProgbarLogger('steps', stateful_metrics=display_metrics))
    else:
        display_metrics = model.metrics_names
        callback_list.append(ProgbarLogger('steps'))

    filename = 'weights_epoch{epoch:02d}-'

    for metric in display_metrics:
        add_str = '{%s:.4f}' % metric
        filename = filename + add_str
    filename = filename + '.hdf5'
    callback_list.append(ModelCheckpoint(os.path.join(model_save_dir,
                                                      timestamp, filename)))
    callback_list.append(ReduceLROnPlateau(verbose=1))

    callback_list.set_model(model)
    return callback_list


def get_logging_dir(callback_list):
    for callback in callback_list:
        if isinstance(callback, CSVLogger):
            return os.path.dirname(callback.filename)


def set_callback_params(callbacks,
                        epochs,
                        batch_size,
                        verbose,
                        do_validation,
                        model,
                        steps=None):
    if do_validation:
        metrics = model.metrics_names + ['val_' + n for n in model.metrics_names]
    else:
        metrics = model.metrics_names
    params_dict = {
        'batch_size': batch_size,
        'epochs': epochs,
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': metrics
    }
    if steps is not None:
        params_dict['steps'] = steps
    callbacks.set_params(params_dict)


def fit_loop_init(model, callbacks):
    callbacks.on_train_begin()
    model.reset_states()
    model._make_train_function()
    model._make_test_function()
    model.stop_training = False


def named_logs(model, logs):
    result = {}
    for metric, l in zip(model.metrics_names, logs):
        result[metric] = l
    return result


def val_named_logs(model, logs):
    result = {}
    for metric, l in zip(model.metrics_names, logs):
        result['val_' + metric] = l
    return result


def fit_loop_tf(model, callbacks, batch_generator, num_epochs, feed_dict=None):
    callbacks.on_train_begin()

    for epoch in range(num_epochs):
        fit_loop_train_one_epoch_tf(model, callbacks, batch_generator, epoch,
                                    feed_dict=feed_dict)

def fit_loop_train_one_epoch_tf(model, callbacks, batch_generator, epoch,
                                feed_dict=None):
    callbacks.on_epoch_begin(epoch)
    batch_generator.init_epoch()

    # set up bookkeeping
    batch_size = batch_generator.batch_size * batch_generator.window_size
    if batch_generator.average_interval is not None:
        batch_size = batch_size // batch_generator.average_interval
    i_step = 0
    sess = K.get_session()
    for i_batch, batch in enumerate(batch_generator.train_batches):
        t0 = time.time()
        model.reset_states()
        batch.initialize(sess)

        while True:
            try:
                tstep = time.time()

                callbacks.on_batch_begin(i_step)

                logs = model.train_on_batch(x=None, y=None)
                train_step_time = time.time() - tstep

                logs = named_logs(model, logs)
                logs['size'] = batch_size
                logs['batch'] = i_step
                logs['time'] = train_step_time

                callbacks.on_batch_end(i_step, logs)
                i_step += 1
            except tf.errors.OutOfRangeError:
                # this batch of timeseries is over
                break
            finally:
                if model.stop_training:
                    raise RuntimeError
        _logger.debug('Training on batch %s took %s s',
                      i_batch, time.time() - t0)

    val_logs = {'val_' + m: [] for m in model.metrics_names}
    i_step = 0
    for i_batch, batch in enumerate(batch_generator.val_batches):
        model.reset_states()
        batch.initialize(sess)

        while True:
            try:
                tstep = time.time()

                callbacks.on_test_batch_begin(i_step)
                logs = model.test_on_batch(x=None, y=None)
                step_time = time.time() - tstep

                logs = val_named_logs(model, logs)
                logs['size'] = batch_size
                logs['batch'] = i_step
                logs['time'] = step_time
                for k, v in val_logs.items():
                    v.append(logs[k])

                callbacks.on_test_batch_end(i_step, logs)
                i_step += 1
            except tf.errors.OutOfRangeError:
                break
        _logger.debug('Evaluating on batch %s took %f s',
                      i_batch, time.time() - t0)

    for k in val_logs.keys():
        val_logs[k] = np.mean(val_logs[k])

    callbacks.on_epoch_end(epoch, val_logs)


class PredictEvalFunction(object):
    def __init__(self, model, extra_outputs=None):

        self.model = model
        if extra_outputs is None:
            extra_outputs = []

        # out holding class defs
        self.Input = namedtuple('Input',
                                map(_clean_str, self.model.input_names))
        self.Output = namedtuple('Output',
                                 map(_clean_str, self.model.output_names))
        self.Metrics = namedtuple('Metrics',
                                  map(_clean_str, self.model.metrics_names))
        self.Extras = namedtuple('Extras',
                                 map(_clean_str,
                                     [out.name for out in extra_outputs]))
        # construction
        inputs = (model._feed_inputs
                  + model._feed_targets
                  + model._feed_sample_weights)
        if model._uses_dynamic_learning_phase():
            inputs += [K.learning_phase()]

        outputs = (model.inputs
                   + model.targets
                   + model.outputs
                   + [model.total_loss]
                   + model.metrics_tensors
                   + extra_outputs)
        # Gets network outputs. Does not update weights.
        # Does update the network states.
        kwargs = getattr(model, '_function_kwargs', {})
        predict_eval_function = K.function(
            inputs,
            outputs,
            updates=model.state_updates + model.metrics_updates,
            name='predict_eval_function',
            **kwargs)
        model.predict_eval_function = self

        self.outputs = outputs
        self.func = predict_eval_function

        # bookkeeping for unpacking
        self.len_inputs = len(model.inputs)
        self.len_targets = len(model.targets)
        self.len_outputs = len(model.outputs)
        self.len_metrics = len(model.metrics_tensors) + 1 # plus loss
        self.len_extra_outputs = len(extra_outputs)

    def call(self, inputs=None):
        inputs = iterfy(inputs)
        func_output = self.func(inputs)
        (inputs, targets, outputs, metrics, extra_outputs
         ) = self._unpack(func_output)

        return dict(inputs=self.Input(*inputs),
                    targets=self.Output(*targets),
                    outputs=self.Output(*outputs),
                    metrics=self.Metrics(*metrics),
                    extra_outputs=self.Extras(*extra_outputs))

    def _unpack(self, output):
        i = 0
        for sublist_len in [self.len_inputs, self.len_targets,
                            self.len_outputs,
                            self.len_metrics, self.len_extra_outputs]:
            yield output[i:i+sublist_len]
            i += sublist_len


def predict_eval_tf(model, callbacks, batch_generator):
    if not hasattr(model, 'predict_eval_function'):
        func = PredictEvalFunction(model,
                                   [batch_generator.t, batch_generator.lanes])
    else:
        func = model.predict_eval_function
    model_write_dir = get_logging_dir(callbacks)
    result_file = os.path.join(model_write_dir, 'results.hdf')

    i_step = 0
    val_logs = {'val_' + m: [] for m in model.metrics_names}

    with pd.HDFStore(result_file, 'w') as result_store:
        for batch in batch_generator.val_batches:
            filenames = [os.path.basename(f) for f in batch.filenames]
            model.reset_states()
            batch.initialize()

            while True:
                try:
                    out = func.call()
                    __append_results(result_store, filenames, out,
                                     batch_generator.x_feature_subset,
                                     batch_generator.y_feature_subset)
                    metrics = out['metrics']

                    logs = val_named_logs(model, metrics)
                    logs['step'] = i_step
                    for k, v in val_logs.items():
                        v.append(logs[k])

                except tf.errors.OutOfRangeError:
                    __sort_store_dfs(result_store, filenames)
                    break

    mean_metrics = {k: np.mean(v) for k, v in val_logs.items()}
    with open(os.path.join(model_write_dir, 'metrics.json'), 'w') as f:
        json.dump(mean_metrics, f)


def _df(data, lane_list, timestep_list, colnames):
    reshaped = np.reshape(data, (len(lane_list)*len(timestep_list), -1))
    index = pd.MultiIndex.from_product([timestep_list, lane_list],
                                        names=['begin', 'lane'])
    dtypes = {col: _col_dtype_key[col] for col in colnames}
    return pd.DataFrame(reshaped, index=index, columns=colnames, dtype=dtypes)


def __append_results(store, filenames, func_out, x_colnames, y_colnames):
    timesteps = func_out['extra_outputs'][0]
    lanes = func_out['extra_outputs'][1]
    X = func_out['inputs'].X
    Y = np.stack(func_out['targets'], -1)
    Yhat = np.stack(func_out['outputs'], -1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', tables.NaturalNameWarning)
        for i, filename in enumerate(filenames):
            t_i = timesteps[i]
            lanes_i = list(map(__maybe_decode, lanes[i]))
            store.append(filename + '/X', _df(X[i], lanes_i, t_i, x_colnames))
            store.append(filename + '/Y', _df(Y[i], lanes_i, t_i, y_colnames))
            store.append(filename + '/Yhat',
                         _df(Yhat[i], lanes_i, t_i, y_colnames))


def __sort_store_dfs(store, filenames):
    for f in filenames:
        for table in [f + t for t in ['/X', '/Y', '/Yhat']]:
            store[table] = store[table].sort_index(level=[0,1])


def __maybe_decode(item):
    try:
        return item.decode()
    except AttributeError:
        return item


def _clean_str(s):
    return re.sub('[^0-9a-zA-Z_]', '_', s)
