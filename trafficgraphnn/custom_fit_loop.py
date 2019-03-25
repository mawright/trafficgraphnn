import datetime
import json
import logging
import os
import re
import time
import warnings
from collections import Iterable, defaultdict, namedtuple

import numpy as np
import pandas as pd
import tables
import tensorflow as tf

import keras.backend as K
from keras.callbacks import (BaseLogger, Callback, CallbackList, CSVLogger,
                             EarlyStopping, History, ModelCheckpoint,
                             ProgbarLogger, ReduceLROnPlateau, TensorBoard,
                             TerminateOnNaN)
from trafficgraphnn.utils import col_type, iterfy

_logger = logging.getLogger(__name__)

class BestWeightRestorer(Callback):
    def __init__(self,
                 monitor='val_loss',
                 mode='auto'):
        self.monitor = monitor
        if mode not in ['auto', 'min', 'max']:
            warnings.warn('BestWeightRestorer mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current, self.best):
            self.best = current
            self.stopped_epoch = epoch
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        _logger.info('Best weights on epoch %05d. Restoring.',
                     self.stopped_epoch + 1)
        self.model.set_weights(self.best_weights)

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                'Best weight saving conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        return monitor_value


class MakeSetOpsCallback(Callback):
    """Hack to prevent callbacks that set tensor values during training
    from making an assignment op during training."""
    def on_train_begin(self, logs):
        lr = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, lr)

        weights = self.model.get_weights()
        self.model.set_weights(weights)


def make_callbacks(model, model_save_dir, do_validation=False,
                   base_model=None):
    timestamp = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
    callback_list = CallbackList()
    callback_list.append(BaseLogger())
    callback_list.append(TerminateOnNaN())
    callback_list.append(CSVLogger(os.path.join(model_save_dir,
                                                timestamp, 'log.csv')))
    callback_list.append(EarlyStopping(patience=20, restore_best_weights=True))
    callback_list.append(
        TensorBoard(log_dir=os.path.join(
            model_save_dir, 'logs', timestamp), update_freq=10
            ))
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

    add_str = '{val_loss:.4f}'
    filename = filename + add_str
    filename = filename + '.hdf5'
    callback_list.append(ModelCheckpoint(os.path.join(model_save_dir,
                                                      timestamp, filename),
                                         save_best_only=True,
                                         mode='min'))
    callback_list.append(ReduceLROnPlateau(verbose=1, cooldown=25))

    # prevent this op from being created during training
    callback_list.append(MakeSetOpsCallback())

    callback_list.set_model(model)
    if base_model is not None:
        for cbk in callback_list:
            if isinstance(cbk, ModelCheckpoint):
                cbk.set_model(base_model)

    return callback_list


def get_logging_dir(callback_list):
    for callback in callback_list:
        if isinstance(callback, CSVLogger):
            return os.path.dirname(callback.filename)
        elif isinstance(callback, ModelCheckpoint):
            return os.path.dirname(callback.filepath)


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


def fit_loop_init(model, callbacks, batch_gen=None):
    callbacks.on_train_begin()
    model.reset_states()
    model._make_train_function()
    model._make_test_function()
    model.stop_training = False
    if batch_gen is not None:
        prep_predict_eval(model, batch_gen)


def prep_predict_eval(model, batch_gen):
    extras = []
    if hasattr(batch_gen, 't'):
        extras.append(batch_gen.t)
    if hasattr(batch_gen, 'lanes'):
        extras.append(batch_gen.lanes)
    PredictEvalFunction(model, extras)


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


def mean_logs(model, logs):
    result = {}
    if hasattr(logs, 'keys'):
        for metric in model.metrics_names:
            result[metric] = np.mean(logs[metric])
    elif isinstance(logs, Iterable):
        for metric, l in zip(model.metrics_names, logs):
            result[metric] = l
    return result


def fit_loop_tf(model, callbacks, batch_generator, num_epochs, feed_dict=None,
                per_step_metrics=False):
    callbacks.on_train_begin()

    for epoch in range(num_epochs):
        fit_loop_train_one_epoch_tf(
            model, callbacks, batch_generator, epoch,
            feed_dict=feed_dict, per_step_metrics=per_step_metrics)
        if model.stop_training:
            break

    callbacks.on_train_end()

def fit_loop_train_one_epoch_tf(model, callbacks, batch_generator, epoch,
                                feed_dict=None, per_step_metrics=False):
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
        if not per_step_metrics:
            callbacks.on_batch_begin(i_batch)
            logs = defaultdict(list)

        while True:
            try:
                tstep = time.time()

                if per_step_metrics:
                    callbacks.on_batch_begin(i_step)

                step_logs = model.train_on_batch(x=None, y=None)
                train_step_time = time.time() - tstep

                step_logs = named_logs(model, step_logs)
                step_logs['size'] = batch_size
                step_logs['batch'] = i_step
                step_logs['time'] = train_step_time

                if per_step_metrics:
                    callbacks.on_batch_end(i_step, step_logs)
                else:
                    for k, v in step_logs.items():
                        logs[k].append(v)
                i_step += 1
            except tf.errors.OutOfRangeError:
                # this batch of timeseries is over
                if not per_step_metrics:
                    logs['batch'] = i_batch
                    logs['time'] = np.sum(logs['time'])
                    logs['size'] = np.sum(logs['size'])
                    logs.update(mean_logs(model, logs))
                    callbacks.on_batch_end(i_batch, logs)
                break
            finally:
                if model.stop_training:
                    break
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
            finally:
                if model.stop_training:
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


def predict_eval_tf(model, write_dir, batch_generator):
    if not hasattr(model, 'predict_eval_function'):
        func = PredictEvalFunction(model,
                                   [batch_generator.t, batch_generator.lanes])
    else:
        func = model.predict_eval_function
    result_file = os.path.join(write_dir, 'results.hdf')

    val_logs = defaultdict(list)

    with pd.HDFStore(result_file, 'w') as result_store:
        for batch in batch_generator.val_batches:
            filenames = [os.path.basename(f) for f in batch.filenames]
            filenums = [os.path.splitext(f)[0] for f in filenames]
            model.reset_states()
            batch.initialize()

            while True:
                try:
                    out = func.call()
                    __append_results(result_store, filenums, out,
                                     batch_generator.x_feature_subset,
                                     batch_generator.y_feature_subset)
                    metrics = out['metrics']

                    logs = val_named_logs(model, metrics)
                    for k, v in logs.items():
                        val_logs[k].append(v)

                except tf.errors.OutOfRangeError:
                    __sort_store_dfs(result_store, filenums)
                    break

    mean_metrics = {k: np.mean(v).tolist() for k, v in val_logs.items()}
    log_str = 'Metrics on validation set:\n' + '\n'.join(
        '{}: {}'.format(k, v) for k, v in mean_metrics.items())
    _logger.info(log_str)
    with open(os.path.join(write_dir, 'metrics.json'), 'w') as f:
        json.dump(mean_metrics, f)


def _df(data, lane_list, timestep_list, colnames):
    reshaped = np.reshape(data, (len(lane_list)*len(timestep_list), -1))
    index = pd.MultiIndex.from_product([timestep_list, lane_list],
                                        names=['begin', 'lane'])
    dtypes = {col: col_type(col) for col in colnames}
    df = pd.DataFrame(reshaped, index=index, columns=colnames)
    return df.astype(dtypes)


def __append_results(store, table_prefixes, func_out, x_colnames, y_colnames):
    timesteps = func_out['extra_outputs'][0]
    lanes = func_out['extra_outputs'][1]
    X = func_out['inputs'].X
    Y = np.stack(func_out['targets'], -1)
    Yhat = np.stack(func_out['outputs'], -1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', tables.NaturalNameWarning)
        for i, prefix in enumerate(table_prefixes):
            t_i = timesteps[i]
            lanes_i = list(map(__maybe_decode, lanes[i]))
            store.append(prefix + '/X', _df(X[i], lanes_i, t_i, x_colnames))
            store.append(prefix + '/Y', _df(Y[i], lanes_i, t_i, y_colnames))
            store.append(prefix + '/Yhat',
                         _df(Yhat[i], lanes_i, t_i, y_colnames))


def __sort_store_dfs(store, prefixes):
    for f in prefixes:
        for key in [f + t for t in ['/X', '/Y', '/Yhat']]:
            df = store[key]
            df = df[~df.index.duplicated(keep='first')] # deduplicate
            df = df.sort_index(level=[0,1])
            store[key] = df


def __maybe_decode(item):
    try:
        return item.decode()
    except AttributeError:
        return item


def _clean_str(s):
    return re.sub('[^0-9a-zA-Z_]', '_', s)
