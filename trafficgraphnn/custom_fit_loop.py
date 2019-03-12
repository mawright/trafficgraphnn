import datetime
import logging
import os
import time

import numpy as np
import tensorflow as tf

import keras.backend as K
from keras.callbacks import (BaseLogger, CallbackList, CSVLogger, History,
                             ModelCheckpoint, ProgbarLogger, ReduceLROnPlateau,
                             TensorBoard, TerminateOnNaN)

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

    # set up bookkeeping
    batch_size = batch_generator.batch_size * batch_generator.window_size
    i_step = 0
    for i_batch, batch in enumerate(batch_generator.train_batches):
        t0 = time.time()
        model.reset_states()
        batch.initialize(K.get_session(), feed_dict)

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
        batch.initialize(K.get_session(), feed_dict)

        while True:
            try:
                tstep = time.time()

                callbacks.on_test_batch_begin(i_step)
                logs = model.test_on_batch(x=None, y=None)
                train_step_time = time.time() - tstep

                logs = val_named_logs(model, logs)
                logs['size'] = batch_size
                logs['batch'] = i_step
                logs['time'] = train_step_time
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


def predict_eval_function(model):
    inputs = (model._feed_inputs
              + model._feed_targets
              + model._feed_sample_weights)
    if model._uses_dynamic_learning_phase():
        inputs += [K.learning_phase()]

    outputs = model.outputs + [model.total_loss] + model.metrics_tensors
    # Gets network outputs. Does not update weights.
    # Does update the network states.
    kwargs = getattr(model, '_function_kwargs', {})
    predict_eval_function = K.function(
        inputs,
        outputs,
        updates=model.state_updates + model.metrics_updates,
        name='predict_eval_function',
        **kwargs)
    model.predict_eval_function = predict_eval_function
    return predict_eval_function

