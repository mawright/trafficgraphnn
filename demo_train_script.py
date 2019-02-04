import logging
import os
import socket
import time

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.callbacks import (BaseLogger, CallbackList, History,
                             ModelCheckpoint, ProgbarLogger, TensorBoard,
                             TerminateOnNaN, ReduceLROnPlateau)
from keras.layers import (RNN, Dense, Dropout, GRUCell, Input, InputSpec,
                          TimeDistributed, Lambda)
from keras.models import Model
from trafficgraphnn.layers import (BatchGraphAttention, ReshapeFoldInLanes,
                                   ReshapeUnfoldLanes,
                                   TimeDistributedMultiInput)
from trafficgraphnn.layers import DenseCausalAttention
from trafficgraphnn.load_data_tf import TFBatcher
from trafficgraphnn.losses import (mean_absolute_error_veh,
                                   mean_square_error_veh,
                                   negative_masked_mae_queue_length,
                                   negative_masked_mse, negative_masked_mae)


_logger = logging.getLogger(__name__)

def main(
    net_prefix='testnet3x3',
    A_name_list=['A_downstream'],
    batch_size=4,
    time_window=150,
    epochs=50,
    attn_dim=64,
    attn_heads=4,
    attn_depth=2,
    rnn_dim=64,
    dense_dim=64,
    dropout_rate=.3,
    attn_dropout=.3,
    seed=123,
):

    tf.set_random_seed(seed)
    np.random.seed(seed)

    dir_string = 'data/networks/{}_{}/preprocessed_data'
    directories = [dir_string.format(net_prefix, i) for i in range(5)]
    train_dirs = directories[:-2] + directories[-1:]
    val_dir = directories[-2]
    num_lanes = 120

    x_feature_subset = ['e1_0/occupancy',
                        'e1_0/speed',
                        'e1_1/occupancy',
                        'e1_1/speed',
                        'liu_estimated_m',
                        'liu_estimated_veh',
                        'green']
    y_feature_subset = ['e2_0/nVehSeen',
                        'e2_0/maxJamLengthInVehicles']

    write_dir = 'data/test'

    batch_gen = TFBatcher(train_dirs, batch_size, time_window,
                          A_name_list=A_name_list,
                          x_feature_subset=x_feature_subset,
                          y_feature_subset=y_feature_subset,
                          val_directories=val_dir,
                          shuffle=True, buffer_size=20)

    # batch_gen.init_feedable_iterator(sess)
    batch_gen.init_initializable_iterator()

    Xtens = batch_gen.X
    Atens = tf.cast(batch_gen.A, tf.float32)
    Atens = tf.squeeze(Atens, 2)
    Ashape = tf.shape(Atens)
    Atens = Atens + tf.eye(Ashape[-2], batch_shape=Ashape[:-2])

    # X dimensions: timesteps x lanes x feature dim
    X_in = Input(batch_shape=(batch_size, None, num_lanes, len(x_feature_subset)),
                name='X', tensor=Xtens)
    # A dimensions: timesteps x lanes x lanes
    A_in = Input(batch_shape=(batch_size, None, num_lanes, num_lanes),
                name='A', tensor=Atens)

    X = X_in
    for _ in range(attn_depth):
        X = TimeDistributed(Dropout(dropout_rate))(X)
        X = TimeDistributedMultiInput(
            BatchGraphAttention(attn_dim,
                                attn_heads=attn_heads,
                                activation='relu',
                                attn_dropout=attn_dropout))([X, A_in])

    predense = TimeDistributed(Dropout(dropout_rate))(X)

    dense1 = TimeDistributed(Dense(dense_dim, activation='relu'))(predense)

    reshaped_1 = ReshapeFoldInLanes()(dense1)

    cells = [GRUCell(rnn_dim), GRUCell(rnn_dim)]

    encoder = RNN(cells, return_sequences=True, stateful=True)
    encoded = encoder(reshaped_1)

    # decoder_cell = DenseAnnotationAttention(cell=GRUCell(rnn_dim), units=rnn_dim)
    decoder_cell = DenseCausalAttention(cell=GRUCell(rnn_dim))
    decoder = RNN(cell=decoder_cell, return_sequences=True, stateful=True)
    decoder.input_spec = [InputSpec(shape=K.int_shape(encoded))]
    u = TimeDistributed(Dense(rnn_dim, use_bias=False))(encoded)
    decoded = decoder(encoded, constants=[encoded, u])
    # decoded = decoder(encoded, constants=encoded)

    reshaped_decoded = ReshapeUnfoldLanes(num_lanes)(decoded)
    output = TimeDistributed(
        Dense(len(y_feature_subset), activation='linear'))(reshaped_decoded)

    # output1 = Lambda(lambda x: x[...,0], name='num_vehicles_on_lane')(output)
    # output2 = Lambda(lambda x: x[...,1], name='queue_length')(output)

    # model = Model([X_in, A_in], [output1, output2])
    model = Model([X_in, A_in], output)

    Ytens = batch_gen.Y
    # Ytens1 = Ytens[...,0]
    # Ytens2 = Ytens[...,1]

    feed_dict = dict()

    model.compile(optimizer='Adam',
                loss=negative_masked_mse,
                # loss=mean_square_error_veh,
                metrics=[mean_absolute_error_veh, negative_masked_mae_queue_length],
                # target_tensors=[Ytens1, Ytens2],
                target_tensors=Ytens,
                feed_dict=feed_dict
                )
    model.stop_training = False

    verbose = 1
    do_validation = True
    metrics = ['loss', 'mean_absolute_error_veh',
            'negative_masked_mae_queue_length']
    stateful_metrics = [
            'val_loss', 'val_mean_absolute_error_veh',
            'val_negative_masked_mae_queue_length']

    callback_list = CallbackList()
    history = History()
    callback_list.append(BaseLogger())
    callback_list.append(history)
    callback_list.append(TensorBoard())
    callback_list.append(ProgbarLogger('steps', stateful_metrics=stateful_metrics))
    filename = ('weights_epoch{epoch:02d}-'
            'val_mae{val_mean_absolute_error_veh:.4f}-'
            'val_queue_mae{val_negative_masked_mae_queue_length:.4f}'
            '.hdf5')
    callback_list.append(ModelCheckpoint(os.path.join(write_dir, filename)))
    callback_list.append(TerminateOnNaN())
    callback_list.append(ReduceLROnPlateau(verbose=1))

    callback_list.set_model(model)
    callback_list.set_params({
        'batch_size': batch_size,
        'epochs': epochs,
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': metrics + stateful_metrics,
        'steps': batch_gen.num_batches * (5000 // time_window)
    })

    callback_list.on_train_begin()

    model.reset_states()
    model._make_train_function()
    model._make_test_function()

    with K.get_session().as_default() as sess:
        sess.graph.finalize()

        for epoch in range(epochs):
            callback_list.on_epoch_begin(epoch)
            _logger.info('beginning epoch %g', epoch)
            t_epoch_start = time.time()
            # sess.run(batch_gen.train_batches[0].initializer)
            i_step = 0
            t0 = time.time()
            batch_iter = batch_gen.train_batches
            for i_batch, batch in enumerate(batch_iter):
                loss_list = []
                veh_mae_list = []
                queue_length_mae_list = []
                step_times = []
                # for out in batch.iterate():

                model.reset_states()

                batch.initialize(sess, feed_dict)

                # _logger.debug('Beginning training on batch %g', i_batch)
                try:
                    # feed_dict[batch_gen.handle] = batch.handle
                    while True:
                        tstep = time.time()

                        callback_list.on_batch_begin(i_step)

                        logs = model.train_on_batch(x=None, y=None)
                        train_step_time = time.time() - tstep
                        step_times.append(train_step_time)
                        # _logger.info('train step took %s s', time.time() - t2)

                        # loss_list.append(logs[0])
                        # veh_mae_list.append(logs[1])
                        # if not np.isnan(logs[2]):
                        #     queue_length_mae_list.append(logs[2])
                        logs = named_logs(model, logs)
                        # print(logs)
                        logs['size'] = batch_size * time_window
                        logs['batch'] = i_step
                        logs['time'] = train_step_time
                        loss_list.append(logs['loss'])
                        veh_mae_list.append(
                            logs['mean_absolute_error_veh'])
                        queue_length_mae_list.append(
                            logs['negative_masked_mae_queue_length'])

                        callback_list.on_batch_end(i_step, logs)
                        i_step += 1
                except tf.errors.OutOfRangeError:
                    # print('')
                    _logger.debug('batch %g losses: MSE = %s, veh MAE = %s, '
                                'q len MAE = %s (%s s)',
                                i_batch, np.mean(loss_list), np.mean(veh_mae_list),
                                np.mean(queue_length_mae_list), time.time() - t0)
                    t0 = time.time()
                finally:
                    if model.stop_training:
                        break

            t_epoch_end = time.time()

            # _logger.info('Doing validation')

            val_loss_list = []
            val_veh_mae_list = []
            val_queue_length_mae_list = []
            val_batch_size_list = []
            batch_iter = batch_gen.val_batches
            for i_batch, batch in enumerate(batch_iter):
                t0 = time.time()
                batch_loss_list = []
                batch_veh_mae_list = []
                batch_queue_length_mae_list = []
                batch_size_list = []

                batch.initialize(sess, feed_dict)

                try:
                    # feed_dict[batch_gen.handle] = batch.handle
                    while True:
                        val_loss, val_veh_mae, val_queue_mae = model.test_on_batch(x=None, y=None)
                        batch_loss_list.append(val_loss)
                        batch_veh_mae_list.append(val_veh_mae)
                        batch_queue_length_mae_list.append(val_queue_mae)
                        batch_size_list.append(batch_size * time_window)
                except tf.errors.OutOfRangeError:

                    sum_batch_sizes = np.sum(np.array(batch_size_list))
                    batch_val_loss = np.nansum(np.array(batch_loss_list) * np.array(batch_size_list) / sum_batch_sizes)
                    batch_veh_mae = np.nansum(np.array(batch_veh_mae_list) * np.array(batch_size_list) / sum_batch_sizes)
                    batch_queue_length_mae = np.nansum(
                        np.array(batch_queue_length_mae_list) * np.array(batch_size_list) / sum_batch_sizes)

                    val_loss_list.append(batch_val_loss)
                    val_veh_mae_list.append(batch_veh_mae)
                    val_queue_length_mae_list.append(batch_queue_length_mae)
                    val_batch_size_list.append(sum_batch_sizes)

                    # print('')
                    _logger.debug('batch %g val losses: MSE = %s, '
                                'veh MAE = %s, q len MAE = %s (%s s)',
                                i_batch, batch_val_loss, batch_veh_mae,
                                batch_queue_length_mae, time.time() - t0)
                    model.reset_states()
                    i_batch += 1

            sum_samples_epoch = np.sum(np.array(val_batch_size_list))

            epoch_logs = {
                'val_loss': np.sum(np.array(val_loss_list) * np.array(val_batch_size_list)) / np.sum(sum_samples_epoch),
                'val_mean_absolute_error_veh': np.sum(
                    np.array(val_veh_mae_list) * np.array(val_batch_size_list)) / sum_samples_epoch,
                'val_negative_masked_mae_queue_length': np.sum(
                    np.array(val_queue_length_mae_list) * np.array(val_batch_size_list)) / sum_samples_epoch
                        }
            callback_list.on_epoch_end(epoch, epoch_logs)
            _logger.info('Time to run epoch = %s s', time.time() - t_epoch_start)
            _logger.debug(epoch_logs)
            # _logger.debug(history.history)
        callback_list.on_train_end(logs)


def named_logs(model, logs):
    result = {}
    for l in zip(model.metrics_names, logs):
        result[l[0]] = l[1]
    return result


if __name__ == '__main__':
    main()
