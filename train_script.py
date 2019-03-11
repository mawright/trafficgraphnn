import logging
import os
import socket
import time

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.layers import (RNN, Dense, Dropout, GRUCell, Input, InputSpec,
                          Lambda, TimeDistributed)
from keras.models import Model
from trafficgraphnn.custom_fit_loop import (fit_loop_init, fit_loop_tf,
                                            make_callbacks,
                                            set_callback_params)
from trafficgraphnn.layers import ReshapeFoldInLanes, ReshapeUnfoldLanes
from trafficgraphnn.load_data_tf import TFBatcher
from trafficgraphnn.losses import (negative_masked_mae, negative_masked_mape,
                                   negative_masked_mse)
from trafficgraphnn.nn_modules import (gat_encoder, gat_single_A_encoder,
                                       output_tensor_slices, rnn_attn_decode,
                                       rnn_encode)

_logger = logging.getLogger(__name__)

def main(
    net_prefix='testnet3x3_v2',
    A_name_list=['A_downstream'],
    batch_size=4,
    time_window=150,
    average_interval=None,
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
    directories = [dir_string.format(net_prefix, i) for i in range(4)]
    train_dirs = directories[1:]
    val_dir = directories[1]
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

    batch_gen = TFBatcher(train_dirs, batch_size,
                          time_window,
                        #   time_window*average_interval,
                        #   average_interval=average_interval,
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

    X = gat_single_A_encoder(X_in, A_in, attn_depth, attn_dim, attn_heads,
                             dropout_rate, attn_dropout, 'relu')

    predense = TimeDistributed(Dropout(dropout_rate))(X)

    dense1 = TimeDistributed(Dense(dense_dim, activation='relu'))(predense)

    reshaped_1 = ReshapeFoldInLanes()(dense1)

    encoded = rnn_encode(reshaped_1, [rnn_dim, rnn_dim], 'GRU', True)

    decoded = rnn_attn_decode('GRU', rnn_dim, encoded, True)

    reshaped_decoded = ReshapeUnfoldLanes(num_lanes)(decoded)
    output = TimeDistributed(
        Dense(len(y_feature_subset), activation='linear'))(reshaped_decoded)

    outputs = output_tensor_slices(output, y_feature_subset)

    model = Model([X_in, A_in], outputs)

    Ytens = batch_gen.Y_slices

    model.compile(optimizer='Adam',
                  loss=negative_masked_mse,
                  metrics=[negative_masked_mae, negative_masked_mape],
                  target_tensors=Ytens,
                  )

    verbose = 1
    do_validation = True

    callback_list = make_callbacks(model, write_dir, do_validation)

    steps = batch_gen.num_batches * (5000 // time_window)
    set_callback_params(callback_list, epochs, batch_size, verbose,
                        do_validation, model, steps)

    fit_loop_init(model, callback_list)

    with K.get_session().as_default():
        # sess.graph.finalize()

        fit_loop_tf(model, callback_list, batch_gen, epochs)

        if hasattr(model, 'history'):
            print(model.history)

if __name__ == '__main__':
    main()
