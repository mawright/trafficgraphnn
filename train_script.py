import argparse
import json
import logging
import math
import os

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.layers import Dense, Dropout, Input, TimeDistributed
from keras.models import Model
from keras.utils.multi_gpu_utils import multi_gpu_model
from trafficgraphnn import SumoNetwork
from trafficgraphnn.custom_fit_loop import (fit_loop_init, fit_loop_tf,
                                            get_logging_dir, make_callbacks,
                                            predict_eval_tf,
                                            set_callback_params)
from trafficgraphnn.layers import ReshapeFoldInLanes, ReshapeUnfoldLanes
from trafficgraphnn.load_data_tf import TFBatcher
from trafficgraphnn.losses import (huber, negative_masked_huber,
                                   negative_masked_mae, negative_masked_mape,
                                   negative_masked_mse)
from trafficgraphnn.nn_modules import (gat_encoder, output_tensor_slices,
                                       rnn_attn_decode, rnn_encode)
from trafficgraphnn.utils import iterfy

_logger = logging.getLogger(__name__)

def main(
    net_name,
    A_name_list=['A_downstream',
                 'A_upstream',
                 'A_neighbors'],
    val_split_proportion=.2,
    loss_function='mse',
    batch_size=4,
    time_window=150,
    average_interval=None,
    epochs=50,
    attn_dim=64,
    attn_heads=4,
    attn_depth=2,
    attn_residual_connection=False,
    rnn_dim=64,
    stateful_rnn=False,
    dense_dim=64,
    dropout_rate=.3,
    attn_dropout=0.,
    seed=123,
    per_step_metrics=False,
    num_gpus=1,
):

    tf.set_random_seed(seed)
    np.random.seed(seed)

    net_dir = os.path.join('data', 'networks', net_name)

    sn = SumoNetwork.from_preexisting_directory(net_dir)
    lanes = sn.lanes_with_detectors()
    num_lanes = len(lanes)

    data_dir = os.path.join(net_dir, 'preprocessed_data')

    x_feature_subset = ['e1_0/occupancy',
                        'e1_0/speed',
                        'e1_1/occupancy',
                        'e1_1/speed',
                        'liu_estimated_veh',
                        'green']
    y_feature_subset = ['e2_0/nVehSeen',
                        'e2_0/maxJamLengthInVehicles']

    write_dir = os.path.join(net_dir, 'models')
    if not os.path.isdir(write_dir):
        os.makedirs(write_dir)
    with tf.device('/cpu:0'):
        batch_gen = TFBatcher(data_dir,
                              batch_size,
                              time_window,
                              average_interval=average_interval,
                              val_proportion=val_split_proportion,
                              shuffle=True,
                              A_name_list=A_name_list,
                              x_feature_subset=x_feature_subset,
                              y_feature_subset=y_feature_subset,
                              )

        Xtens = batch_gen.X
        Atens = tf.cast(batch_gen.A, tf.float32)

    # X dimensions: timesteps x lanes x feature dim
    X_in = Input(batch_shape=(None, None, num_lanes, len(x_feature_subset)),
                 name='X', tensor=Xtens)
    # A dimensions: timesteps x lanes x lanes
    A_in = Input(batch_shape=(None, None, len(A_name_list),
                              num_lanes, num_lanes),
                 name='A', tensor=Atens)

    attn_dim = iterfy(attn_dim) * attn_depth
    attn_heads = iterfy(attn_heads) * attn_depth

    def make_model(X_in, A_in):
        X = gat_encoder(X_in, A_in, attn_dim, attn_heads,
                        dropout_rate, attn_dropout, gat_activation='relu',
                        residual_connection=attn_residual_connection)

        predense = TimeDistributed(Dropout(dropout_rate))(X)

        dense1 = TimeDistributed(Dense(dense_dim, activation='relu'))(predense)

        if stateful_rnn:
            reshape_batch_size = batch_size
        else:
            reshape_batch_size = None
        reshaped_1 = ReshapeFoldInLanes(batch_size=reshape_batch_size)(dense1)

        encoded = rnn_encode(reshaped_1, [rnn_dim], 'GRU',
                             stateful=stateful_rnn)

        decoded = rnn_attn_decode('GRU', rnn_dim, encoded,
                                  stateful=stateful_rnn)

        reshaped_decoded = ReshapeUnfoldLanes(num_lanes)(decoded)
        output = TimeDistributed(
            Dense(len(y_feature_subset), activation='relu'))(reshaped_decoded)

        outputs = output_tensor_slices(output, y_feature_subset)

        model = Model([X_in, A_in], outputs)
        return model

    if num_gpus > 1:
        with tf.device('/cpu:0'):
            base_model = make_model(X_in, A_in)
            model = multi_gpu_model(base_model, num_gpus)
    else:
        base_model = make_model(X_in, A_in)
        model = base_model

    Ytens = batch_gen.Y_slices

    if loss_function.lower() == 'mse':
        losses = ['mse', negative_masked_mse]
        metrics = [negative_masked_mae, negative_masked_huber,
                   negative_masked_mape]
    elif loss_function.lower() == 'mae':
        losses = ['mae', negative_masked_mae]
        metrics = [negative_masked_mse, negative_masked_huber,
                   negative_masked_mape]
    elif loss_function.lower() == 'huber':
        losses = [huber, negative_masked_huber]
        metrics = [negative_masked_mse, negative_masked_mae,
                   negative_masked_mape]

    model.compile(optimizer='Adam',
                  loss=losses,
                  metrics=metrics,
                  target_tensors=Ytens,
                  )

    verbose = 1
    if val_split_proportion > 0:
        do_validation = True
    else:
        do_validation = False

    callback_list = make_callbacks(model, write_dir, do_validation, base_model)

    # record hyperparameters
    hyperparams = dict(
        net_name=net_name, A_name_list=A_name_list,
        x_feature_subset=x_feature_subset, y_feature_subset=y_feature_subset,
        val_split_proportion=val_split_proportion,
        loss_function=loss_function, batch_size=batch_size,
        time_window=time_window, average_interval=average_interval,
        epochs=epochs, attn_dim=attn_dim, attn_depth=attn_depth,
        attn_residual_connection=attn_residual_connection,
        attn_heads=attn_heads, rnn_dim=rnn_dim, stateful_rnn=stateful_rnn,
        dense_dim=dense_dim, dropout_rate=dropout_rate,
        attn_dropout=attn_dropout, seed=seed, num_gpus=num_gpus)
    logdir = get_logging_dir(callback_list)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    with open(os.path.join(logdir, 'params.json'), 'w') as f:
        json.dump(hyperparams,f)
    _logger.info('Run dir: %s', logdir)

    # Guess at the number of steps per simulation. This only affects Keras's
    # progress bar per training epoch so it can be wrong.
    if per_step_metrics:
        timesteps_per_simulation = 3600
        steps = batch_gen.num_train_batches * math.ceil(timesteps_per_simulation
                                                        / time_window)
    else:
        steps = batch_gen.num_train_batches

    set_callback_params(callback_list, epochs, batch_size, verbose,
                        do_validation, model, steps)

    fit_loop_init(model, callback_list, batch_gen)

    with K.get_session().as_default():

        fit_loop_tf(model, callback_list, batch_gen, epochs,
                    per_step_metrics=per_step_metrics)

        predict_eval_tf(model, write_dir, batch_gen)

        if hasattr(model, 'history'):
            return model.history #pylint: disable=no-member

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('net_name', type=str, help='Name of Sumo Network')
    parser.add_argument('--A_downstream', '-Adown', action='store_true',
                        help='Use the downstream-lane adjacency matrix.')
    parser.add_argument('--A_upstream', '-Aup', action='store_true',
                        help='Use the downstream-lane adjacency matrix.')
    parser.add_argument('--A_neighbors', '-Aneigh', action='store_true',
                        help='Use the neighboring-lane adjacency matrix.')
    parser.add_argument('--val_split', '-v', type=float, default=.2,
                        help='Data proportion to use for validation')
    parser.add_argument('--loss_function', '-l', type=str, default='mse',
                        help='Training loss function. '
                             'Valid: "mse", "mae", "huber". Default: mse')
    parser.add_argument('--batch_size', '-b', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--time_window', '-tw', type=int, default=150,
                        help='Subsequence time window (s)')
    parser.add_argument('--average_interval', '-a', type=int,
                        help='Hard averaging downsampling interval (s)')
    parser.add_argument('--epochs', '-e', type=int, default=30,
                        help='Number of training epochs.')
    parser.add_argument('--attn_dim', type=int, default=64,
                        help='Dimensionality of attentional embeddings')
    parser.add_argument('--attn_heads', type=int, default=4,
                        help='Number of attention heads per layer')
    parser.add_argument('--attn_depth', type=int, default=2,
                        help='Number of stacked attentional layers')
    parser.add_argument('--attn_residual_connection', action='store_true',
                        help='Use residual connections in the attenion encoders.')
    parser.add_argument('--dense_dim', type=int, default=64,
                        help='Dimensionality of FC layers after attention ones')
    parser.add_argument('--rnn_dim', type=int, default=64,
                        help='Dimensionality of per-lane RNN embedding')
    parser.add_argument('--stateful_rnn', action='store_true',
                        help='Set to use stateful RNNs.')
    parser.add_argument('--dropout_rate', type=float, default=.3,
                        help='Inter-layer dropout probability')
    parser.add_argument('--attn_dropout', type=float, default=0.,
                        help='Probability of dropout on attention weights')
    parser.add_argument('--seed', '-s', type=int, help='Random seed',
                        default=123)
    parser.add_argument('--per_step_metrics', action='store_true',
                        help='Set to record metrics per gradient step '
                             'instead of averaged over each simulation batch.')
    parser.add_argument('--num_gpus', '-g', type=int, default=1,
                        help='Number of GPUs to use.')
    args = parser.parse_args()

    A_name_list = []
    if args.A_downstream:
        A_name_list.append('A_downstream')
    if args.A_upstream:
        A_name_list.append('A_upstream')
    if args.A_neighbors:
        A_name_list.append('A_neighbors')

    main(args.net_name,
         A_name_list,
         val_split_proportion=args.val_split,
         loss_function=args.loss_function,
         batch_size=args.batch_size,
         time_window=args.time_window,
         average_interval=args.average_interval,
         epochs=args.epochs,
         attn_dim=args.attn_dim,
         attn_heads=args.attn_heads,
         attn_depth=args.attn_depth,
         attn_residual_connection=args.attn_residual_connection,
         dense_dim=args.dense_dim,
         rnn_dim=args.rnn_dim,
         stateful_rnn=args.stateful_rnn,
         dropout_rate=args.dropout_rate,
         attn_dropout=args.attn_dropout,
         seed=args.seed,
         per_step_metrics=args.per_step_metrics,
         num_gpus=args.num_gpus,
         )
