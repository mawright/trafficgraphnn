import argparse
import logging
import os

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.layers import Dense, Dropout, Input, TimeDistributed
from keras.models import Model
from trafficgraphnn import SumoNetwork
from trafficgraphnn.custom_fit_loop import (fit_loop_init, fit_loop_tf,
                                            make_callbacks,
                                            set_callback_params)
from trafficgraphnn.layers import ReshapeFoldInLanes, ReshapeUnfoldLanes
from trafficgraphnn.load_data_tf import TFBatcher
from trafficgraphnn.losses import (negative_masked_mae, negative_masked_mape,
                                   negative_masked_mse)
from trafficgraphnn.nn_modules import (gat_encoder, output_tensor_slices,
                                       rnn_attn_decode, rnn_encode)

_logger = logging.getLogger(__name__)

def main(
    net_name,
    A_name_list=['A_downstream',
                 'A_upstream',
                 'A_neighbors'],
    val_split_proportion=.2,
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
    attn_dropout=0.,
    seed=123,
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

    batch_gen.init_initializable_iterator()

    Xtens = batch_gen.X
    Atens = tf.cast(batch_gen.A, tf.float32)

    # X dimensions: timesteps x lanes x feature dim
    X_in = Input(batch_shape=(batch_size, None, num_lanes, len(x_feature_subset)),
                name='X', tensor=Xtens)
    # A dimensions: timesteps x lanes x lanes
    A_in = Input(batch_shape=(batch_size, None, None, num_lanes, num_lanes),
                name='A', tensor=Atens)

    X = gat_encoder(X_in, A_in, attn_depth, attn_dim, attn_heads,
                    dropout_rate, attn_dropout, 'relu')

    predense = TimeDistributed(Dropout(dropout_rate))(X)

    dense1 = TimeDistributed(Dense(dense_dim, activation='relu'))(predense)

    reshaped_1 = ReshapeFoldInLanes()(dense1)

    encoded = rnn_encode(reshaped_1, [rnn_dim], 'GRU', True)

    decoded = rnn_attn_decode('GRU', rnn_dim, encoded, True)

    reshaped_decoded = ReshapeUnfoldLanes(num_lanes)(decoded)
    output = TimeDistributed(
        Dense(len(y_feature_subset), activation='relu'))(reshaped_decoded)

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

    steps = batch_gen.num_batches * (2000 // time_window)
    set_callback_params(callback_list, epochs, batch_size, verbose,
                        do_validation, model, steps)

    fit_loop_init(model, callback_list)

    with K.get_session().as_default():
        # sess.graph.finalize()

        fit_loop_tf(model, callback_list, batch_gen, epochs)

        if hasattr(model, 'history'):
            print(model.history)

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
    parser.add_argument('--batch_size', '-b', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--time_window', '-tw', type=int, default=150,
                        help='Subsequence time window (s)')
    parser.add_argument('--average_interval', '-a', type=int,
                        help='Hard averaging downsampling interval (s)')
    parser.add_argument('--epochs', '-e', type=int,
                        help='Number of training epochs.')
    parser.add_argument('--attn_dim', type=int, default=64,
                        help='Dimensionality of attentional embeddings')
    parser.add_argument('--attn_heads', type=int, default=4,
                        help='Number of attention heads per layer')
    parser.add_argument('--attn_depth', type=int, default=2,
                        help='Number of stacked attentional layers')
    parser.add_argument('--dense_dim', type=int, default=64,
                        help='Dimensionality of FC layers after attention ones')
    parser.add_argument('--rnn_dim', type=int, default=64,
                        help='Dimensionality of per-lane RNN embedding')
    parser.add_argument('--dropout_rate', type=float, default=.3,
                        help='Inter-layer dropout probability')
    parser.add_argument('--attn_dropout', type=float, default=0.,
                        help='Probability of dropout on attention weights')
    parser.add_argument('--seed', '-s', type=int, help='Random seed',
                        default=123)
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
         batch_size=args.batch_size,
         time_window=args.time_window,
         average_interval=args.average_interval,
         epochs=args.epochs,
         attn_dim=args.attn_dim,
         attn_heads=args.attn_heads,
         attn_depth=args.attn_depth,
         dense_dim=args.dense_dim,
         dropout_rate=args.dropout_rate,
         attn_dropout=args.attn_dropout,
         seed=args.seed,
         )
