import keras.backend as K
from keras.layers import (RNN, Dense, Dropout, GRUCell, InputSpec, LSTMCell,
                          TimeDistributed, Lambda)
from trafficgraphnn.layers import (BatchGraphAttention,
                                   BatchMultigraphAttention,
                                   DenseCausalAttention,
                                   TimeDistributedMultiInput)
from trafficgraphnn.utils import broadcast_lists, iterfy


def gat_single_A_encoder(X_tensor, A_tensor, attn_depth, attn_dims, num_heads,
                         dropout_rate, attn_dropout_rate,
                         gat_activation='relu'):
    attn_dims, num_heads, dropout_rate, attn_dropout_rate = map(
        iterfy, [attn_dims, num_heads, dropout_rate, attn_dropout_rate])

    assert len(attn_dims) == len(num_heads)
    attn_dims, num_heads, dropout_rate, attn_dropout_rate = broadcast_lists(
        [attn_dims, num_heads, dropout_rate, attn_dropout_rate])

    X = X_tensor
    # if A is 5-dimensional (batch, time, edgetype, lane, lane), squeeze out
    # the edge type dim
    if len(A_tensor.shape) > 4:
        A_tensor = K.squeeze(A_tensor, -3)
    for dim, head, drop, attndrop in zip(attn_dims, num_heads, dropout_rate,
                                         attn_dropout_rate):
        X = TimeDistributed(Dropout(drop))(X)
        X = TimeDistributedMultiInput(
            BatchGraphAttention(dim,
                                attn_heads=head,
                                attn_dropout=attndrop,
                                activation=gat_activation))([X, A_tensor])
    return X


def gat_encoder(X_tensor, A_tensor, attn_dims, num_heads,
                dropout_rate, attn_dropout_rate, attn_reduction='concat',
                gat_activation='relu', dense_dim=None,
                residual_connection=False):
    attn_dims, num_heads, dropout_rate, attn_dropout_rate, attn_reduction = map(
        iterfy, [attn_dims, num_heads, dropout_rate, attn_dropout_rate,
                 attn_reduction])

    assert len(attn_dims) == len(num_heads)
    attn_dims, num_heads, dropout_rate, attn_dropout_rate, attn_reduction \
        = broadcast_lists([attn_dims, num_heads, dropout_rate,
                           attn_dropout_rate, attn_reduction])

    X = X_tensor
    for dim, head, drop, attndrop, reduct in zip(attn_dims, num_heads,
                                                 dropout_rate,
                                                 attn_dropout_rate,
                                                 attn_reduction):
        out = TimeDistributed(Dropout(drop))(X)
        out = TimeDistributedMultiInput(
            BatchMultigraphAttention(dim,
                                     attn_heads=head,
                                     attn_heads_reduction=reduct,
                                     attn_dropout=attndrop,
                                     activation=gat_activation))([X, A_tensor])
        if residual_connection: # in transformer, res connection done here (eg on concatted heads)
            X = X + out
        else:
            X = out
        if dense_dim is not None:
            out = TimeDistributed(Dense(dense_dim, activation=gat_activation))(X)
            if residual_connection:
                X = X + out
            else:
                X = out
    return X


def rnn_encode(input_tensor, rnn_dims, cell_type, stateful=True):
    rnn_dims = iterfy(rnn_dims)

    cell_fn = _get_cell_fn(cell_type)

    cells = []
    for dim in rnn_dims:
        cells.append(cell_fn(dim))

    encoder = RNN(cells, return_sequences=True, stateful=stateful,
                  name='rnn_encoder')
    return encoder(input_tensor)


def rnn_attn_decode(cell_type, rnn_dim, encoded_seq, stateful=True):
    cell_fn = _get_cell_fn(cell_type)

    decoder_cell = DenseCausalAttention(cell=cell_fn(rnn_dim))
    decoder = RNN(
        cell=decoder_cell, return_sequences=True, stateful=stateful,
        name='attn_decoder')
    decoder.input_spec = [InputSpec(shape=K.int_shape(encoded_seq))]

    tdd = TimeDistributed(Dense(rnn_dim, use_bias=False))
    u = tdd(encoded_seq)
    return decoder(encoded_seq, constants=[encoded_seq, u])


def output_tensor_slices(output_tensor, feature_names):
    outputs = []
    for i, feature in enumerate(feature_names):
        out = Lambda(lambda x: x[...,i], name=feature)(output_tensor)
        outputs.append(out)
    return outputs


def _get_cell_fn(cell_type):
    if cell_type.upper() == 'GRU':
        return GRUCell
    elif cell_type.upper() == 'LSTM':
        return LSTMCell
    else:
        raise ValueError("Param 'cell_type' must be 'LSTM' or 'GRU', "
                         "got {}".format(cell_type))
