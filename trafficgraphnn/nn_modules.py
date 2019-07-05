import keras.backend as K
from keras.layers import (RNN, Dense, Dropout, GRUCell, InputSpec, Lambda,
                          LSTMCell, TimeDistributed)
from trafficgraphnn.layers import (BatchGraphAttention,
                                   BatchMultigraphAttention,
                                   DenseCausalAttention, LayerNormalization,
                                   TimeDistributedMultiInput)
from trafficgraphnn.layers.modified_thirdparty import BatchGraphConvolution
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


def gcn_encoder(X_tensor, A_tensor, filter_type, filter_dims, dropout_rate,
                dense_dims, cheb_polynomial_degree=2, layer_norm=False,
                activation='relu'):
    import tensorflow as tf
    from kegra.utils import chebyshev_polynomial, normalized_laplacian, \
                            rescale_laplacian
    from scipy import sparse
    if len(A_tensor.shape) > 4: # flatten out edge type dimension
        A_tensor = Lambda(lambda A: K.max(A, 2, keepdims=False),
                          name='flatten_A')(A_tensor)

    if filter_type == 'localpool':
        """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
        print('Using local pooling filters...')
        def func(A):
            A = tf.maximum(A, tf.matrix_transpose(A)) # symmetrize
            diag = tf.matrix_diag_part(A)
            A = tf.matrix_set_diag(A, diag + 1)
            degree = tf.reduce_sum(A, -1)
            degree_invsqrt = tf.rsqrt(degree)
            D_invsqrt = tf.matrix_diag(degree_invsqrt)
            return tf.matmul(tf.matmul(D_invsqrt, A), D_invsqrt)
        output_shape = K.int_shape(A_tensor)[1:]

    elif filter_type == 'chebyshev':
        SYM_NORM = True
        """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
        print('Using Chebyshev polynomial basis filters...')
        def gcn_preprocess(A):
            A = sparse.lil_matrix(A)
            A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A) # symmetrize
            L = normalized_laplacian(A, SYM_NORM)
            L_scaled = rescale_laplacian(L)
            cheb = chebyshev_polynomial(L_scaled, cheb_polynomial_degree)
            return [c.todense().A.astype('float32') for c in cheb]
        support = cheb_polynomial_degree + 1
        return_type = [tf.float32] * support
        output_shape = lambda x: [x] * support
        # G = [tf.py_func(gcn_preprocess, [A_tensor], return_type,
        #                 stateful=False)]
        def func(A):
            shape = tf.shape(A)
            A = tf.reshape(A, tf.concat((tf.reduce_prod(shape[:2],
                                                        keepdims=True),
                                        shape[2:]), axis=0))
            G = tf.map_fn(
                lambda a: tf.py_func(gcn_preprocess, [a], return_type,
                                    stateful=False), A, dtype=return_type)
            G = [tf.reshape(g, shape) for g in G]
            return [tf.ensure_shape(g, A_tensor.shape) for g in G]
    else:
        raise Exception('Invalid filter type.')

    l = Lambda(
        lambda A: func(A),
        # gcn_preprocess,
        output_shape=output_shape,
        name='gcn_preproc')
    G = l(A_tensor)
    if K.is_tensor(G):
        G = [G]

    X = X_tensor
    for i, gc_units in enumerate(filter_dims):
        X = TimeDistributed(
            Dropout(dropout_rate), name='dropout_{}'.format(i))(X)
        X = TimeDistributedMultiInput(
            BatchGraphConvolution(gc_units, support=support, activation=activation,
                                  name='GC_{}'.format(i)))([X]+G)
        if layer_norm:
            X = LayerNormalization(name='GC_layernorm_{}'.format(i))(X)
        if dense_dims is not None:
            X = TimeDistributed(Dense(dense_dims, activation=activation),
                                name='FC_{}'.format(i))(X)
            if layer_norm:
                X = LayerNormalization(name='FC_layernorm_{}'.format(i))(X)
    return X


def gat_encoder(X_tensor, A_tensor, attn_dims, num_heads,
                dropout_rate, attn_dropout_rate, attn_reduction='concat',
                gat_highway_connection=True,
                layer_norm=False,
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
    for i, (dim, head, drop, attndrop, reduct) in enumerate(
        zip(attn_dims, num_heads,
            dropout_rate,
            attn_dropout_rate,
            attn_reduction)):
        out = TimeDistributed(Dropout(drop), name='dropout_{}'.format(i))(X)
        out = TimeDistributedMultiInput(
            BatchMultigraphAttention(dim,
                                     attn_heads=head,
                                     attn_heads_reduction=reduct,
                                     attn_dropout=attndrop,
                                     activation=gat_activation,
                                     highway_connection=gat_highway_connection
                                     ), name='GAT_{}'.format(i))([X, A_tensor])
        if residual_connection: # in transformer, res connection done here (eg on concatted heads)
            X = X + out
        else:
            X = out
        if layer_norm:
            X = LayerNormalization(name='GAT_layernorm_{}'.format(i))(X)
        if dense_dim is not None:
            out = TimeDistributed(Dense(dense_dim, activation=gat_activation),
                                  name='FC_{}'.format(i))(X)
            if residual_connection:
                X = X + out
            else:
                X = out
            if layer_norm:
                X = LayerNormalization(name='FC_layernorm_{}'.format(i))(X)
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
