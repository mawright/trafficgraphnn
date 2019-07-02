"""A wrapper for RNN cells that provide encoder-decoderneural attention
(Bahdanau-style additive), with causality such that the output sequence at
time t may not attend to timesteps in the input sequence before time t.
"""
from __future__ import print_function

from keras import backend as K
from keras import constraints, initializers, regularizers
from trafficgraphnn.layers.modified_thirdparty import AttentionCellWrapper


class DenseCausalAttention(AttentionCellWrapper):
    """Recurrent attention mechanism for attending sequences, with causality enforced.

    This class implements the attention mechanism used in [1] for machine
    translation. It is, however, a generic sequence attention mechanism that can be
    used for other sequence-to-sequence problems.

    This class differs from ``DenseAnnotationAttention'' in that the output
    sequence's potential attention targets in the input sequence are limited to
    those at the same timestep or before. This is done by masking out those
    timsteps' attention weights (they are set to -10e10)

    As any recurrent attention mechanism extending `_RNNAttentionCell`, this class
    should be used in conjunction with a wrapped (non attentive) RNN Cell, such as
    the `SimpleRNNCell`, `LSTMCell` or `GRUCell`. It modifies the input of the
    wrapped cell by attending to a constant sequence (i.e. independent of the time
    step of the recurrent application of the attention mechanism). The attention
    encoding is obtained by computing a scalar weight for each time step of the
    attended by applying two stacked Dense layers to the concatenation of the
    attended feature vector at the respective time step with the previous state of
    the RNN Cell. The attention encoding is the weighted sum of the attended feature
    vectors using these weights.

    Half of the first Dense transformation is independent of the RNN Cell state and
    can be computed once for the attended sequence. Therefore this transformation
    should be computed externally of the attentive RNN Cell (for efficiency) and this
    layer expects both the attended sequence and the output of a Dense transformation
    of the attended sequence (see Example below). The number of hidden units of the
    attention mechanism is subsequently defined by the number of units of this
    (external) dense transformation.

    # Arguments
        cell: A RNN cell instance. The wrapped RNN cell wrapped by this attention
            mechanism. See docs of `cell` argument in the `RNN` Layer for further
            details.
        kernel_initializer: Initializer for all weights matrices
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for all bias vectors
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            all weights matrices. (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to all biases
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            all weights matrices. (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to all bias vectors
            (see [constraints](../constraints.md)).

    # References
    [1] Neural Machine Translation by Jointly Learning to Align and Translate
        https://arxiv.org/abs/1409.0473
    """
    def __init__(self, cell,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DenseCausalAttention, self).__init__(cell, **kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def attention_call(self,
                       inputs,
                       cell_states,
                       attended,
                       attention_states,
                       attended_mask,
                       training=None):
        [attended, u] = attended
        # attended: hidden states of encoder (h)
        # u: dot product of encoder state and attention weight matrix Ua
        attended_mask = attended_mask[0]
        h_cell_tm1 = cell_states[0]
        tm1 = attention_states[1]

        # TODO move this definition to the "constants" part
        attended_shape = K.shape(attended)
        length = attended_shape[1]

        timesteps = K.arange(length)
        timesteps = K.expand_dims(timesteps, 0)

        causal_mask = K.less_equal(timesteps, K.cast(tm1, 'int32'))
        if attended_mask is None:
            attended_mask = causal_mask
        else:
            attended_mask = K.minimum(K.cast(attended_mask, 'int32'),
                                      K.cast(causal_mask, 'int32'))

        # compute attention weights
        w = K.repeat(K.dot(h_cell_tm1, self.W_a) + self.b_UW, length) #TODO replace repeat with broadcasting
        e = K.exp(K.dot(K.tanh(w + u), self.v_a) + self.b_v)

        if attended_mask is not None:
            e = e * K.cast(K.expand_dims(attended_mask, -1), K.dtype(e))

        # weighted average of attended
        a = e / K.sum(e, axis=1, keepdims=True)
        c = K.sum(a * attended, axis=1, keepdims=False)

        # timestep
        t = tm1 + 1

        return c, [c, t]

    def attention_build(self, input_shape, cell_state_size, attended_shape):
        if not len(attended_shape) == 2:
            raise ValueError('There must be two attended tensors')
        for a in attended_shape:
            if not len(a) == 3:
                raise ValueError('only support attending tensors with dim=3')
        [attended_shape, u_shape] = attended_shape

        # NOTE _attention_size must always be set in `attention_build`
        self._attention_size = attended_shape[-1]
        units = u_shape[-1]

        kernel_kwargs = dict(initializer=self.kernel_initializer,
                             regularizer=self.kernel_regularizer,
                             constraint=self.kernel_constraint)
        self.W_a = self.add_weight(shape=(cell_state_size[0], units),
                                   name='W_a', **kernel_kwargs)
        self.v_a = self.add_weight(shape=(units, 1),
                                   name='v_a', **kernel_kwargs)

        bias_kwargs = dict(initializer=self.bias_initializer,
                           regularizer=self.bias_regularizer,
                           constraint=self.bias_constraint)
        self.b_UW = self.add_weight(shape=(units,),
                                    name="b_UW", **bias_kwargs)
        self.b_v = self.add_weight(shape=(1,),
                                   name="b_v", **bias_kwargs)

    def get_config(self):
        config = {
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(DenseCausalAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @property
    def attention_state_size(self):
        # second element is the current timestep
        return [self.attention_size, 1]
