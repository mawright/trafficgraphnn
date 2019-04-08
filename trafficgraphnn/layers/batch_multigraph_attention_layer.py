from __future__ import absolute_import

from keras import backend as K
from keras import activations, constraints, initializers, regularizers
from keras.layers import Layer, LeakyReLU
from trafficgraphnn.layers.utils import batch_matmul, NEGINF

class BatchMultigraphAttention(Layer):

    def __init__(self,
                 F_,
                 attn_heads=1,
                 highway_connection=False,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 attn_dropout=0.5,
                 feature_dropout=0., # unused
                 use_bias=True,
                 activation='relu',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 num_edge_types=None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possible reduction methods: concat, average')

        self.F_ = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.highway_connection = highway_connection # add an edge type that embeds in the identity adjacency matrix
        self.attn_heads_reduction = attn_heads_reduction  # 'concat' or 'average' (Eq 5 and 6 in the paper)
        self.attn_dropout = attn_dropout  # Internal dropout rate for attention coefficients
        self.feature_dropout = feature_dropout # Dropout rate for node features
        self.activation = activations.get(activation)  # Optional nonlinearity (Eq 4 in the paper)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = None       # Layer kernels for attention heads
        self.biases = None
        self.attn_kernels_self = None  # Attention kernels for attention heads
        self.attn_kernels_neighs = None

        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.F_ * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_

        super(BatchMultigraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2 # data tensor and A tensor
        assert len(input_shape[0]) >= 3 # dimensions: batch, node, features
        assert len(input_shape[1]) >= 4 # dimensions: batch, edge type, node, node
        F = input_shape[0][-1]

        self.num_edge_types = input_shape[1][1]

        # Initialize kernels for each attention head
        # Layer kernel
        self.kernels = [
            self.add_weight(shape=(F, self.F_),
                            initializer=self.kernel_initializer,
                            name='kernel_{}'.format(h),
                            regularizer=self.kernel_regularizer,
                            constraint=self.kernel_constraint)
            for h in range(self.attn_heads)]

        if self.use_bias:
            self.biases = [
                self.add_weight(shape=(self.F_ * self.num_edge_types,), # same bias for every node
                                initializer=self.bias_initializer,
                                regularizer=self.bias_regularizer,
                                constraint=self.bias_constraint,
                                name='bias_{}'.format(h))
                for h in range(self.attn_heads)]

        # Attention kernel
        self.attn_kernels_self = [
            self.add_weight(shape=(self.F_, 1),
                            initializer=self.attn_kernel_initializer,
                            name='att_kernel_self_{}'.format(h),
                            regularizer=self.attn_kernel_regularizer,
                            constraint=self.attn_kernel_constraint)
            for h in range(self.attn_heads)]

        self.attn_kernels_neighs = [
            [
                self.add_weight(shape=(self.F_, 1),
                initializer=self.attn_kernel_initializer,
                name='att_kernel_neighs_{}_chan_{}'.format(h, chan),
                regularizer=self.attn_kernel_regularizer,
                constraint=self.attn_kernel_constraint)
                for chan in range(self.num_edge_types)
            ] for h in range(self.attn_heads)
        ]

        self.built = True

    def call(self, inputs):
        X = inputs[0]  # Node features (batch x N x F)
        A = inputs[1]  # Adjacency matrices (batch x E x N x N)

        outputs = []
        for h in range(self.attn_heads):
            kernel = self.kernels[h]
            attn_kernel_self = self.attn_kernels_self[h]
            attn_kernel_neighs = self.attn_kernels_neighs[h]
            if self.use_bias:
                bias = self.biases[h]

            # Compute inputs to attention network
            features = K.dot(X, kernel)  # (batch x N x F')

            # Compute feature combinations
            # Note: [[a_i], [a_j]]^T [[Wh_i], [Wh_2]] = [a_i]^T [Wh_i] + [a_j]^T [Wh_j]
            attn_for_self = K.dot(features, attn_kernel_self)    # (N x 1), [a_1]^T [Wh_i]
            # (N x 1), [a_j^c]^T [Wh_j^c]
            attn_for_neighs = [K.dot(features, k) for k in attn_kernel_neighs]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]

            #transpose just dimension 1 and 2, not batches
            trans_attn_for_neighs = [K.permute_dimensions(an, [0, 2, 1])
                                     for an in attn_for_neighs] # (batch x 1 x N)
            stacked_attn_for_neighs = K.stack(trans_attn_for_neighs, 1) # (batch x E x 1 x N)
            attn_for_self = K.expand_dims(attn_for_self, 1) # (batch x 1 x N x 1)
            scores = attn_for_self + stacked_attn_for_neighs  # (batch x E x N x N) via broadcasting

            # Add nonlinearty
            scores = LeakyReLU(alpha=0.2)(scores)

            # Mask values before activation (Vaswani et al., 2017)
            mask = (1.0 - A) * NEGINF # (batch x E x N x N)
            masked = scores + mask

            # Feed masked values to softmax
            softmax = K.softmax(masked)  # (batch x E x N x N), attention coefficients

            shape = K.shape(softmax)
            noise_shape = [shape[0], 1, shape[2], shape[3]]

            dropout_lambda = lambda: K.dropout(softmax, self.attn_dropout, noise_shape)  # (batch x E x N x N)

            dropout = K.in_train_phase(dropout_lambda, softmax)

            features_expanded = K.expand_dims(features, 1)
            features_expanded = K.repeat_elements(features_expanded,
                                                  self.num_edge_types, 1)

            # Linear combination with neighbors' features
            node_features = batch_matmul(dropout, features_expanded) # (batch x E x N x F')

            node_features = K.permute_dimensions(node_features, [0, 2, 1, 3])
            if self.highway_connection:
                node_features = K.concatenate(
                    [node_features, K.expand_dims(features, -2)], -1)

            shape = K.shape(node_features)
            node_features = K.reshape(node_features,
                                      K.concatenate([shape[:2],
                                                     K.prod(shape[2:], keepdims=True)])) # (batch x N x EF')

            if self.use_bias:
                node_features = K.bias_add(node_features, bias)

            # Add output of attention head to final output
            outputs.append(node_features)

        # Reduce the attention heads output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs, -1)  # (batch x N x EKF')
        else:
            output = K.mean(K.stack(outputs, axis=0), axis=0)  #( batch x N x EF')

        output = self.activation(output)
        if 0. < self.attn_dropout < 1.:
            output._uses_learning_phase = True

        return output

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        X_shape = input_shape[0]
        num_edge_types = input_shape[1][1]
        assert input_shape[-1] is not None
        output_shape = list(X_shape)
        output_shape[-1] = self.output_dim * num_edge_types
        return tuple(output_shape)

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'F_': self.F_,
            'attn_heads': self.attn_heads,
            'attn_heads_reduction': self.attn_heads_reduction,
            'attn_dropout': self.attn_dropout,
            'feature_dropout': self.feature_dropout,
            'use_bias': self.use_bias,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'attn_kernel_initializer': self.attn_kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'attn_kernel_regularizer': self.attn_kernel_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint,
            'attn_kernel_constraint': self.attn_kernel_constraint
        }
        base_config = super(BatchMultigraphAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
