from __future__ import absolute_import

from keras import backend as K
from keras import activations, constraints, initializers, regularizers
from keras.layers import Layer, Dropout, LeakyReLU
from keras.utils.generic_utils import to_list

class BatchMultigraphAttention(Layer):

    def __init__(self,
                 F_,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 attn_dropout=0.5,
                 feature_dropout=0.,
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
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possible reduction methods: concat, average')

        self.F_ = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
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
        assert len(input_shape) == 2 # data tensor and list of A tensors
        assert len(input_shape[0]) >= 3 # dimensions: batch, node, features
        F = input_shape[0][-1]

        A_list = to_list(input_shape[1])

        self.num_edge_types = len(A_list)

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
                self.add_weight(shape=(self.F_,), # same bias for every node
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
        raise NotImplementedError
        X = inputs[0]  # Node features (batch x N x F)
        A = inputs[1]  # Adjacency matrices (E-long list of (N x N))

        outputs = []
        for h in range(self.attn_heads):
            kernel = self.kernels[h]
            attn_kernel_self = self.attn_kernels_self[h]
            attn_kernel_neighs = self.attn_kernels_neighs[h]
            if self.use_bias:
                bias = self.biases[h]

            # Compute inputs to attention network
            features = K.dot(X, kernel)  # (batch N x F')

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(features, attn_kernel_self)    # (N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = K.dot(features, attn_kernel_neighs)  # (N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]

            #transpose just dimension 1 and 2, not batches
            trans_attn_for_neighs = K.permute_dimensions(attn_for_neighs, perm=[0, 2, 1])
            scores = attn_for_self + trans_attn_for_neighs  # (N x N) via broadcasting

            # Add nonlinearty
            scores = LeakyReLU(alpha=0.2)(scores)

            # Mask values before activation (Vaswani et al., 2017)
            mask = K.exp(A * -10e9) * -10e9
            masked = scores + mask

            # Feed masked values to softmax
            softmax = K.softmax(masked)  # (N x N), attention coefficients
            dropout = Dropout(self.attn_dropout)(softmax)  # (N x N)

            # Linear combination with neighbors' features
            node_features = K.batch_dot(dropout, features)  # (N x F')

            if self.use_bias:
                node_features = K.bias_add(node_features, bias)

            if self.attn_heads_reduction == 'concat' and self.activation is not None:
                # In case of 'concat', we compute the activation here (Eq 5)
                node_features = self.activation(node_features)

            # Add output of attention head to final output
            outputs.append(node_features)

        # Reduce the attention heads output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs)  # (N x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0)  # N x F')
            if self.activation is not None:
                # In case of 'average', we compute the activation here (Eq 6)
                output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        X_shape = input_shape[0]
        assert input_shape[-1] is not None
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)
