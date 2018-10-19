from __future__ import absolute_import

from keras import backend as K
from keras import activations, constraints, initializers, regularizers
from keras.layers import Layer, Dropout, LeakyReLU
from keras.engine import InputSpec

class BatchGraphAttention(Layer):
    """Keras Graph Attention Layer that lets multiple batches be passed in in a single call.

    Uses additive attention.
    """
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

        super(BatchGraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        assert len(input_shape[0]) >= 3 # dimensions: batch, node, features
        assert len(input_shape[1]) >= 2 # dimensions: node, node
        F = input_shape[0][-1] # input feature dim

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
            self.add_weight(shape=(self.F_, 1),
                            initializer=self.attn_kernel_initializer,
                            name='att_kernel_neighs_{}'.format(h),
                            regularizer=self.attn_kernel_regularizer,
                            constraint=self.attn_kernel_constraint)
            for h in range(self.attn_heads)]

        self.input_spec = [InputSpec(shape=s) for s in input_shape]
        self.built = True

    def call(self, inputs):
        X = inputs[0]  # Node features (batch x N x F)
        A = inputs[1]  # Adjacency matrix (batch x N x N)

        outputs = []
        for (kernel, bias, attn_kernel_self, attn_kernel_neighs
        ) in zip(self.kernels, self.biases, self.attn_kernels_self, self.attn_kernels_neighs
                 ):
            # Compute inputs to attention network
            features = K.dot(X, kernel)  # (batch x N x F')

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            # broadcast the attention kernel across all batches and nodes
            attn_for_self = K.dot(features, attn_kernel_self)
            attn_for_neighs = K.dot(features, attn_kernel_neighs)
            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]

            trans_attn_for_neighs = K.permute_dimensions(attn_for_neighs, (0, 2, 1))

            # add dimensions to compute additive attention with broadcasting
            scores = attn_for_self + trans_attn_for_neighs  # (batch x N x N) via broadcasting

            # Add nonlinearty
            scores = LeakyReLU(alpha=0.2)(scores)

            # Mask values before activation (Vaswani et al., 2017)
            mask = (1.0 - A) * -10e9
            scores = scores + mask

            # Feed masked values to softmax
            attn_weights = K.softmax(scores)  # (batch x N x N), attention coefficients

            dropout_attn_coeffs = Dropout(self.attn_dropout)(attn_weights)  # (batch x N x N)
            dropout_features = Dropout(self.feature_dropout)(features)

            # Linear combination with neighbors' features
            # (batch x N x N) * (batch x N x F') = (batch x N x F')
            node_features = K.batch_dot(dropout_attn_coeffs, dropout_features)

            if self.use_bias:
                node_features = K.bias_add(node_features, bias)

            outputs.append(node_features)

        # Reduce the attention heads output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs, -1)  # (batch x N x KF')
        else:
            output = K.mean(K.stack(outputs, axis=0), axis=0)  # (batch x N x F')

        output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        X_shape = input_shape[0]
        assert X_shape[-1] is not None
        output_shape = list(X_shape)
        output_shape[-1] = self.output_dim
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
        base_config = super(BatchGraphAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
