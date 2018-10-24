from __future__ import absolute_import

from keras import backend as K
from keras import activations, constraints, initializers, regularizers
from keras.layers import Layer, Dropout, InputSpec

class BatchShawMultigraphAttention(Layer):
    """Batch Graph Attention Layer with Shaw et al.'s edge-type-specific biases for multigraphs.

    Uses scaled dot-product attention.
    """
    def __init__(self,
                 F_,
                 attn_heads=1,
                 attention_type='multiplicative',
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 edge_type_reduction='concat',
                 attn_dropout=0.5,
                 feature_dropout=0.,
                 zero_isolated=True,
                 use_value_bias=True,
                 use_key_bias=True,
                 activation='relu',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 attn_bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 attn_bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 attn_bias_constraint=None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')
        if attention_type not in {'multiplicative', 'additive'}:
            raise ValueError("Allowed attention types: 'multiplicative', 'additive'")

        self.F_ = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attention_type = attention_type
        self.attn_heads_reduction = attn_heads_reduction  # 'concat' or 'average' (Eq 5 and 6 in the paper)
        self.edge_type_reduction = edge_type_reduction
        self.attn_dropout = attn_dropout  # Internal dropout rate for attention coefficients
        self.feature_dropout = feature_dropout
        self.activation = activations.get(activation)  # Optional nonlinearity (Eq 4 in the paper)
        self.zero_isolated = zero_isolated

        self.use_value_bias = use_value_bias
        self.use_key_bias = use_key_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.attn_bias_initializer = initializers.get(attn_bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.attn_bias_regularizer = regularizers.get(attn_bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.attn_bias_constraint = constraints.get(attn_bias_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = None       # Layer kernels (value) for attention heads
        self.attn_kernels_self = None  # Attention kernels for attention heads (Query)
        self.attn_kernels_neighs = None # (Key)

        self.biases = None
        self.attn_biases = None

        self.output_dim = None

        super(BatchShawMultigraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2 # data tensor and A tensor
        assert len(input_shape[0]) == 3 # dimensions: batch, node, features
        F = input_shape[0][-1]

        assert len(input_shape[1]) == 4 # dimensions: batch, num_edge_types, node, node

        num_edge_types = input_shape[1][-1]
        if num_edge_types is None:
            raise ValueError('Number of edge types (dimension -1 of the A tensor) must be specified')

        self.num_edge_types = num_edge_types

        self.output_dim = self.F_
        if self.edge_type_reduction == 'concat':
            self.output_dim *= self.num_edge_types
        if self.attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim *= self.attn_heads

        # Initialize kernels for each attention head
        # Layer kernel: dims (F x F')
        self.kernels = [
            self.add_weight(shape=(F, self.F_),  # "value" weight matrix
                            initializer=self.kernel_initializer,
                            name='kernel_head_{}'.format(h),
                            regularizer=self.kernel_regularizer,
                            constraint=self.kernel_constraint)
            for h in range(self.attn_heads)]

        # per-edge-type value bias vectors: dims (E x F')
        if self.use_value_bias:
            self.biases = [
                self.add_weight(shape=(self.num_edge_types, self.F_),
                                initializer=self.bias_initializer,
                                name='output_bias_{}'.format(h),
                                regularizer=self.bias_regularizer,
                                constraint=self.bias_constraint)
                for h in range(self.attn_heads)]

        # Attention kernel
        # Self ("Query") kernel: dims (F, F')
        self.attn_kernels_self = [
            self.add_weight(shape=(F, self.F_),
                            initializer=self.attn_kernel_initializer,
                            name='att_kernel_self_{}'.format(h),
                            regularizer=self.attn_kernel_regularizer,
                            constraint=self.attn_kernel_constraint)
            for h in range(self.attn_heads)]
        # Neighbors ("key") kernel: dims (F, F')
        self.attn_kernels_neighs = [
            self.add_weight(shape=(F, self.F_), # "key" weight matrix
                            initializer=self.attn_kernel_initializer,
                            name='att_kernel_neighs_{}'.format(h),
                            regularizer=self.attn_kernel_regularizer,
                            constraint=self.attn_kernel_constraint)
            for h in range(self.attn_heads)]

        # per-edge-type attention ("key") biases: dim (F' x E)
        if self.use_key_bias:
            self.attn_biases = [
                self.add_weight(shape=(self.F_, self.num_edge_types),
                                initializer=self.attn_kernel_initializer,
                                name='att_biases_{}'.format(h),
                                regularizer=self.attn_kernel_regularizer,
                                constraint=self.attn_kernel_constraint)
                for h in range(self.attn_heads)]

        self.input_spec = [InputSpec(shape=s) for s in input_shape]
        self.built = True

    def call(self, inputs):
        X = inputs[0]  # Node features (batch x N x F)
        A = inputs[1]  # Adjacency matrices (batch x N x N x E)

        N = K.shape(X)[1]

        outputs = []
        for h in range(self.attn_heads):
            kernel = self.kernels[h]
            attn_kernel_self = self.attn_kernels_self[h]
            attn_kernel_neighs = self.attn_kernels_neighs[h]
            if self.use_value_bias:
                value_bias = self.biases[h]
            if self.use_key_bias:
                key_bias = self.attn_biases[h]
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(X, attn_kernel_self)    # (batch x N x F'), [a_1]^T [Wh_i]
            attn_for_neighs = K.dot(X, attn_kernel_neighs)  # (batch x N x F'), [a_2]^T [Wh_j]

            # calculate attention score by dot-producting the attention for self
            # and attention for neighbors vectors for each head and each node
            dotproduct = K.batch_dot(attn_for_self, attn_for_neighs, axes=[2, 2]) # (batch x N x N)

            if self.use_key_bias:
                dotproduct_bias = K.dot(attn_for_self, key_bias) # (batch x N x E)
                attn_scores = K.expand_dims(dotproduct, -1) + K.expand_dims(dotproduct_bias, 1) # (batch x N x N x E)
            else:
                attn_scores = K.expand_dims(dotproduct, -1)

            # scaled dot-product: normalize by feature dimension
            attn_scores = attn_scores / K.sqrt(K.constant(self.F_, dtype=attn_scores.dtype))

            # Mask values before activation (Vaswani et al., 2017)
            mask = (1.0 - A) * -10e9
            masked = attn_scores + mask

            # Feed masked values to softmax
            attn_weights = K.softmax(masked, 2)  # (batch x N x N x E), attention coefficients

            # if self.zero_isolated:
            #     attn_weights = K.switch(K.sum(A, 2) >= 1, attn_weights, K.zeros_like(attn_weights))
            attn_weight_dropout = Dropout(self.attn_dropout)(attn_weights)

            # now compute the "value" transformations
            features = K.dot(X, kernel)  # (batch x N x F')

            weight_slices = [attn_weight_dropout[...,e] for e in range(self.num_edge_types)]
            if self.use_value_bias:
                bias_slices = [value_bias[e] for e in range(self.num_edge_types)]
                weighted_features = [K.batch_dot(ws, features + b) for ws, b in zip(weight_slices, bias_slices)]
            else:
                weighted_features = [K.batch_dot(ws, features) for ws in weight_slices]
            weighted_sum_features = K.stack(weighted_features, 2) # (batch x N x E x F')

            outputs.append(weighted_sum_features)

        # Reduce the edge type output according to the specified reduction method
        if self.edge_type_reduction == 'concat':
            outputs = [K.reshape(output, (-1, N, self.num_edge_types*self.F_)) for output in outputs]
        else:
            outputs = [K.mean(output, 2) for output in outputs]

        # Reduce the attention heads output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs, -1)
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
        """For rebuilding models on load time
        """
        config = {
            'F_': self.F_,
            'attn_heads': self.attn_heads,
            'attn_heads_reduction': self.attn_heads_reduction,
            'edge_type_reduction': self.edge_type_reduction,
            'attention_type': self.attention_type,
            'attn_dropout': self.attn_dropout,
            'feature_dropout': self.feature_dropout,
            'activation': self.activation,
            'use_value_bias': self.use_value_bias,
            'use_key_bias': self.use_key_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'attn_kernel_initializer': self.attn_kernel_initializer,
            'attn_bias_initalizer': self.attn_bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'attn_kernel_regularizer': self.attn_kernel_regularizer,
            'attn_bias_regularizer': self.attn_bias_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint,
            'attn_kernel_constraint': self.attn_kernel_constraint,
            'attn_bias_constraint': self.attn_bias_constraint
        }
        base_config = super(BatchShawMultigraphAttention, self).get_config()
        return dict(list(base_config.items())) + list(config.items())
