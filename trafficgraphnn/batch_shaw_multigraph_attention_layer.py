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
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 edge_type_reduction='concat',
                 attn_dropout=0.5,
                 feature_dropout=0.,
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

        self.F_ = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # 'concat' or 'average' (Eq 5 and 6 in the paper)
        self.edge_type_reduction = edge_type_reduction
        self.attn_dropout = attn_dropout  # Internal dropout rate for attention coefficients
        self.feature_dropout = feature_dropout
        self.activation = activations.get(activation)  # Optional nonlinearity (Eq 4 in the paper)

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
        self.kernel = None       # Layer kernels (value) for attention heads
        self.attn_kernel_self = None  # Attention kernels for attention heads (Query)
        self.attn_kernel_neighs = None # (Key)

        self.biases = None
        self.attn_biases = None

        self.output_dim = None

        super(BatchShawMultigraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2 # data tensor and A tensor
        assert len(input_shape[0]) == 3 # dimensions: batch, node, features
        F = input_shape[0][-1]

        assert len(input_shape[1]) == 4 # dimensions: batch, num_edge_types, node, node

        if input_shape[1][1] is None:
            raise ValueError('Number of edge types (dimension 1 of the A tensor) must be specified')

        self.num_edge_types = input_shape[1][1]

        self.output_dim = self.F_
        if self.edge_type_reduction == 'concat':
            self.output_dim *= self.num_edge_types
        if self.attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim *= self.attn_heads

        # Initialize kernels for each attention head
        # Layer kernel: dims (h X F x F')
        self.kernel = self.add_weight(shape=(self.attn_heads, F, self.F_),  # "value" weight matrix
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        # per-edge-type value bias vectors: dims (E x h x F')
        if self.use_value_bias:
            self.biases = self.add_weight(shape=(self.num_edge_types, self.attn_heads, self.F_),
                                          initializer=self.bias_initializer,
                                          name='output_biases',
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint)

        # Attention kernel
        # Self ("Query") kernel: dims (h, F, F')
        self.attn_kernel_self = self.add_weight(shape=(self.attn_heads, F, self.F_),
                                                initializer=self.attn_kernel_initializer,
                                                name='att_kernel_self',
                                                regularizer=self.attn_kernel_regularizer,
                                                constraint=self.attn_kernel_constraint)
        # Neighbors ("key") kernel: dims (h, F, F')
        self.attn_kernel_neighs = self.add_weight(shape=(self.attn_heads, F, self.F_), # "key" weight matrix
                                                  initializer=self.attn_kernel_initializer,
                                                  name='att_kernel_neighs',
                                                  regularizer=self.attn_kernel_regularizer,
                                                  constraint=self.attn_kernel_constraint)

        # per-edge-type attention ("key") biases: dim (E x h X F')
        if self.use_key_bias:
            self.attn_biases = self.add_weight(shape=(self.num_edge_types, self.attn_heads, self.F_),
                                               initializer=self.attn_kernel_initializer,
                                               name='att_biases',
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint)

        self.input_spec = [InputSpec(shape=s) for s in input_shape]
        self.built = True

    def call(self, inputs):
        X = inputs[0]  # Node features (batch x N x F)
        A = inputs[1]  # Adjacency matrices (batch x |E| x N x N)

        # Compute feature combinations
        # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
        attn_for_self = K.dot(X, self.attn_kernel_self)    # (batch x N x h x F'), [a_1]^T [Wh_i]
        attn_for_neighs = K.dot(X, self.attn_kernel_neighs)  # (batch x N x h x F'), [a_2]^T [Wh_j]

        # calculate attention score by dot-producting the attention for self
        # and attention for neighbors vectors for each head and each node
        transpose_self = K.permute_dimensions(attn_for_self, (0, 2, 1, 3)) # (batch x h x N x F_)
        transpose_neighs = K.permute_dimensions(attn_for_neighs, (0, 2, 3, 1)) # (batch x h x F_ x N)

        # expand to batch and node dims for broadcasting
        key_tensor = K.expand_dims(transpose_neighs, 1)
        if self.use_key_bias:
            expanded_biases = K.expand_dims(K.expand_dims(self.attn_biases, 0), -1)
            key_tensor = key_tensor + expanded_biases # (batch x E x h x F_ x N)
        else:
            key_tensor = K.repeat_elements(key_tensor, self.num_edge_types, 1)

        # expand the self (query) tensor across edge types
        tranpose_self_expand = K.expand_dims(transpose_self, 1)
        # no implementation of broadcasted batch matrix multiply, so we need to tile the
        # self ("query") tensor across the edge-type dimension
        tiled_transpose_self = K.repeat_elements(tranpose_self_expand,
                                                 self.num_edge_types, 1)
        attn_scores = K.batch_dot(tiled_transpose_self, key_tensor) # (batch x E x h x N x N)

        # scaled dot-product: normalize by feature dimension
        attn_scores = attn_scores / K.sqrt(K.constant(self.F_, dtype=attn_scores.dtype))

        expanded_A = K.expand_dims(A, 2) # same A matrices per attention head

        # Mask values before activation (Vaswani et al., 2017)
        mask = (1.0 - expanded_A) * -10e9
        masked = attn_scores + mask

        # Feed masked values to softmax
        attn_weights = K.softmax(masked)  # (batch x E x h x N x N), attention coefficients
        dropout = Dropout(self.attn_dropout)(attn_weights)

        # now compute the "value" transformations
        linear_transf_X = K.dot(X, self.kernel)  # (batch x N x h x F')
        # reshape the X-value tensor: swap head and node dimensions
        permuted_value_x = K.permute_dimensions(linear_transf_X, (0, 2, 1, 3))
        # add the edge-type dimension
        expanded_value_x = K.expand_dims(permuted_value_x, 1)
        # repeat over edge-type dimension
        # (needed because broadcasted batch matrix multiply not implemented in tf)
        value_x = K.repeat_elements(expanded_value_x, self.num_edge_types, 1)
        # new dims: (batch x E x h x v x F')

        # add edge-type-dependent bias vectors
        if self.use_value_bias:
            expanded_value_biases = K.expand_dims(K.expand_dims(self.biases, 0), -2)
            value_x = value_x + expanded_value_biases

        value_x = Dropout(self.feature_dropout)(value_x)

        # Linear combination with neighbors' features
        node_features = K.batch_dot(dropout, value_x)  # (batch x E x h x N x F')
        output = K.permute_dimensions(node_features, (0, 3, 1, 2, 4)) # (batch x N x E x h x F')

        if (self.attn_heads_reduction == 'concat'
                and self.edge_type_reduction == 'concat'
                and self.activation is not None):
            # In case of 'concat', we compute the activation here (Eq 5)
            output = self.activation(output)

        # Reduce the edge type output according to the specified reduction method
        if self.edge_type_reduction == 'concat':
            shape = K.shape(output)
            output = K.reshape(output, (shape[0], shape[1], shape[2]*shape[3], shape[4]))
        else:
            output = K.mean(output, 2)

        # Reduce the attention heads output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            shape = K.shape(output)
            output = K.reshape(output, (shape[0], shape[1], shape[2]*shape[3]))
        else:
            output = K.mean(output, axis=2)  # (batch x N x F')
            if self.activation is not None:
                # In case of 'average', we compute the activation here (Eq 6)
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
