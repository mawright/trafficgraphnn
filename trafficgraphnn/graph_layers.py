from six.moves import range

from keras.layers import Layer
from keras.engine import InputSpec
from keras import backend as K
from keras import activations, initializers, constraints, regularizers
import tensorflow as tf


class LocalSamplingUndirectedLayer(Layer):
    """ Local graph sampling layer (no different logic for in-edges and out-edges)

    Input shape: (batch_dim, )
    """
    def __init__(
        self, num_filters, filter_path_length, activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform', bias_initalizer='zeros',
        kernel_regularizer=None, bias_regularizer=None,
        kernel_constraint=None, bias_constraint=None,
        **kwargs
    ):
        super(LocalSamplingUndirectedLayer, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.filter_path_length = filter_path_length
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initalizer = initializers.get(bias_initalizer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=3)

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 3
        assert len(input_shape[1]) == 2

        data_shape, adj_matrix_shape = input_shape

        self.input_feature_dim = data_shape[1]
        self.kernel_shape = (
            self.filter_path_length + 1,
            self.input_feature_dim,
            self.num_filters)
        self.kernel = self.add_weight(
            shape=self.kernel_shape,
            name='kernel', initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.num_filters,), name='bias',
                initalizer=self.bias_initalizer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None

        self.input_spec = [
            InputSpec(ndim=3),  # data matrix
            InputSpec(ndim=2)   # adjacency matrix
        ]
        super(LocalSamplingUndirectedLayer, self).build(input_shape)

    def update_graph(self, graph):
        if not type(self.graph) == type(graph):
            raise ValueError((
                'New graph should be of same type as current graph: ({}).'
                'Received graph of type {}').format(
                type(self.graph), type(graph))
            )
        self.graph = graph

    def call(self, inputs):
        # TODO:
        # Generalization of a matrix-product-based convolution operation
        # 1. flatten the 3D filter tensor to a 2D matrix with shape
        # [path_legth * input_feature_dim, output_dim]
        # 2. Extract image patches from the input tensor to produce a tensor of
        # shape [batch, out_]
        data, adj_matrix = inputs

        # compute the multiplication times all filters at each node
        multiplied = K.reshape(
            K.dot(data, K.reshape(self.kernel, (self.input_feature_dim, -1))),
            (self.filter_path_length + 1, -1, self.num_filters)
        )

        num_datapoints = K.shape(data)[0]

        output = multiplied[0]
        A_k = K.eye(num_datapoints)
        included = K.eye(num_datapoints)

        for k in range(1, self.filter_path_length + 1):
            A_k = K.dot(adj_matrix, A_k)
            included_k = K.cast(
                K.all(
                    K.concatenate(
                        A_k,
                        K.not_equal(included, K.ones_like(included)),
                        axis=-1
                    ), axis=-1
                ), 'float32'
            )
            # included_k = K.cast(
                # tf.logical_and(A_k, tf.logical_not(included)), 'float32')
            K.update_add(included, included_k)

            K.update_add(output, K.dot(included_k, multiplied[k]))

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)

        return output

        # How many classes of conections? Two for this type (undirected graph): self and neighs
        # A conv2d operation with a 3x3 filter has nine classes (one for each direction)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_filters)

    def get_config(self):
        config = {
            'num_filters': self.num_filters,
            'filter_path_length': self.filter_path_length,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        return config


class LocalSamplingDirectedGraphLayer(Layer):
    pass
