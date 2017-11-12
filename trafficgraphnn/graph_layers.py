from keras.layers import Layer
from keras.engine import InputSpec
from keras import backend as K
from keras import activations, initializers, constraints, regularizers


class LocalGraphLayer(Layer):
    """ Local graph sampling layer (no different logic for in-edges and out-edges)

    Input shape: (batch_dim, k-step neigh dim, neigh dim, feature dim)
    """
    def __init__(
        self, num_filters, filter_path_length, activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform', bias_initalizer='zeros',
        kernel_regularizer=None, bias_regularizer=None,
        kernel_constraint=None, bias_constraint=None,
        include_back_hops=False,
        **kwargs
    ):
        super(LocalGraphLayer, self).__init__(**kwargs)
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
        self.include_back_hops = include_back_hops
        self.input_spec = InputSpec(ndim=4)

    def build(self, input_shape):

        self.input_feature_dim = input_shape[-1]
        if self.include_back_hops:
            self.num_of_neighborhoods = 2 * self.filter_path_length + 1
        else:
            self.num_of_neighborhoods = self.filter_path_length + 1

        self.kernel_shape = (
            self.num_of_neighborhoods,
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
            InputSpec(ndim=4),  # data matrix
        ]
        super(LocalGraphLayer, self).build(input_shape)

    def update_graph(self, graph):
        if not type(self.graph) == type(graph):
            raise ValueError((
                'New graph should be of same type as current graph: ({}).'
                'Received graph of type {}').format(
                type(self.graph), type(graph))
            )
        self.graph = graph

    def call(self, data):
        # Input shape: (batch, neighs, neigh nodes, input dim)
        # TODO:
        # Generalization of a matrix-product-based convolution operation
        # 1. flatten the 3D filter tensor to a 2D matrix with shape
        # [path_legth * input_feature_dim, output_dim]
        # 2. Extract image patches from the input tensor to produce a tensor of
        # shape [batch, out_]

        num_datapoints = K.shape(data)[0]
        multiplied = K.dot(
            K.reshape(data, (
                num_datapoints,
                -1,
                self.input_feature_dim * self.num_of_neighborhoods)),
            K.reshape(self.kernel, (
                self.input_feature_dim * self.num_of_neighborhoods,
                self.num_filters)))

        output = K.sum(multiplied, axis=1)

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
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'include_back_hops': self.include_back_hops,
        }
        return config


class LocalSamplingDirectedGraphLayer(Layer):
    pass
