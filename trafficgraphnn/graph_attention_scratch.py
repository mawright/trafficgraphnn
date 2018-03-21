from keras.engine.topology import Layer, InputSpec
from keras import initializers, regularizers, activations, constraints


class GraphAttentionLayer(Layer):
    """Graph attention layer from _Graph Attention Networks_ by
    Veličković et al. (2017)"""

    def __init__(
        self,
        num_output_features,
        num_attention_heads=1,
        attention_head_aggregation='concat',
        activation='relu',
        attention_coefficient_dropout_rate=0.2,
        attention_kernel_initializer='glorot_uniform',
        kernel_initializer='glorot_uniform',
        attention_kernel_regularizer=None,
        kernel_regularizer=None,
        attention_kernel_constraint=None,
        kernel_constraint=None,
        **kwargs
    ):
        assert attention_head_aggregation in [
            'concat', 'average', 'max'], [
            'Permitted attention head aggregations: \'concat\', \'average\', '
            '\'max\'']
        self.num_output_features = num_output_features
        self.num_attention_heads = num_attention_heads
        self.attention_head_aggregation = attention_head_aggregation
        self.activation = activations.get(activation),
        self.attention_kernel_initializer = initializers.get(
            attention_kernel_initializer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.attention_kernel_regularizer = regularizers.get(
            attention_kernel_regularizer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.attention_kernel_constraint = constraints.get(
            attention_kernel_constraint)
        self.kernel_constraint = constraints.get(kernel_constraint)

        self.input_spec = InputSpec(min_ndim=2)

        self.kernels = []
        self.attention_kernels = []

        if self.attention_head_aggregation == 'concat':
            self.output_dim = (
                self.num_output_features * self.num_attention_heads)
        else:
            self.output_dim = self.num_output_features

        super(GraphAttentionLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.output_dim)
        return output_shape
