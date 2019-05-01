from keras import backend as K
from keras.layers import Layer, InputSpec
from keras import constraints, initializers, regularizers


class LayerNormalization(Layer):
    """Based on tf.contrib.layers.layer_norm and keras.layers.BatchNormalization"""
    def __init__(self,
                 begin_norm_axis=1,
                 begin_params_axis=-1,
                 center=True,
                 scale=True,
                 epsilon=1e-3,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.begin_norm_axis = begin_norm_axis
        self.begin_params_axis = begin_params_axis
        self.center = center
        self.scale = scale
        self.epsilon = epsilon
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.supports_masking = True

    def build(self, input_shape):
        param_shape = input_shape[self.begin_params_axis:]
        if any([dim is None for dim in param_shape]):
            raise ValueError('Axes ' + str(self.begin_params_axis) + ' and later '
                             'of input tensor should have a defined dimension'
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        if self.begin_norm_axis < 0:
            begin_norm_axis = len(input_shape) + self.begin_norm_axis
        else:
            begin_norm_axis = self.begin_norm_axis
        if (begin_norm_axis >= len(input_shape)
            or self.begin_params_axis >= len(input_shape)
        ):
            raise ValueError('begin_params_axis (%d) and begin_norm_axis (%d) '
                             'must be < input dims (%d)' % (
                                 self.begin_params_axis, begin_norm_axis,
                                 len(input_shape))
                            )

        self.input_spec = InputSpec(
            ndim=len(input_shape),
            axes=dict(list(enumerate(input_shape))[self.begin_params_axis:])
        )

        if self.scale:
            self.gamma = self.add_weight(shape=param_shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=param_shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs):
        input_shape = K.int_shape(inputs)
        input_dims = len(input_shape)
        norm_axes = list(range(input_dims))[self.begin_norm_axis:]

        mean = K.mean(inputs, norm_axes, keepdims=True)
        variance = K.var(inputs, norm_axes, keepdims=True)

        inv_sqrt = 1 / K.sqrt(variance + self.epsilon)
        normalized = (inputs - mean) * inv_sqrt

        if self.gamma is not None:
            normalized *= self.gamma
        if self.beta is not None:
            normalized += self.beta

        return normalized

    def get_config(self):
        config = {
            'begin_norm_axis': self.begin_norm_axis,
            'begin_params_axis': self.begin_params_axis,
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
