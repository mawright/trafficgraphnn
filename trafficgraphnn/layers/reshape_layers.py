#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 10:47:56 2018

@author: simon
"""

from keras import backend as K
from keras.engine.topology import Layer

class ReshapeForLSTM(Layer):
    def __init__(self,
                 **kwargs):
        super(ReshapeForLSTM, self).__init__(**kwargs)

    def call(self, x):
        x_permuted = K.permute_dimensions(x, (0, 2, 1, 3))
        shape = K.shape(x_permuted)
        x_reshaped = K.reshape(x_permuted, (-1, shape[2], shape[3]))
        return x_reshaped

    def compute_output_shape(self, input_shape):
        if input_shape[0] is not None:
            first_dim = input_shape[0] * input_shape[2]
        else:
            first_dim = None
        return (first_dim, input_shape[1], input_shape[3])

    def get_config(self):
        base_config = super(ReshapeForLSTM, self).get_config()
        return base_config


class ReshapeForOutput(Layer):

    def __init__(self,
                 num_lanes,
                 **kwargs):
        self.num_lanes = num_lanes
        super(ReshapeForOutput, self).__init__(**kwargs)

    def call(self, x):
        shape = K.shape(x)
        assert shape.shape == 3 # number of dimensions
        x_reshaped = K.reshape(x, (-1, self.num_lanes, shape[-2], shape[-1]))
        x_permuted = K.permute_dimensions(x_reshaped, (0, 2, 1, 3))
        self.shape_output = x_permuted.shape
        return x_permuted

    def compute_output_shape(self, input_shape):
#        shape = [-1, input_shape[-2], self.num_lanes, input_shape[-1]]
        shape = [None, input_shape[-2], self.num_lanes, input_shape[-1]] #test
        return tuple(shape)

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'num_lanes': self.num_lanes
        }
        base_config = super(ReshapeForOutput, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
