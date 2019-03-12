#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 10:47:56 2018

@author: simon
"""

from keras import backend as K
from keras.engine.topology import Layer


class ReshapeFoldInLanes(Layer):
    def __init__(self,
                 batch_size=None,
                 **kwargs):
        self.batch_size = batch_size
        super(ReshapeFoldInLanes, self).__init__(**kwargs)

    def call(self, x):
        x_permuted = K.permute_dimensions(x, (0, 2, 1, 3))
        shape = K.shape(x_permuted)
        int_shape = K.int_shape(x_permuted)

        if int_shape[0] is not None:
            shape1 = -1 # can be inferred
        elif self.batch_size is not None and int_shape[1] is not None:
            shape1 = self.batch_size * int_shape[1] # needs to be specified

        if int_shape[2] is not None:
            shape2 = int_shape[2]
        else:
            shape2 = shape[2]
        if int_shape[3] is not None:
            shape3 = int_shape[3]
        else:
            shape3 = shape[3]

        x_reshaped = K.reshape(x_permuted, (shape1, shape2, shape3))
        return x_reshaped

    def compute_output_shape(self, input_shape):
        if input_shape[0] is not None and input_shape[2] is not None:
            first_dim = input_shape[0] * input_shape[2]
        elif self.batch_size is not None and input_shape[2] is not None:
            first_dim = self.batch_size * input_shape[2]
        else:
            first_dim = None
        return (first_dim, input_shape[1], input_shape[3])

    def get_config(self):
        base_config = super(ReshapeFoldInLanes, self).get_config()
        return base_config


class ReshapeUnfoldLanes(Layer):
    def __init__(self,
                 num_lanes,
                 **kwargs):
        self.num_lanes = num_lanes
        super(ReshapeUnfoldLanes, self).__init__(**kwargs)

    def call(self, x):
        shape = K.shape(x)
        assert shape.shape == 3 # number of dimensions
        int_shape = K.int_shape(x)

        if int_shape[1] is not None:
            shape1 = int_shape[1]
        else:
            shape1 = shape[1]
        if int_shape[2] is not None:
            shape2 = int_shape[2]
        else:
            shape2 = shape[2]

        x_reshaped = K.reshape(x, (-1, self.num_lanes, shape1, shape2))
        x_permuted = K.permute_dimensions(x_reshaped, (0, 2, 1, 3))
        self.shape_output = x_permuted.shape
        return x_permuted

    def compute_output_shape(self, input_shape):
        shape = [input_shape[0], input_shape[-2], self.num_lanes, input_shape[-1]]
        return tuple(shape)

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'num_lanes': self.num_lanes
        }
        base_config = super(ReshapeUnfoldLanes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# legacy names
ReshapeForLSTM = ReshapeFoldInLanes
ReshapeForOutput = ReshapeUnfoldLanes
