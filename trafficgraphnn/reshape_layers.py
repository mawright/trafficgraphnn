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
                 num_simulations, 
                 num_timesteps, 
                 num_lanes,
                 **kwargs):
        self.num_simulations = num_simulations
        self.num_timesteps = num_timesteps
        self.num_lanes = num_lanes
        super(ReshapeForLSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_shape = input_shape
        super(ReshapeForLSTM, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        x_permuted = K.permute_dimensions(x, (0, 2, 1, 3))
        x_reshaped = K.reshape(x_permuted, (self.num_simulations*self.num_lanes, self.num_timesteps, self.input_shape[3]))
        return x_reshaped

    def compute_output_shape(self, input_shape):
        return (self.num_simulations*self.num_lanes, self.num_timesteps, input_shape[3])

    
class ReshapeForOutput(Layer):

    def __init__(self, 
                 num_simulations, 
                 num_timesteps, 
                 num_lanes,
                 **kwargs):
        self.num_simulations = num_simulations
        self.num_timesteps = num_timesteps
        self.num_lanes = num_lanes
        super(ReshapeForOutput, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_shape = input_shape
        super(ReshapeForOutput, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        x_reshaped = K.reshape(x, (-1, self.num_lanes, self.num_timesteps, self.num_features))
        x_permuted = K.permute_dimensions(x_reshaped, (0, 2, 1, 3))
        return x_permuted

    def compute_output_shape(self, input_shape):
        return (self.num_simulations*self.num_lanes, self.num_timesteps, input_shape[3])