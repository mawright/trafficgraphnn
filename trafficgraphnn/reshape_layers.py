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
                 **kwargs):
        self.num_simulations = num_simulations
        super(ReshapeForLSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape
#        self.num_simulations = self.input_dim[0]
        self.num_timesteps = self.input_dim[1]
        self.num_lanes = self.input_dim[2]
        self.num_features = self.input_dim[3]
        super(ReshapeForLSTM, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        x_permuted = K.permute_dimensions(x, (0, 2, 1, 3))
        x_reshaped = K.reshape(x_permuted, (self.num_simulations*self.num_lanes, self.num_timesteps, self.num_features))
        return x_reshaped

    def compute_output_shape(self, input_shape):
        return (self.num_simulations*self.num_lanes, self.num_timesteps, self.num_features)
    
    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
#            'input_dim': self.input_dim,
            'num_simulations': self.num_simulations,
#            'num_timesteps': self.num_timesteps,
#            'num_lanes': self.num_lanes,
#            'num_features': self.num_features
        }
        base_config = super(ReshapeForLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#    @classmethod
#    def from_config(cls, config):
#        return cls(**config)
    
class ReshapeForOutput(Layer):

    def __init__(self, 
                 num_lanes,
                 **kwargs):
        self.num_lanes = num_lanes
        super(ReshapeForOutput, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape
#        self.num_simulations_x_num_lanes = self.input_dim[0]
        self.num_timesteps = self.input_dim[1]
        self.num_features = self.input_dim[2]
        super(ReshapeForOutput, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        x_reshaped = K.reshape(x, (-1, self.num_lanes, self.num_timesteps, self.num_features))
        x_permuted = K.permute_dimensions(x_reshaped, (0, 2, 1, 3))
        self.shape_output = x_permuted.shape
        return x_permuted

    def compute_output_shape(self, input_shape):
        #return (self.num_simulations_x_num_lanes//self.num_lanes, self.num_timesteps, self.num_lanes, self.num_features)
        return self.shape_output
    
    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
#            'input_dim': self.input_dim,
#            'num_simulations_x_num_lanes': self.num_simulations_x_num_lanes,
#            'num_timesteps': self.num_timesteps,
            'num_lanes': self.num_lanes
#            'num_features': self.num_features,
#            'shape_output': self.shape_output
        }
        base_config = super(ReshapeForOutput, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
#    @classmethod
#    def from_config(cls, config):
#        return cls(**config)