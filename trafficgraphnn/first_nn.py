#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 15:33:27 2018

@author: imported and modified from keras-gat/examples/gat.py
"""
from __future__ import division
import numpy as np

from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from keras_gat import GraphAttention
from keras_gat.utils import load_data
