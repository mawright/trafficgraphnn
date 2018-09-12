#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 13:13:45 2018

@author: simon
"""
from __future__ import division
from trafficgraphnn.genconfig import ConfigGenerator
from trafficgraphnn.sumo_network import SumoNetwork
import numpy as np
import math
from trafficgraphnn.liumethod import LiuEtAlRunner

from trafficgraphnn.preprocess_data import PreprocessData

import tensorflow as tf
import keras.backend as K
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Input, Dropout, Dense, TimeDistributed, Reshape, Lambda, LSTM
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD, Adagrad
from keras.regularizers import l2
from trafficgraphnn.batch_graph_attention_layer import  BatchGraphAttention
from keras.utils.vis_utils import plot_model

#------ Configuration of the whole simulation -------


### Configuration of the Network ###
grid_number = 3
grid_length = 600 #meters
num_lanes =3

### Configuration of the Simulation ###
end_time = 1500 #seconds
period = 0.4
binomial = 2
seed = 50
fringe_factor = 1000

### Configuration of Liu estimation ###
use_started_halts = False #use startet halts as ground truth data or maxJamLengthInMeters
show_plot = False
show_infos = True

### Configuration for preprocessing the detector data
average_interval = 1  #Attention -> right now no other average interval than 1 is possible  -> bugfix necessary!
sample_size = 10

### Configuration of the deep learning model
width_1gat = 100 # Output dimension of first GraphAttention layer
F_ = 4         # Output dimension of last GraphAttention layer
n_attn_heads = 5              # Number of attention heads in first GAT layer
dropout_rate = 0            # Dropout rate applied to the input of GAT layers
attn_dropout = 0            #Dropout of the adjacency matrix in the gat layer
l2_reg = 5e-100               # Regularization rate for l2
learning_rate = 5e-2       # Learning rate for SGD
epochs = 3              # Number of epochs to run for
es_patience = 100             # Patience fot early stopping
n_units = 128   #number of units of the LSTM cells

#----------------------------------------------------


### Creating Network and running simulation
config = ConfigGenerator(net_name='test_net')

# Parameters for network, trips and sensors (binomial must be an integer!!!)
config.gen_grid_network(grid_number = grid_number, grid_length = grid_length, num_lanes = num_lanes, simplify_tls = False)
config.gen_rand_trips(period = period, binomial = binomial, seed = seed, end_time = end_time, fringe_factor = fringe_factor)

config.gen_e1_detectors(distance_to_tls=[5, 125], frequency=1)
config.gen_e2_detectors(distance_to_tls=0, frequency=1)
config.define_tls_output_file()

# run the simulation to create output files
list_X = []
list_Y = []
list_A = []

for simu_num in range(3):
    config.gen_rand_trips(period = period, binomial = binomial, seed = seed, end_time = end_time, fringe_factor = fringe_factor)
    sn = SumoNetwork.from_gen_config(config, lanewise=True)
    sn.run()
    print('Simulation run number', simu_num, 'finished')


    ### Running the Liu Estimation
    #creating liu runner object
    liu_runner = LiuEtAlRunner(sn, store_while_running = True, use_started_halts = use_started_halts, simu_num = simu_num)
    
    # caluclating the maximum number of phases and run the estimation
    max_num_phase = liu_runner.get_max_num_phase(end_time)
    liu_runner.run_up_to_phase(max_num_phase)
    
    # show results for every lane
    liu_runner.plot_results_every_lane(show_plot = show_plot, show_infos = show_infos)
    
    
    ### preprocess data for deep learning model
    preprocess = PreprocessData(sn, simu_num)
    preproccess_end_time = preprocess.get_preprocessing_end_time(liu_runner.get_liu_lane_IDs())
    A, X, Y = preprocess.preprocess_X_Y(
            average_interval = average_interval, sample_size = sample_size, start = 200, end = preproccess_end_time, simu_num = simu_num)
    A = np.eye(120,120) + A
    
    list_A.append(A)
    list_X.append(X)
    list_Y.append(Y)
    
X_train_tens = list_X[0]
X_val_tens = list_X[1]
X_test_tens = list_X[2]

Y_train_tens = list_Y[0]
Y_val_tens = list_Y[1]
Y_test_tens = list_Y[2]

A = list_A[0] # A does not change 


print('X_train_tens.shape:', X_train_tens.shape)
print('X_test_tens.shape:', X_test_tens.shape)
print('X_val_tens.shape:', X_val_tens.shape)

print('now reduce batch size')
### delete reduction later!!!
# reduce batch size
X_train_tens = X_train_tens[0:16, :, :, :]
Y_train_tens = Y_train_tens[0:16, :, :, :]
X_val_tens = X_train_tens[0:16, :, :, :]
Y_val_tens = Y_train_tens[0:16, :, :, :]
X_test_tens = X_train_tens[0:16, :, :, :]
Y_test_tens = Y_train_tens[0:16, :, :, :]
print('X_train_tens.shape:', X_train_tens.shape)
print('Y_train_tens.shape:', Y_train_tens.shape)
###



### Train the deep learning model ###
sample_size_train = int(X_train_tens.shape[0]) 
sample_size_val = int(X_val_tens.shape[0]) 
sample_size_test = int(X_test_tens.shape[0])

print('sample_size_train:', sample_size_train)


timesteps_per_sample = X_train_tens.shape[1] #Number of timesteps in a sample
N = X_train_tens.shape[2]          # Number of nodes in the graph
F = X_train_tens.shape[3]          # Original feature dimensionality

print('timesteps_per_sample', timesteps_per_sample)
print('N', N)
print('F', F)

#define necessary functions
def shape_X1(x):
    print('x.shape:', x.shape)
    return K.reshape(x, (int(x.shape[0])//N, timesteps_per_sample, N, F))

def reshape_X1(x):
    return K.reshape(x, (-1, timesteps_per_sample, F_))

def reshape_X2(x):
    return K.reshape(x, (-1, timesteps_per_sample, 1))

#def reshape_output(x, sample_size):
#    return K.reshape(x, (sample_size, timesteps_per_sample, N, 1))

#def reshape_encoder_states(x, sample_size):
#    return K.reshape(x, (sample_size, N, n_units))

def calc_X2(Y):
    sample_size = Y.shape[0]
    start_slice = np.zeros((1, N, 1))
    X2 = np.zeros((sample_size, timesteps_per_sample, N, 1))
    for sample in range(sample_size):  
        X2[sample, :, :, :] = np.concatenate([start_slice, Y[sample, :-1, :, :]], axis = 0)
    X2 = tf.Variable(X2) 
    return X2

##reshape X and Y tensor for the deep learning model
    
#training
X1_train = K.reshape(X_train_tens, (sample_size_train*N, timesteps_per_sample, F))
X2_train = calc_X2(Y_train_tens)
X2_train = K.reshape(X2_train, (sample_size_train*N, timesteps_per_sample, 1))
Y_train = K.reshape(Y_train_tens, (sample_size_train*N, timesteps_per_sample, 1))
print('X1_train.shape:', X1_train.shape)
print('X2_train.shape:', X2_train.shape)
print('Y_train.shape:', Y_train.shape)

#validation
X1_val = K.reshape(X_val_tens, (sample_size_val*N, timesteps_per_sample, F))
X2_val = calc_X2(Y_val_tens)
X2_val = K.reshape(X2_val, (sample_size_val*N, timesteps_per_sample, 1))
Y_val = K.reshape(Y_val_tens, (sample_size_val*N, timesteps_per_sample, 1))
print('X1_val.shape:', X1_val.shape)
print('X2_val.shape:', X2_val.shape)
print('Y_val.shape:', Y_val.shape)

#test
X1_test = K.reshape(X_test_tens, (sample_size_test*N, timesteps_per_sample, F))
X2_test = calc_X2(Y_test_tens)
X2_test = K.reshape(X2_test, (sample_size_test*N, timesteps_per_sample, 1))
Y_test = K.reshape(Y_test_tens, (sample_size_test*N, timesteps_per_sample, 1))
print('X1_test.shape:', X1_test.shape)
print('X2_test.shape:', X2_test.shape)
print('Y_test.shape:', Y_test.shape)

A_tf = tf.convert_to_tensor(A, dtype=np.float32)

#define the training model
#define the training model
X1_in = Input(batch_shape=(sample_size_train*N, timesteps_per_sample, F))
X2_in = Input(batch_shape=(sample_size_train*N, timesteps_per_sample, 1))

shaped_X1_in = Lambda(shape_X1)(X1_in)

dropout1 = TimeDistributed(Dropout(dropout_rate))(shaped_X1_in)

graph_attention_1 = TimeDistributed(BatchGraphAttention(width_1gat,
                                   A_tf,
                                   attn_heads=n_attn_heads,
                                   attn_heads_reduction='average',
                                   attn_dropout=attn_dropout,
                                   activation='linear',
                                   kernel_regularizer=l2(l2_reg)),
                                   )(dropout1)

dropout2 = TimeDistributed(Dropout(dropout_rate))(graph_attention_1)

graph_attention_2 = TimeDistributed(BatchGraphAttention(F_,
                                   A_tf,
                                   attn_heads=n_attn_heads,
                                   attn_heads_reduction='average',
                                   attn_dropout=attn_dropout,
                                   activation='linear',
                                   kernel_regularizer=l2(l2_reg)))(dropout2)

#make sure that the reshape is made correctly!
encoder_inputs = Lambda(reshape_X1)(graph_attention_2)
decoder_inputs = Lambda(reshape_X2)(X2_in)
#X2_lstm_input = K.cast(X2, tf.float32)

### defining seq2seq model ###
# define training encoder
encoder_outputs, state_h, state_c = LSTM(n_units, return_state=True)(encoder_inputs)
encoder_states = [state_h, state_c]

# define training decoder
decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# define training model
train_model = Model(inputs=[X1_in,X2_in], outputs=decoder_outputs)

optimizer = Adagrad(lr=learning_rate)
train_model.compile(optimizer=optimizer,
              loss='mean_squared_error',
              weighted_metrics=['accuracy'])
train_model.summary()
plot_model(train_model, to_file='train_model_plot.png', show_shapes=True, show_layer_names=True)
validation_data = ([X1_val, X2_val], Y_val)

train_model.fit([X1_train, X2_train],
          Y_train,
          epochs=epochs,
          steps_per_epoch = 1,
          validation_data = validation_data,
          validation_steps = 1,
          shuffle=False,  # Shuffling data means shuffling the whole graph
         )


### Predict results ###

#define inference encoder model
encoder_model = Model(X1_in, encoder_states)

encoder_model.compile(optimizer=optimizer,
              loss='mean_squared_error',
              weighted_metrics=['accuracy'])
encoder_model.summary()
plot_model(encoder_model, to_file='encoder_model_plot.png', show_shapes=True, show_layer_names=True)

#define inference decoder model
inference_decoder_inputs = Input(batch_shape=(1, 1, 1))
decoder_state_input_h = Input(batch_shape=(1, n_units))
decoder_state_input_c = Input(batch_shape=(1, n_units))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(inference_decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([inference_decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

decoder_model.compile(optimizer=optimizer,
              loss='mean_squared_error',
              weighted_metrics=['accuracy'])
decoder_model.summary()
plot_model(decoder_model, to_file='decoder_model_plot.png', show_shapes=True, show_layer_names=True)

def predict_sequence(infenc, infdec, source, n_steps, cardinality):
    """    
    infenc: Encoder model used when making a prediction for a new source sequence.
    infdec: Decoder model use when making a prediction for a new source sequence.
    source:Encoded source sequence.
    n_steps: Number of time steps in the target sequence.
    cardinality: The cardinality of the output sequence, e.g. the number of features, words, or characters for each time step.
    """
    sample_size = source.shape[0]
    # encode
    samples_state = infenc.predict(source, steps = 1)
    print(samples_state[0].shape)
    print(samples_state[1].shape)
    Y_hat = np.zeros((sample_size, timesteps_per_sample, 1))
    
    # start of sequence input
    target_seq = np.array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
    # collect predictions
    
    for sample in range(sample_size):
        output = list() #clearing output
        state_h = np.reshape(samples_state[0][sample], (1, 128))
        state_c = np.reshape(samples_state[1][sample], (1, 128))
        state = [state_h, state_c] #selecting the states (h,c) to pass to decoder
        for t in range(n_steps):
            # predict next queue length
            yhat, h, c = infdec.predict([target_seq] + state)
            # store prediction
            output.append(yhat[0,0,:])
            # update state
            state = [h, c]
            # update target sequence
            target_seq = yhat
        Y_hat[sample, :, :] = output
    return Y_hat


Y_hat = predict_sequence(encoder_model, decoder_model, X1_test, timesteps_per_sample, 1)
print('X1_test.shape:', X1_test.shape)
print('Y_hat.shape:', Y_hat.shape)
prediction = K.reshape(Y_hat, (int(Y_hat.shape[0])//N, timesteps_per_sample, N, 1))
print('prediction.shape:', prediction.shape)

#save models with architecture, weights, training config
train_model.save('models/train_model_all.h5')
encoder_model.save('models/encoder_model_all.h5')
decoder_model.save('models/decoder_model_all.h5')

#save train model as JSON
# serialize model to JSON
train_model_json = train_model.to_json()
with open("models/train_model.json", "w") as json_file:
    json_file.write(train_model_json)
# serialize weights to HDF5
train_model.save_weights("models/train_model_weights.h5")
print("Saved train model to disk")

#save encoder model as JSON
# serialize model to JSON
encoder_model_json = encoder_model.to_json()
with open("models/encoder_model.json", "w") as json_file:
    json_file.write(encoder_model_json)
# serialize weights to HDF5
encoder_model.save_weights("models/encoder_model_weights.h5")
print("Saved encoder model to disk")

#save decoder model as JSON
# serialize model to JSON
decoder_model_json = decoder_model.to_json()
with open("models/decoder_model.json", "w") as json_file:
    json_file.write(decoder_model_json)
# serialize weights to HDF5
decoder_model.save_weights("models/decoder_model_weights.h5")
print("Saved decoder model to disk")

#TODO: Implement postprocessing of data with reorder lanes and resample time and store in a pandas file