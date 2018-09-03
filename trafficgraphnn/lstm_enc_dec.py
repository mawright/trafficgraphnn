#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 09:59:45 2018
Functions from "How to Develop an Encoder-Decoder Model for Sequence-to-Sequence
Prediction in Keras" from Jason Brownlee. 
Link: https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/
"""
import logging
from keras.layers import Input, Dense, LSTM
from keras.models import Model
import numpy as np

_logger = logging.getLogger(__name__)


# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_units):
    """
    The function takes 3 arguments, as follows:
    
    n_input: The cardinality of the input sequence, e.g. number of features, words, or characters for each time step.
    n_output: The cardinality of the output sequence, e.g. number of features, words, or characters for each time step.
    n_units: The number of cells to create in the encoder and decoder models, e.g. 128 or 256.
    
    The function then creates and returns 3 models, as follows:
    
    train: Model that can be trained given source, target, and shifted target sequences.
    inference_encoder: Encoder model used when making a prediction for a new source sequence.
    inference_decoder Decoder model use when making a prediction for a new source sequence.
    """
    # define training encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model

# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
    """
    The function below named predict_sequence() can be used after the model 
    is trained to generate a target sequence given a source sequence.
    
    This function takes 5 arguments as follows:

    infenc: Encoder model used when making a prediction for a new source sequence.
    infdec: Decoder model use when making a prediction for a new source sequence.
    source:Encoded source sequence.
    n_steps: Number of time steps in the target sequence.
    cardinality: The cardinality of the output sequence, e.g. the number of features, words, or characters for each time step.

    The function then returns a list containing the target sequence.
    """
    # encode
    state = infenc.predict(source)
    # start of sequence input
    target_seq = np.array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
    # collect predictions
    output = list()
    for t in range(n_steps):
        # predict next char
        yhat, h, c = infdec.predict([target_seq] + state)
        # store prediction
        output.append(yhat[0,0,:])
        # update state
        state = [h, c]
        # update target sequence
        target_seq = yhat
    return np.array(output)
