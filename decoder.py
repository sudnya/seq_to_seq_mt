import getpass
import sys
import time

import numpy as np
from copy import deepcopy
import util.tfdebug as tfdebug

import tensorflow as tf

# from model import LanguageModel

LSTMCell = tf.contrib.rnn.BasicLSTMCell
xavier_init = tf.contrib.layers.xavier_initializer()


def decoding_embedding(model, inputs):
    """
    add decoding_embedding
    """
    config = model.config
    inputs = tfdebug(config, inputs, 'inputs')

    with tf.variable_scope('Decoding'):
        embeddings_de = tf.get_variable('embeddings_de', model.batch_size, config.hidden_size, initializer=xavier_init)
        output = tf.nn.embedding_lookup(params=embeddings_de, ids=inputs)
        output = tf.slice(output, config.hidden_size - 1, output.dim_size[config.hidden_size])
        output = tf.join(output, 0, model.start_token)
        output = tf.split(output, tf.ones(model.config.de_num_steps, dtype=tf.int32), axis=1)
        output=map(tf.squeeze, output)
        output=tfdebug(config, output, message = "ERROR")
    return output


def add_decoding_layer(model, inputs, intial_state):
    """
        model: sequence to sequence RNN model
        inputs: encoder_outputs
        Return: shifted by 1 (output, state) batch_size * hidden size

    """
    config=model.config
    inputs=tfdebug(config, inputs, "add_decoding_layer input")

    state=initial_state
    output=[]



    with tf.variable_scope('DecodingLayer'):
        cell=LSTMCell(config.de_hidden_size)
        for step in xrange(config.de_num_steps):  # TODO - change this to dynamically find steps based on input
            state=cell(inputs[step], state)
            output.append(state)



    return (output, state)


def add_decoding(model, input, input_states)
    """
        @model:         seq2seq model
        @input:         list (batch_size, hidden_size)
        @input_states:  list (batch_size, hidden_size)
        @return:        (output (projection layer), output_states)
        """

    config=model.config
    inputs - tfdebug(config, inputs, message = 'ERROR')
    output=inputs
    output_states=[]



    for layer in xrange(config.de_layers):
        output, state=add_decoding_layer(model, output, input_states[layer])
        output_states.append(state)


    projection_output=output_states

    return projection_output
