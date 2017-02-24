import getpass
import sys
import time

import numpy as np
from copy import deepcopy


import tensorflow as tf
import tf.contrib.rnn.BasicLSTMCell as LSTMCell
#from model import LanguageModel


xavier_init = tf.contrib.layers.xavier_initializer()

def add_decoding_layer(model, inputs, intial_state): 
    """
        model: sequence to sequence RNN model 
        inputs: encoder context/ 
        Retrun: (output, state) batch_size * hidden size
        """  
   
    with tf.variable_scope('Decoding') :
        state = pass
        outputs = []
        for inputs in enumerate(decoder_outputs):

def add_decoding (model, input, input_states)
    ""
        outputs: (output, output_states) 
        """

def add_projection(): 
    pass 
def test_decoder(): 
    pass
