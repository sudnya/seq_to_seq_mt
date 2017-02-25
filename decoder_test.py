
import argparse
import logging
import time
import tensorflow as tf

from config import Config
from s2smt import S2SMTModel
from s2smt import translate_sentence


def test_decoder(t_model, en_output):

    #ref_num_steps = t_model.config.de_num_steps
    #ref_batch_size = t_model.config.batch_size

    #ref_hidden_size = t_model.config.hidden_size
    #ref_layer_size = t_model.config.layers

    t_inputs = t_model.add_decoding(en_output, t_model.de_ref_placeholder)
    #assert len(t_inputs) == ref_num_steps
    return t_inputs
