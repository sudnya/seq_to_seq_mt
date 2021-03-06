import tensorflow as tf
import numpy as np


class Config(object):
    """Holds model hyperparams and data information.
    """
    # Fixed datatype per model
    dtype = tf.float32
    np_raw_dtype = np.int32
    tf_raw_dtype = tf.int32

    # Realtime Adjustable
    debug = True  # switch off for production
    train = True  # switch off for testing

    # change on feeding
    seq_len = 20
    batch_size = 16

    # Fixed for encoder and decoder
    # embed_size is the same as hidden_size for later optimization
    hidden_size = 256
    layers = 1

    # Attention
    att_hidden_size = 200

    # Training Hyper Params
    max_epochs = 10
    early_stopping = 2
    dropout = 0.9
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    l2 = 0.001

    # Fixed language source and target
    source_lang = 'vi'
    target_lang = 'en'


    # Number of samples to load for training
    vocab_max_size = 20000
    train_samples = 16384
    dev_samples = 1024
    test_samples = 100

    # set at train time
    en_pad_token = None
    de_pad_token = None
    start_token  = None

class PredConfig(Config):
    """Holds model hyperparams and data information.
    """
    # Realtime Adjustable
    train = False  # switch off for testing
