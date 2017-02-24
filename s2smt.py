import argparse
import logging
import tensorflow as tf
import numpy as np
import getpass
import sys
import time
from copy import deepcopy

from utils import calculate_perplexity

from config import Config
from model import LanguageModel
from data_loader import DataLoader
import encoder
import decoder

sequence_loss = tf.contrib.seq2seq.sequence_loss


def _make_lstm_initial_states(config):
    return (tf.zeros([config.batch_size, config.hidden_size], dtype=config.dtype),
            tf.zeros([config.batch_size, config.hidden_size], dtype=config.dtype))


class S2SMTModel(LanguageModel):

    def __init__(self):
        self.en_initial_states = None
        self.config = Config()
        self.add_placeholders()

    def load_data(self, debug=False):
        data_loader = DataLoader(self.config)

        self.en_train = data_loader.src_encoded_train_rev
        self.en_dev = data_loader.src_encoded_dev_rev
        self.en_test = data_loader.src_encoded_test_rev

        self.de_train = data_loader.tgt_encoded_train
        self.de_dev = data_loader.tgt_encoded_dev
        self.de_test = data_loader.tgt_encoded_test

        if debug:
            num_debug = 1024
            self.en_train = self.en_train[:num_debug]
            self.en_dev = self.en_dev[:num_debug]
            self.en_test = self.en_test[:num_debug]

            self.de_train = self.de_train[:num_debug]
            self.de_dev = self.de_dev[:num_debug]
            self.de_test = self.de_test[:num_debug]

        self.en_vocab_size = data_loader.en_vocab_size + 1
        self.de_vocab_size = data_loader.de_vocab_size + 1

        self.start_token = self.de_vocab_size - 1

    def add_placeholders(self):
        self.en_placeholder = tf.placeholder(self.config.input_dtype, shape=[None, None], name='input')
        self.de_placeholder = tf.placeholder(self.config.input_dtype, shape=[None, None], name='labels')
        self.dropout_placeholder = tf.placeholder(self.config.dtype, name='dropout')

    def add_embedding(self):
        return encoder.add_embedding(self, self.en_placeholder)

    def add_encoding(self, source):
        initial_states = [_make_lstm_initial_states(self.config) for x in xrange(self.config.layers)]
        return encoder.add_encoding(self, source, initial_states)

    def add_decoding(self, target, encoder_final_state):
        initial_states = [_make_lstm_initial_states(self.config) for x in xrange(self.config.layers)]
        initial_states[0] = encoder_final_state
        return decoder.add_decoding(self, target, initial_states)

    def add_attention(self):
        pass

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr, beta1=self.config.beta1, beta2=self.config.beta2)
        train_op = optimizer.minimize(loss)

    def create_feed_dict(self, input_batch, label_batch):
        """Creates the feed_dict for training the given step.
        Args:
          input_batch: A batch of input data.
          label_batch: A batch of label data.
        Returns: feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {}

        feed_dict[self.en_placeholder] = input_batch
        # only in train mode will we have labels provided
        if label_batch is not None:
            feed_dict[self.de_placeholder] = label_batch

        return feed_dict

    def add_model(self, inputs):
        """Implements core of model that transforms input_data into predictions.
        The core transformation for this model which transforms a batch of input
        data into a batch of predictions.

        Args: input_data: A tensor of shape (batch_size, n_features).
        Returns: out: A tensor of shape (batch_size, n_classes)
        """
        with tf.variable_scope('S2SMT') as scope:
            en_output, en_final_state = self.add_encoding(self.add_embedding(self.en_placeholder))

            de_output = self.add_decoding(self.de_placeholder, en_final_state)

        return de_output

    def add_loss_op(self, pred):
        target_labels = tf.reshape(self.de_placeholder, [self.config.batch_size, self.num_steps_placeholder])
        weights = tf.ones([self.config.batch_size, self.num_steps_placeholder])
        pred_logits = tf.reshape(pred, [self.config.batch_size, self.num_steps_placeholder, self.de_vocab_size])
        loss = sequence_loss(logits=pred_logits, targets=target_labels, weights=weights)
        self.sMax = tf.nn.softmax(pred_logits)

        tf.add_to_collection('total_loss', loss)

        return loss

    def run_epoch(self, session, data, train_op=None, verbose=10):
        """Runs an epoch of training.  Trains the model for one-epoch.
        Args:
          sess: tf.Session() object
          input_data: np.ndarray of shape (n_samples, n_features)
          input_labels: np.ndarray of shape (n_samples, n_classes)
        Returns: average_loss: scalar. Average minibatch loss of model on epoch.
        """
        config = self.config
        dp = config.dropout

        if not train_op:
            train_op = tf.no_op()
            dp = 1
        total_steps = sum(1 for x in ptb_iterator(data, config.batch_size, config.num_steps))
        total_loss = []

        state = self.initial_state.eval()

        for step, (x, y) in enumerate(ptb_iterator(data, config.batch_size, config.num_steps)):
            # We need to pass in the initial state and retrieve the final state to give
            # the RNN proper history
            feed = {self.input_placeholder: x,
                    self.labels_placeholder: y,
                    self.initial_state: state,
                    self.dropout_placeholder: dp}

            loss, state, _ = session.run(
                [self.calculate_loss, self.final_state, train_op], feed_dict=feed)
            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : pp = {}'.format(
                    step, total_steps, np.exp(np.mean(total_loss))))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')
        return np.exp(np.mean(total_loss))

    def fit(self, sess, input_data, input_labels):
        """Fit model on provided data.
        Args:
          sess: tf.Session()
          input_data: np.ndarray of shape (n_samples, n_features)
          input_labels: np.ndarray of shape (n_samples, n_classes)
        Returns: losses: list of loss per epoch
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def predict(self, sess, input_data, input_labels=None):
        """Make predictions from the provided model.
        Args:
          sess: tf.Session()
          input_data: np.ndarray of shape (n_samples, n_features)
          input_labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
          average_loss: Average loss of model.
          predictions: Predictions of model on input_data
        """
        raise NotImplementedError("Each Model must re-implement this method.")


def generate_text(session, model, config, starting_text='<eos>', stop_length=100, stop_tokens=None, temp=1.0):
    pass


def test_S2SMTModel():
    pass


def test_encoder():
    t_model = S2SMTModel()
    t_model.load_data()

    ref_num_steps = t_model.config.en_num_steps
    ref_batch_size = t_model.config.batch_size

    ref_hidden_size = t_model.config.hidden_size
    ref_layer_size = t_model.config.layers

    t_inputs = t_model.add_embedding()
    assert len(t_inputs) == ref_num_steps

    # 20  x  <unknown> so cannot be verified
    # print t_inputs[0].get_shape() , "woooo"
    #assert t_inputs[0].get_shape() == (ref_batch_size, ref_hidden_size)

    t_rnn_y, f_state = t_model.add_encoding(t_inputs)
    #assert len(t_rnn_y) == ref_num_steps
    #assert len(f_state) == ref_layer_size


def test_decoder():
    pass


def run_tests():
    test_encoder()
    test_decoder()
    test_S2SMTModel()


def main():
    parser = argparse.ArgumentParser(description="Sequence to sequence machine translation model")
    parser.add_argument("-v", "--verbose", default=False, action="store_true")

    parsedArguments = parser.parse_args()
    arguments = vars(parsedArguments)
    isVerbose = arguments['verbose']

    if isVerbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run_tests()


if __name__ == '__main__':
    main()
