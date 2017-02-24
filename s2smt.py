import sys
import tensorflow as tf
import numpy as np

from data_iterator import data_iterator
from model import LanguageModel
from data_loader import DataLoader
from encoder import add_encoding
from encoder import add_embedding
from decoder import add_decoding

from utils import calculate_perplexity

sequence_loss = tf.contrib.seq2seq.sequence_loss


def _make_lstm_initial_states(config):
    return (tf.zeros([config.batch_size, config.hidden_size], dtype=config.dtype),
            tf.zeros([config.batch_size, config.hidden_size], dtype=config.dtype))


class S2SMTModel(LanguageModel):

    def __init__(self, config):
        self.en_initial_states = None
        self.config = config
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
            num_debug = 512
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
        self.en_placeholder = tf.placeholder(self.config.tf_raw_dtype, shape=[None, None], name='input')
        self.de_placeholder = tf.placeholder(self.config.tf_raw_dtype, shape=[None, None], name='labels')
        self.dropout_placeholder = tf.placeholder(self.config.dtype, name='dropout')

    def add_embedding(self):
        return add_embedding(self, self.en_placeholder)

    def add_encoding(self, source):
        initial_states = [_make_lstm_initial_states(self.config) for x in xrange(self.config.layers)]
        return add_encoding(self, source, initial_states)

    def add_decoding(self, target, encoder_final_state):
        initial_states = [_make_lstm_initial_states(self.config) for x in xrange(self.config.layers)]
        initial_states[0] = encoder_final_state
        return add_decoding(self, target, initial_states)

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

    def train_step(self):
        pass

    def run_epoch(self, session, en_data, de_data, train_op=None, verbose=10):
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
            dp = 1.

        total_steps = sum(1 for x in data_iterator(en_data, de_data, config.batch_size, config.np_raw_dtype))
        total_loss = []

        state = self.initial_state.eval()

        for step, (x, y) in enumerate(data_iterator(en_data, de_data, config.batch_size, config.np_raw_dtype)):
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


def translate_sentence(session, model, config, en_text='<eos>', stop_length=100, stop_tokens=None, temp=1.0):
    pass
