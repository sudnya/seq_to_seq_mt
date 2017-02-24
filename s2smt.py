import tensorflow as tf
from model import LanguageModel
from config import Config
from encoder import add_embedding, add_encoding
from encoder import add_decoding
from data_loader import DataLoader

sequence_loss = tf.contrib.seq2seq.sequence_loss


class S2SMTModel(LanguageModel):

    def __init__(self):
        self.en_initial_states = None
        self.config = Config()

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
            self.en_dev   = self.en_dev[:num_debug]
            self.en_test  = self.en_test[:num_debug]

            self.de_train = self.de_train[:num_debug]
            self.de_dev   = self.de_dev[:num_debug]
            self.de_test  = self.de_test[:num_debug]

        self.en_vocab_size = data_loader.en_vocab_size + 1
        self.de_vocab_size = data_loader.de_vocab_size + 1

        self.start_token = self.de_vocab_size - 1

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(self.config.input_dtype, shape=[None, None], name='input')
        self.labels_placeholder = tf.placeholder(self.config.input_dtype, shape=[None, None], name='labels')
        self.num_steps_placeholder = tf.placeholder(tf.int32, name='num_steps')
        self.dropout_placeholder = tf.placeholder(self.config.dtype, name='dropout')

    def add_embedding(self):
        return add_embedding(self, self.input_placeholder)

    def add_encoding(self, inputs):
        initial_states = [tf.zeros([self.config.batch_size, self.config.hidden_size], dtype=self.config.dtype) for x in xrange(self.config.en_layers)]
        return add_encoding(self, inputs, initial_states)

    def add_decoding(self, encode_final_state):
        initial_states = [tf.zeros([self.config.batch_size, self.config.hidden_size], dtype=self.config.dtype) for x in xrange(self.config.de_layers)]
        initial_states[0] = en_final_state
        return add_decoding(self, inputs, initial_states)

    def add_attention(self):
        pass

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        train_op  = optimizer.minimize(loss)

    def create_feed_dict(self, input_batch, label_batch):
        """Creates the feed_dict for training the given step.
        Args:
          input_batch: A batch of input data.
          label_batch: A batch of label data.
        Returns: feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {}

        feed_dict[self.input_placeholder] = input_batch
        # only in train mode will we have labels provided
        if label_batch is not None:
            feed_dict[self.labels_placeholder] = label_batch

        return feed_dict

    def add_model(self, inputs):
        """Implements core of model that transforms input_data into predictions.
        The core transformation for this model which transforms a batch of input
        data into a batch of predictions.

        Args: input_data: A tensor of shape (batch_size, n_features).
        Returns: out: A tensor of shape (batch_size, n_classes)
        """
        outputs = []

        with tf.variable_scope('S2SMT') as scope:
            en_output, en_final_states = self.add_encoding(self.add_embedding(self.source_placeholder))
            de_output, de_final_states = self.add_decoding(self.labels_placeholder)

        return outputs

    def add_loss_op(self, pred):
        target_labels = tf.reshape(self.labels_placeholder, [self.config.batch_size, self.num_steps_placeholder])
        weights       = tf.ones([self.config.batch_size, self.num_steps_placeholder])
        pred_logits   = tf.reshape(pred, [self.config.batch_size, self.num_steps_placeholder, self.de_vocab_size])
        loss          = sequence_loss(logits=pred_logits, targets=target_labels, weights=weights)
        self.sMax     = tf.nn.softmax(pred_logits)

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
        pass

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


def _test_S2SMTModel():
    pass
