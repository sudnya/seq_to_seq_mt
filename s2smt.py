import sys
import tensorflow as tf
import numpy as np


from data_iterator import data_iterator
from model import LanguageModel
from data_loader import DataLoader
from encoder import add_encoding
from encoder import add_embedding
from decoder import add_decoding

# from util import calculate_perplexity


sequence_loss = tf.contrib.seq2seq.sequence_loss


def _make_lstm_initial_states(config):
    return tf.tuple([tf.zeros([config.batch_size, config.hidden_size], dtype=config.dtype),
            tf.zeros([config.batch_size, config.hidden_size], dtype=config.dtype)])


class S2SMTModel(LanguageModel):

    def __init__(self, config):
        self.en_initial_states = None
        self.config = config
        self.load_data(debug=False)
        self.add_placeholders()
        self.outputs = self.add_model()

        self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]

        self.calculate_loss = self.add_loss_op(self.outputs)
        self.train_step = self.add_training_op(self.calculate_loss)

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

        config = self.config

        config.en_vocab_size = data_loader.en_vocab_size + 1
        config.de_vocab_size = data_loader.de_vocab_size + 2

        config.en_pad_token = config.en_vocab_size - 1

        config.de_pad_token = config.de_vocab_size - 1
        config.start_token = config.de_vocab_size - 2

    def add_placeholders(self):
        self.en_placeholder = tf.placeholder(self.config.tf_raw_dtype, shape=[None, None], name='en_placeholder')
        self.de_ref_placeholder = tf.placeholder(self.config.tf_raw_dtype, shape=[None, None], name='de_ref_placeholder')
        self.de_pred_placeholder = tf.placeholder(self.config.tf_raw_dtype, shape=[None, None], name='de_pred_placeholder')
        self.dropout_placeholder = tf.placeholder(self.config.dtype, name='dropout')

    def add_embedding(self):
        return add_embedding(self, self.en_placeholder)

    def add_encoding(self, source):
        initial_states = [_make_lstm_initial_states(self.config) for x in xrange(self.config.layers)]
        return add_encoding(self, source, initial_states)

    def add_decoding(self, encoder_final_state, de_ref):
        initial_states = [_make_lstm_initial_states(self.config) for x in xrange(self.config.layers)]
        initial_states[0] = encoder_final_state
        return add_decoding(self, initial_states, de_ref)

    def add_attention(self):
        pass

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr, beta1=self.config.beta1, beta2=self.config.beta2)
        train_op = optimizer.minimize(loss)

    def create_feed_dict(self, en_batch, de_ref_batch, de_pred_batch):
        """
            @en_batch       (config.np_raw_dtype)     numpy [batch_size x en_num_steps]
            @de_ref_batch   (config.np_raw_dtype)     numpy [batch_size x de_num_steps]
            @de_pred_batch  (config.np_raw_dtype)     numpy [batch_size x de_num_steps]
            @return         (dictionary)              feed_dict
        """
        feed_dict = {}
        feed_dict[self.en_placeholder] = en_batch
        feed_dict[self.de_ref_placeholder] = de_ref_batch
        # only in train mode will we have decoded batches
        if de_pred_batch is not None:
            feed_dict[self.de_pred_placeholder] = de_pred_batch
            feed_dict[self.dropout_placeholder] = self.config.dropout
        return feed_dict

    def add_model(self):
        """
            @return         (config.dtype)            tensor [batch_size x hidden_size]
        """
        with tf.variable_scope('S2SMT') as scope:
            embed = self.add_embedding()
            en_output, en_final_state = self.add_encoding(embed)
            # TODO add Attention
            de_output = self.add_decoding(en_final_state, self.de_ref_placeholder)


        return de_output

    def add_loss_op(self, pred_logits):
        # pred_logits input is list (num_steps) of Tensor[batch_size x de_vocab_size]
        # pred_logits should be Tensor[batch_size x de_num_steps x de_vocab_size]
        pred_logits = tf.stack(pred_logits, axis=1)

        # targets should be Tensor[batch_size x de_num_steps]
        targets = self.de_pred_placeholder

        weights = tf.ones([self.config.batch_size, self.config.de_num_steps])

        loss = sequence_loss(logits=pred_logits, targets=targets, weights=weights)

        self.last_softmax = tf.nn.softmax(pred_logits)

        tf.add_to_collection('total_loss', loss)

        return loss


    def run_epoch(self, session, en_data, de_data, train_op=None, verbose=10):
        """Runs an epoch of training.  Trains the model for one-epoch.
        Args:
          sess: tf.Session() object
          input_data: np.ndarray of shape (n_samples, n_features)
          input_labels: np.ndarray of shape (n_samples, n_classes)
        Returns: average_loss: scalar. Average minibatch loss of model on epoch.
        """
        config = self.config
        dp     = config.dropout

        if not train_op:
            train_op = tf.no_op()
            dp = 1.0

        total_steps = sum(1 for x in data_iterator(en_data, de_data, config.batch_size, config.en_pad_token, config.de_pad_token, config.start_token, config.np_raw_dtype))

        total_loss = []

        state = self.initial_state.eval()


        for step, (en_batch, de_ref_batch, de_prod_batch) in enumerate(data_iterator(en_data, de_data, config.batch_size, config.en_pad_token, config.de_pad_token, config.start_token, config.np_raw_dtype)):
            # We need to pass in the initial state and retrieve the final state to give
            # the RNN proper history
            feed = {self.en_placeholder: en_batch,
                    self.de_ref_placeholder: de_ref_batch,
                    self.de_prod_placeholder: de_prod_batch,
                    self.dropout_placeholder: dp}

            loss, state, _ = session.run([self.calculate_loss, self.final_state, train_op], feed_dict=feed)
            total_loss.append(loss)

            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : pp = {}'.format(step, total_steps, np.exp(np.mean(total_loss))))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')

        return np.exp(np.mean(total_loss))

    def fit(self, session, X, y):
        """Fit model on provided data.
        Args:
          sess: tf.Session()
          input_data: np.ndarray of shape (n_samples, n_features)
          input_labels: np.ndarray of shape (n_samples, n_classes)
        Returns: losses: list of loss per epoch
        """
        losses = []

        if np.any(y):
            data = data_iterator(X, y, batch_size=self.config.batch_size, label_size=self.config.label_size, shuffle=False)

        for step, (x, y) in enumerate(data):
            feed = self.create_feed_dict(input_batch=x, dropout=dp)
            if np.any(y):
                feed[self.labels_placeholder] = y
                loss, preds = session.run([self.loss, self.predictions], feed_dict=feed)
                losses.append(loss)

        return losses

    def predict(self, session, en_data, y=None):
        """Make predictions from the provided model.
        Args:
          sess: tf.Session()
          input_data: np.ndarray of shape (n_samples, n_features)
          input_labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
          average_loss: Average loss of model.
          predictions: Predictions of model on input_data
        """
        config = self.config
        # We deactivate dropout by setting it to 1
        dp = 1
        losses = []
        predictions = []

        # train or test mode?
        if np.any(y):
            data = data_iterator(X, y, config.batch_size, en_data, de_ref_data)
        else:
            data = data_iterator(X, config.batch_size, en_data, de_ref_data)

        for step, (x, y) in enumerate(data):
            feed = self.create_feed_dict(input_batch=x, dropout=dp)
            if np.any(y):
                feed[self.labels_placeholder] = y
                loss, preds = session.run([self.loss, self.predictions], feed_dict=feed)
                losses.append(loss)
            else:  # no loss
                preds = session.run(self.predictions, feed_dict=feed)
                predicted_indices = preds.argmax(axis=1)
                predictions.extend(predicted_indices)

        return np.mean(losses), predictions


def translate_text(session, model, config, starting_text='<eos>',
                   stop_length=100, stop_tokens=None, temp=1.0):
    """Generate text from the model.

    Hint: Create a feed-dictionary and use sess.run() to execute the model. Note
          that you will need to use model.initial_state as a key to feed_dict
    Hint: Fetch model.final_state and model.predictions[-1]. (You set
          model.final_state in add_model() and model.predictions is set in
          __init__)
    Hint: Store the outputs of running the model in local variables state and
          y_pred (used in the pre-implemented parts of this function.)

    Args:
      session: tf.Session() object
      model: Object of type RNNLM_Model
      config: A Config() object
      starting_text: Initial text passed to model.
    Returns:
      output: List of word idxs
    """
    state = model.initial_state.eval()
    # Imagine tokens as a batch size of one, length of len(tokens[0])
    tokens = [model.vocab.encode(word) for word in starting_text.split()]

    print tokens

    # for i in xrange(stop_length):
    #     # YOUR CODE HERE
    #
    #     feed = {
    #         model.input_placeholder: tokens[i - 1],
    #         model.initial_state: state,
    #         model.dropout_placeholder: 1
    #     }
    #     loss, y_pred, _ = session.run(
    #         [model.calculate_loss, model.final_state, tf.no_op()], feed_dict=feed)
    #
    #     # END YOUR CODE
    #     next_word_idx = sample(y_pred[0], temperature=temp)
    #     tokens.append(next_word_idx)
    #     if stop_tokens and model.vocab.decode(tokens[-1]) in stop_tokens:
    #         break
    # output = [model.vocab.decode(word_idx) for word_idx in tokens]
    # return output


def translate_sentence(session, model, config, en_text='<eos>', stop_length=100, stop_tokens=None, temp=1.0):
    pass
