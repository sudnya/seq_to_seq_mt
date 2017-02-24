import tensorflow as tf
from model import LanguageModel

from encoder import add_embedding, add_encoding
from DataLoader import DataLoader


class S2SMTModel(LanguageModel):

    def __init__(self):
        self.en_initial_states = None
        self.config = Config()

    def load_data(self, debug=False):
        data_loader = DataLoader(self.config)

        self.en_train = data_loader.src_encoded_train
        self.en_dev = data_loader.src_encoded_dev
        self.en_test = data_loader.src_encoded_test

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

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(self.config.input_dtype, shape=(None, self.config.num_steps), name='input')
        self.labels_placeholder = tf.placeholder(self.config.input_dtype, shape=(None, self.config.num_steps), name='labels')
        self.dropout_placeholder = tf.placeholder(self.config.dtype, name='dropout')

    def add_embedding(self):
        return add_embedding(self, self.input_placeholder)

    def add_encoding(self, inputs):
        if not self.en_initial_states:
            self.en_initial_states = [tf.zeros([self.config.batch_size, self.config.en_hidden_size], dtype=self.config.dtype) for x in xrange(self.config.en_layers)]
        return add_encoding(self, inputs, self.en_initial_states)

    def add_decoding(self):
        """
            @return (output, final_states)
        """
        pass

    def add_attention(self):
        pass

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        train_op = optimizer.minimize(loss)

    def add_model(self, input_data):
        """
            @input_data
        """
        pass

    def run_epoch(self, session, data, train_op=None, verbose=10):
        pass


def generate_text(session, model, config, starting_text='<eos>', stop_length=100, stop_tokens=None, temp=1.0):
    pass


def _test_S2SMTModel():
    pass
