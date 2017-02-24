import tensorflow as tf
from model import LanguageModel

from encoder import add_embedding, add_encoding
from DataLoader import DataLoader


class S2SMTModel(LanguageModel):

    def __init__(self):
        self.en_initial_states = None


    def load_data(self, debug=False):
        cfg           = Config()
        data_loader   = DataLoader(cfg)
        
        self.en_train = data_loader.src_encoded_train
        self.en_dev   = data_loader.src_encoded_dev
        self.en_test  = data_loader.src_encoded_test

        self.de_train = data_loader.tgt_encoded_train
        self.de_dev   = data_loader.tgt_encoded_dev
        self.de_test  = data_loader.tgt_encoded_test

        if debug:
            num_debug     = 1024
            self.en_train = self.en_train[:num_debug]
            self.en_dev   = self.en_dev[:num_debug]
            self.en_test  = self.en_test[:num_debug]

            self.de_train = self.de_train[:num_debug]
            self.de_dev   = self.de_dev[:num_debug]
            self.de_test  = self.de_test[:num_debug]

        
    def add_placeholders(self):
        config = self.config
        self.input_placeholder = tf.placeholder(config.input_dtype, shape=(None, self.config.num_steps), name='input')
        self.labels_placeholder = tf.placeholder(config.input_dtype, shape=(None, self.config.num_steps), name='labels')
        self.dropout_placeholder = tf.placeholder(config.dtype, name='dropout')


    def add_embedding(self):
        return add_embedding(self, self.input_placeholder)

    def add_encoding(self, inputs):
        config = self.config
        if not self.en_initial_states:
            self.en_initial_states = [tf.zeros([config.batch_size, config.en_hidden_size], dtype=config.dtype) for x in xrange(config.en_layers)]
        return add_encoding(self, inputs, self.en_initial_states)

    def add_decoding(self):
        pass

    def add_attention(self):
        pass


    def add_training_op(self, loss):
        pass


    def add_model(self, inputs):
        pass


    def run_epoch(self, session, data, train_op=None, verbose=10):
        pass

def generate_text(session, model, config, starting_text='<eos>',
              stop_length=100, stop_tokens=None, temp=1.0):

    pass


def _test_S2SMTModel():
    pass
