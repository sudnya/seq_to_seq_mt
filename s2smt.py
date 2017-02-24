import tensorflow as tf
from model import LanguageModel

from encoder import add_embedding, add_encoding


class S2SMTModel(LanguageModel):
    def load_data(self, debug=False):
        pass

    def add_placeholders(self):
        config = self.config
        self.input_placeholder = tf.placeholder(config.enc_dtype, shape=(None, self.config.num_steps), name='input')
        self.labels_placeholder = tf.placeholder(config.enc_dtype, shape=(None, self.config.num_steps), name='labels')
        self.dropout_placeholder = tf.placeholder(config.dtype, name='dropout')


    def add_embedding(self):
        return add_embedding(self, self.input_placeholder)

    def add_encoding(self, inputs):
        return add_encoding(self, inputs)

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
