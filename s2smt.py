
from model import LanguageModel


class S2SMTModel(LanguageModel):
    def load_data(self, debug=False):
        pass

    def add_placeholders(self):
        pass

    def add_embedding(self):
        pass

    def add_encoding(self):
        pass

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
