import tensorflow as tf
import util

tfdebug = util.tfdebug

LSTMCell = tf.contrib.rnn.BasicLSTMCell

xavier_init = tf.contrib.layers.xavier_initializer()

# TODO set hidden_size equal to hidden_size and fixed for all layers (future proof for residuals)


def add_embedding(model, inputs):
    """
        @model:
        @inputs:        (int32)             batch_size x num_steps
        @return:        (config.dtype)      list (num_steps) x batch_size x hidden_size
    """
    config = model.config
    inputs = tfdebug(config, inputs, message='ADD EMBEDDING IN')

    with tf.variable_scope('Embedding'):
        w2v = tf.get_variable('w2v', [model.en_vocab_size, config.en_hidden_size], initializer=xavier_init)
        output = tf.nn.embedding_lookup(params=w2v, ids=inputs)
        output = tf.split(output, tf.ones(model.num_steps, dtype=tf.int32), axis=1)
        output = map(tf.squeeze, output)

        output = tfdebug(config, output, message='ADD EMBEDDING OUT')

    return output


def add_encoding_layer(model, inputs, initial_state):
    """
        @model:
        @inputs:        (config.dtype)      list (num_steps) of batch_size x hidden_size
        @initial_state: (config.dtype)      batch_size x hidden_size
        @return:
            output:     (config.dtype)      list (num_steps) of batch_size x hidden_size
            state:      (config.dtype)      final state for this layer batch_size x hidden_size
    """
    config = model.config
    inputs = tfdebug(config, inputs, message='ADD ENCODING IN')

    state = initial_state
    output = []

    with tf.variable_scope('Encoding'):
        for step in xrange(model.num_steps):
            cell = LSTMCell(config.en_hidden_size)
            state = cell(inputs[step], state)
            output.append(state)

        output = tfdebug(config, inputs, message='ADD ENCODING OUT')

    return (output, state)


def add_encoding(model, inputs, initial_states):
    """
        @model:
        @inputs:        (config.dtype)    list (num_steps) x batch_size x hidden_size
        @initial_state: (config.dtype)    list (layers) x batch_size x hidden_size
        @return:
            output:     (config.dtype)    list (num_steps) x batch_size x hidden_size
            states:     (config.dtype)    list (layers) x batch_size x hidden_size
    """
    config = model.config
    output = inputs
    final_states = []

    # TODO
    # bidirectional first layer

    for layer in xrange(config.en_layers):
        output, state = add_encoding_layer(model, output, initial_states[layer])
        final_states.append(state)

    return output, final_states

def _test():
    pass
