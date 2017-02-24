import tensorflow as tf
import tf.contrib.rnn.BasicLSTMCell as LSTMCell
import util.tfdebug as tfdebug

xavier_init = tf.contrib.layers.xavier_initializer()

# TODO set hidden_size equal to hidden_size and fixed for all layers (future proof for residuals)

def add_embedding(model, inputs):
    """
        @model:             the s2smt model
        @inputs:  (int32)   batch_size x num_steps
        @return:  (config.dtype)  list (num_steps) of batch_size x hidden_size
    """
    config = model.config
    inputs = tfdebug(config, inputs, message='ADD EMBEDDING IN')

    with tf.variable_scope('Embedding'):
        w2v = tf.get_variable('w2v', [model.vocab_size, config.en_hidden_size], initializer=xavier_init)
        output = tf.nn.embedding_lookup(params=w2v, ids=inputs)
        output = tf.split(output, tf.ones(config.en_num_steps, dtype=tf.int32), axis=1)
        output = map(tf.squeeze, output)

        output = tfdebug(config, output, message='ADD EMBEDDING IN')

    return output


def add_encoding_layer(model, inputs, initial_state):
    """
        @model:             the s2smt model
        @inputs:            (config.dtype)    list (num_steps) of batch_size x hidden_size
        @initial_state:     (config.dtype)    batch_size x hidden_size
        @return:            (output, state)
                            output: (config.dtype)    list (num_steps) of batch_size x hidden_size
                            state:  (config.dtype)    final state for this layer batch_size x hidden_size
    """
    config = model.config
    inputs = tfdebug(config, inputs, message='ADD ENCODING IN')

    state = initial_state
    output = []

    with tf.variable_scope('Encoding'):
        for step in xrange(config.en_num_steps):
            cell = LSTMCell(config.en_hidden_size)
            state = cell(inputs[step], state)
            output.append(state)

        output = tfdebug(config, inputs, message='ADD ENCODING OUT')

    return (output, state)

def add_encoding(model, inputs, inputs_states):
    """
        @model:     the s2smt model
        @inputs:    (int32) batch_size x num_steps
        @return:    (output, last_final_state)
                    output: (model.config.dtype)    list (num_steps) of batch_size x hidden_size (config.dtype)
                    state:  (model.config.dtype)    final state for the last layer batch_size x hidden_size
    """
    config = model.config
    if not hasattr(model, 'en_initial_states'):
        model.en_inital_states = [tf.zeros([config.batch_size, config.en_hidden_size], dtype=model.dtype) for x in xrange(config.en_layers)]

    inital_states = model.en_inital_states
    final_states = []
    output = add_embedding(model, inputs)

    # TODO
    # bidirectional first layer

    for layer in xrange(config.en_layers):
        output, state = add_encoding_layer(model, output, inital_states[layer])
        final_states.append(state)

    model.en_final_states = final_states
    return output, state

def _test():
    pass
