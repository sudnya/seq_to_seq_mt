import tensorflow as tf
import util.tfdebug as tfdebug
import tf.contrib.rnn.BasicLSTMCell as LSTMCell

xavier_init = tf.contrib.layers.xavier_initializer()

# TODO set hidden_size equal to hidden_size and fixed for all layers (future proof for residuals)

def add_embedding(model, inputs):
    # inputs should be in shape of batch_size x num_steps x encoding_hidden_size
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
        @model:             the model
        @inputs:            list (num_steps) of batch_size x hidden_size (any float type)
        @initial_state:     batch_size x hidden_size (same as inputs float type)
        @return:            (output, state)
                            output: list (num_steps) of batch_size x hidden_size (same as inputs float type)
                            state: final sate batch_size x hidden_size
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

def add_encoding(model, inputs):
    config = model.config
    if not hasattr(model, 'en_initial_states'):
        model.en_inital_states = [tf.zeros([config.batch_size, config.en_hidden_size], dtype=model.dtype) for x in xrange(config.en_layers)]

    output = add_embedding(model, inputs)



    output, state = add_encoding_layer(model, output, state)

    states = model.en_initial_states
