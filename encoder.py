import tensorflow as tf

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
    #print "inputs are: ", inputs.get_shape()

    with tf.variable_scope('EncodingEmbeddingLayer'):
        w2v = tf.get_variable('w2v', [config.en_vocab_size, config.hidden_size], initializer=xavier_init)
        #print "w2v ", w2v.get_shape()
        output = tf.nn.embedding_lookup(params=w2v, ids=inputs)
        #print "after embed look up ", output.get_shape()
        output = tf.split(output, tf.ones(config.en_num_steps, dtype=tf.int32), axis=1)
        #print "before sqz ", len(output), " xxxx ", output[0].get_shape()
        output = map(lambda x : tf.squeeze(x, axis=1), output)
        #print "after squuezed shape ", len(output) , " --> " , output[0].get_shape()

    return output


def _add_encoding_layer(model, inputs, initial_state, layer):
    """
        @model:
        @inputs:        (config.dtype)      list (num_steps) of batch_size x hidden_size
        @initial_state: (config.dtype)      batch_size x hidden_size
        @return:
            output:     (config.dtype)      list (num_steps) of batch_size x hidden_size
            state:      (config.dtype)      final state for this layer batch_size x hidden_size
    """
    config = model.config
    state = initial_state
    output = []

    with tf.variable_scope('EncodingLayer' + str(layer)):
        cell = LSTMCell(config.hidden_size)
        for step in xrange(config.en_num_steps):
            #print inputs[step].get_shape()
            state = cell(inputs[step], state)
            output.append(state)

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

    # TODO
    # bidirectional first layer
    #

    for layer in xrange(config.layers):
        output, state = _add_encoding_layer(model, output, initial_states[layer], layer)

    return output, state
