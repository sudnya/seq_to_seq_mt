import tensorflow as tf

LSTMCell = tf.contrib.rnn.BasicLSTMCell
xavier_init = tf.contrib.layers.xavier_initializer()


def add_embedding(model, inputs, step=False):
    """
        @model:
        @inputs:        (int32)             batch_size x num_steps
        @step:          (bool)              batch_size x 1 if true
        @return:        (config.dtype)      list (num_steps) x batch_size x hidden_size
    """
    config = model.config

    with tf.variable_scope('DecodingEmbeddingLayer'):
        w2v = tf.get_variable('w2v', [config.de_vocab_size, config.batch_size], initializer=xavier_init)
        output = tf.nn.embedding_lookup(params=w2v, ids=inputs)
        if not step:
            output = tf.split(output, tf.ones(config.de_num_steps, dtype=tf.int32), axis=1)
            output = map(tf.squeeze, output)

    return output


def _add_projection(model, inputs):
    """
        @model: 
        @inputs:        (config.dtype)      list (num_steps) of batch size x hidden_size
        @return:        (config.dytpe)      list (num_steps) of batch_size x de_vocab_size

    """
    with tf.variable_scope("ProjectionLayer"): 
        U = tf.get_variable("U", (config.hidden_size, config.de_vocab_size), dtype=tf.float32)
        b_2 = tf.Variable(tf.zeros(config.de_vocab_size), dtype=tf.float32)

        outputs = []

        for i in range (config.de_num_steps): 
            outputs.append(tf.matmul(inputs[i], U) + b_2)
    return outputs


def _add_decoding_layer(model, inputs, initial_state, layer):
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

    with tf.variable_scope('DecodingLayer' + str(layer)):
        cell = LSTMCell(config.hidden_size)
        for step in xrange(config.de_num_steps):
            state = cell(inputs[step], state)
            output.append(state)

    return (output, state)


def add_decoding(model, inputs, initial_states):
    """
        @model:         seq2seq model
        @input:         list (batch_size, hidden_size)
        @input_states:  list (batch_size, hidden_size)
        @return:        (output (projection layer), output_states)
    """

    config = model.config
    output = inputs

    if config.train:
        output = add_embedding(model, inputs)

    for layer in xrange(config.layers):
        output, state = _add_decoding_layer(model, output, initial_states[layer], layer)

    return _add_projection(model, output)
