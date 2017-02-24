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

    with tf.variable_scope('EmbeddingLayer'):
        w2v = tf.get_variable('w2v', [config.de_vocab_size, config.batch_size], initializer=xavier_init)
        output = tf.nn.embedding_lookup(params=w2v, ids=inputs)
        if not step:
            output = tf.split(output, tf.ones(config.de_num_steps, dtype=tf.int32), axis=1)
            output = map(tf.squeeze, output)

    return output


def _add_projection(model, inputs):
    pass


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

    if model.train:
        output = add_embedding(model, inputs)
        output.pop()
        token_embedding = tf.ones([config.batch_size, config.hidden_size], dtype=tf.int32) * model.start_token
        token_embedding = add_embedding(model, token_embedding, step=True)

        output.insert(0, token_embedding) 


    for layer in xrange(config.de_layers):
        output, state = _add_decoding_layer(model, output, initial_states[layer], layer)

    return _add_projection(model, output)
