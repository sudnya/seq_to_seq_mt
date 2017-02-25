import tensorflow as tf

BasicLSTMCell = tf.contrib.rnn.BasicLSTMCell
xavier_init = tf.contrib.layers.xavier_initializer()


def add_step_embedding(config, step_input):
    """
        @config:                            model.config
        @inputs:        (int32)             batch_size x 1
        @return:        (config.dtype)      batch_size x hidden_size
    """
    with tf.variable_scope('DecodingEmbeddingLayer'):
        w2v = tf.get_variable('w2v', [config.de_vocab_size, config.batch_size], initializer=xavier_init)
        return tf.nn.embedding_lookup(params=w2v, ids=step_input)

def add_step_projection(config, step_input):
    """
        @config:                            model.config
        @inputs:        (config.dtype)      batch size x hidden_size
        @return:        (config.dtype)      batch_size x de_vocab_size
    """
    with tf.variable_scope("ProjectionLayer"):
        Up = tf.get_variable("Up", [config.hidden_size, config.de_vocab_size], dtype=config.dtype)
        bp = tf.Variable("bp", [config.de_vocab_size], dtype=config.dtype)
        return tf.matmul(step_input, Up) + bp

def add_decoding(model, en_final_state, de_data):
    """
        @model:                             model
        @en_final_state:                 tftuple (2) x batch_size x hidden_size
        @de_data:       (config.dtype)      batch_size x num_steps
        @return:        (config.dtype)      list (num_steps) x batch_size x de_vocab_size
    """
    config = model.config
    train = config.train

    # Initiates the cells
    if not model.de_cells:
        model.de_cells = []
        for layer in xrange(config.layers):
            with tf.variable_scope('DecodingLayer' + str(layer)):
                model.de_cells.append(BasicLSTMCell(config.hidden_size))

    states = []
    outputs = []
    if train:
        de_data = tf.split(de_data, tf.ones(config.num_steps, dtype=config.dtype), axis = 1)

    for step in xrange(config.num_steps):
        if step == 0:
            output = tf.ones([config.batch_size], dtype=config.dtype) * config.start_token
        else:
            if train:
                output = de_data[step - 1]
            else:
                # find the most likely last output prediction
                output = outputs[step - 1]
                output = tf.to_int32(tf.argmax(tf.nn.softmax(output)))

        next_states = []

        # add embedding batch_size x 1 -> batch_size x hidden_size
        output = add_step_embedding(config, output)

        # go through LSTM layers
        for layer in xrange(config.layers):
            cell = model.de_cells[layer]
            if step == 0:
                if layer == 0:
                    state = en_final_state
                else:
                    state = cell.zero_state(config.batch_size)
            else:
                state = states[layer]

            output, state = cell(output, state)
            next_states.append(state)

        # add projection batch_size x hidden_size -> batch_size x de_vocab_size
        output = add_step_projection(config, output)

        states = next_states
        outputs.append(output)
