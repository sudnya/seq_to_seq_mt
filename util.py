
import tensorflow as tf

def tfdebug(config, inputs, message):
    if config.debug:
        return tf.Print(inputs, [input], message=message)
    else:
        return input
