
import tensorflow as tf

def tfdebug(config, input, message):
    if config.debug:
        return tf.Print(input, [input], message=message)
    else:
        return input
