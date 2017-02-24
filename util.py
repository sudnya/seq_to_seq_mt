
import tensorflow as tf

def tfdebug(config, inputs, message):
    if config.debug:
        return tf.Print(inputs, [inputs], message=message)
    else:
        return inputs
