import numpy as np
import tensorflow as tf


def tfdebug(config, inputs, message):
    if config.debug:
        return tf.Print(inputs, [inputs], message=message)
    else:
        return inputs


def calculate_perplexity(log_probs):
    # https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf
    perp = 0
    for p in log_probs:
        perp += -p
    return np.exp(perp / len(log_probs))
