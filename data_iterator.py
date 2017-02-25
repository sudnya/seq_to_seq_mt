import argparse
import logging
import random
from config import Config

import numpy as np


def max_pad(ret_data, r, sample, max_len, pad_token):
    print "sample ", sample
    dlen = min(max_len, len(sample))
    ret_data[r, :dlen] = sample[:dlen]
    return ret_data


def padded_mini_b_fixed(data_slice, batch_size, max_len, pad_token, dtype):
    ret_data = np.ones([batch_size, max_len], dtype=dtype) * pad_token
    print "slice befoer max pad: ", data_slice
    for r, x in enumerate(data_slice):
        #print " and x ", x
        max_pad(ret_data, r, x, max_len, pad_token)
    return ret_data


def data_iterator(config, en_data, de_data = None):

    batch_size = config.batch_size
    en_pad_token = config.en_pad_token
    de_pad_token = config.de_pad_token
    seq_len = config.seq_len
    dtype = config.np_raw_dtype

    if de_data == None:
        # predict mode, no refs here
        #logger.info("decoder data is None, which means we are in predict mode, so no references. creating fake de_data for decoder")
        de_data = [de_pad_token] * len(en_data)
        #print "created in predict y of len ", len(de_data)

    assert len(en_data) == len(de_data), 'encoder data length does not match decoder data length'

    total_batches = len(en_data) // batch_size

    for batch in range(total_batches):

        start = batch * batch_size
        end = start + batch_size

        if seq_len != 0:
            max_len_for_this_batch = seq_len
        else:
            max_len_for_this_batch = en_data[end - 1].shape[0] + 1  # last element in this miniB

        t_en_batch = padded_mini_b_fixed(en_data[start:end], batch_size, max_len_for_this_batch, en_pad_token, dtype)
        #print "encoder batch is fine"
        de_batch = padded_mini_b_fixed(de_data[start:end], batch_size, max_len_for_this_batch, de_pad_token, dtype)

        #print "yielding ", t_en_batch, "\n and \n", de_batch
        yield(t_en_batch, de_batch)


def main():
    parser = argparse.ArgumentParser(description="DataIterator")
    parser.add_argument("-v", "--verbose", default=False, action="store_true")

    parsedArguments = parser.parse_args()
    arguments = vars(parsedArguments)

    isVerbose = arguments['verbose']

    if isVerbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    X_test = []
    Y_test = []
    s_lengths = [int(100 * random.random()) for i in xrange(5)]
    s_lengths.sort()

    for col in s_lengths:
        X_test.append(np.ones(col))
        Y_test.append(np.ones(col))  # TODO/int(random.random() + 1)))

    batch_size = 4
    en_pad_token = -888
    de_pad_token = -555

    print "Test data iterator"
    config = Config()
    for i, (enc, pred_dec) in enumerate(data_iterator(config, X_test, Y_test)):
        print "enc \n", enc, " --- \n pred dec\n", pred_dec


if __name__ == '__main__':
    main()
