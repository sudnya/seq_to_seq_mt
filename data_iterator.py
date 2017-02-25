import argparse
import logging
import random

import numpy as np

def padded_mini_b(data_slice, batch_size, max_len, pad_token, dtype):
    ret_data = np.ones([batch_size, max_len], dtype=dtype)*pad_token

    for i in range(batch_size):
        #one sample
        sample_len  = data_slice[i].shape[0]
        assert sample_len <= max_len, " sample length can never be greater than max length allowed "

        ret_data[i][:sample_len] = data_slice[i][:sample_len]
    return ret_data



def data_iterator(en_data, de_data, batch_size, en_pad_token, de_pad_token, dtype=np.int32):

    if de_data == None:
        #predict mode, no refs here
        logger.info("decoder data is None, which means we are in predict mode, so no references. creating fake de_data for decoder")
        de_data = [de_pad_token]*len(en_data)
    # num_samples x ?

    print len(en_data) , " with first entry ", en_data[0].shape
    print len(de_data) , " with first entry ", de_data[0].shape

    assert len(en_data) == len(de_data), 'encoder data length does not match decoder data length'

    total_batches = len(en_data) // batch_size

    for batch in range(total_batches):

        start = batch * batch_size
        end   = start + batch_size

        max_len_for_this_batch = en_data[end - 1].shape[0] + 1 #last element in this miniB

        t_en_batch      = padded_mini_b(en_data[start:end], batch_size, max_len_for_this_batch, en_pad_token, dtype)
        de_batch = padded_mini_b(de_data[start:end], batch_size, max_len_for_this_batch, de_pad_token, dtype)

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
    s_lengths = [int(100*random.random()) for i in xrange(5)]
    s_lengths.sort()

    for col in s_lengths:
        X_test.append(np.ones(col))
        Y_test.append(np.ones(col))#TODO/int(random.random() + 1)))


    batch_size   = 4
    en_pad_token = -888
    de_pad_token = -555

    for i, (enc, ref_dec, pred_dec) in enumerate(data_iterator(X_test, Y_test, batch_size, en_pad_token, de_pad_token)):
        print "enc \n", enc , " --- \n ref (shifted) dec\n", ref_dec, " --- \n pred dec\n", pred_dec




if __name__ == '__main__':
    main()
