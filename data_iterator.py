import argparse
import logging
import random

import numpy as np

def padded_mini_b(data_slice, batch_size, max_len, dtype):
    ret_data = np.zeros([batch_size, max_len], dtype=dtype)

    for i in range(batch_size):
        #one sample
        sample_len  = data_slice[i].shape[0]
        print sample_len , " vs. ", max_len
        assert sample_len <= max_len, " sample length can never be greater than max length allowed "
        #print "sample has ", sample_len, " words"
        #print "data slice is ", data_slice[i][:sample_len]

        ret_data[i][:sample_len] = data_slice[i][:sample_len]
        #print "over wrote as ", ret_data[i]
    return ret_data



def data_iterator(en_data, de_data, batch_size, dtype=np.int32):
    # num_samples x ?
    print len(en_data) , " with first entry ", en_data[0].shape
    print len(de_data) , " with first entry ", de_data[0].shape

    #en_data = np.array(en_data, dtype=dtype)
    #de_data = np.array(de_data, dtype=dtype)


    assert len(en_data) == len(de_data), 'encoder data length does not match decoder data length'

    total_buckets = len(en_data) // batch_size

    max_len_for_each_bucket = []
    t_en_data = []
    t_de_data = []

    for bucket in range(total_buckets):

        start = bucket * batch_size
        end   = start + batch_size

        max_len_for_this_bucket = en_data[end - 1].shape[0] #last element in this miniB

        t_en_data.append(padded_mini_b(en_data[start:end], batch_size, max_len_for_this_bucket, dtype))
        t_de_data.append(padded_mini_b(de_data[start:end], batch_size, max_len_for_this_bucket, dtype))
        yield(t_en_data, t_de_data)



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

    #print "X " , X_test
    batch_size = 4
    for i, (enc, dec_ref, dec_pred) in enumerate(data_iterator(X_test, Y_test, batch_size)):
        print "enc \n", enc , " --- \ndec\n", dec



if __name__ == '__main__':
    main()
