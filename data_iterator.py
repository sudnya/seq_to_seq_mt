import numpy as np

def padded_mini_b(data_slice, batch_size, max_len, dtype):
    ret_data = np.zeroes([batch_size, max_len], dtype=dtype)

    for i in range(data_slice):
        #one sample
        sample_len = data_slice[i]
        ret_data   = [data_slice[x] for x in range(sample_len)]
        


def data_iterator(en_data, de_data, batch_size, dtype=np.int32):
    # num_samples x ?
    en_data = np.array(en_data, dtype=dtype)
    de_data = np.array(de_data, dtype=dtype)
    

    assert len(en_data) == len(de_data), 'encoder data length does not match decoder data length'

    total_buckets = len(en_data) // batch_size

    max_len_for_each_bucket = []
    for bucket in range(total_buckets):
        
        start = bucket * batch_size
        end   = start + batch_size
        
        max_len_for_this_bucket = en_data[end - 1].shape[0] #last element in this miniB
        
        t_en_data[bucket] = padded_mini_b(en_data[start:end], batch_size, max_len_for_this_bucket, dtype)
        t_de_data[bucket] = padded_mini_b(de_data[start:end], batch_size, max_len_for_this_bucket, dtype)

        
    
