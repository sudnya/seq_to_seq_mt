import numpy as np


def data_iterator(en_data, de_data, batch_size, num_steps, dtype=np.int32):
    en_data = np.array(en_data, dtype=dtype)
    de_data = np.array(de_data, dtype=dtype)
    
    data_len = len(en_data)

    assert data_len == len(de_data), 'encoder data length does not match decoder data length'
    
    len_per_batch = data_len // batch_size

    t_en_data = np.zeroes([batch_size, len_per_batch], dtype=dtype)
    t_de_data = np.zeroes([batch_size, len_per_batch], dtype=dtype)


    for i in range(batch_size):
        t_en_data[i] = en_data[len_per_batch * i : len_per_batch * (i+1)]
        t_de_data[i] = de_data[len_per_batch * i : len_per_batch * (i+1)]

    # we need to run at least 1 epoch and eg: 32 is len of sentence per batch and recurrence
    # length i.e. time steps in RNN is 40 then we are in trouble since we will not have 
    # sufficient length of words to go over
    epochs = (len_per_batch - 1) // num_steps

    assert epochs != 0 , "epochs == 0, decrease batch_size or num_steps"

    # co iterate over encoder and decoder over stride of RNN length
    for i in range(epochs):
        x = t_en_data[:, i * num_steps : (i+1)*num_steps]
        y = t_de_data[:, i * num_steps : (i+1)*num_steps]
        yield(x,y)

