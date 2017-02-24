import numpy as np


def data_iterator(en_data, de_data, batch_size, dtype=np.int32):
    # Pulled from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py#L82
    en_data = np.array(en_data, dtype=dtype)
    de_data = np.array(de_data, dtype=dtype)
    data_len = len(en_data)

    assert data_len == len(de_data), 'encoder data length does not match decoder data length'


    data_len = len(full_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=dtype)
    for i in range(batch_size):
        data[i] = full_data[batch_len * i:batch_len * (i + 1)]
    epoch_size = (batch_len - 1) // num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)
