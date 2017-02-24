
class Config(object):
  """Holds model hyperparams and data information.
  """
  # encoder
  # best to keep embed size the same as hidden_size for later residuals
  # embed_size = 100
  en_hidden_size = 100
  en_num_steps = 20
  en_layers = 1

  # decoder
  de_hidden_size = 100
  de_num_steps = 20
  de_layers = 1

  # attention
  at_hidden_size = 200


  batch_size = 64
  max_epochs = 1
  early_stopping = 2
  dropout = 0.9
  lr = 0.001
  l2 = 0.001

  dtype = tf.float32

  # lang src is always in reverse
  lang_src = 'vi'
  lang_tgt = 'en'