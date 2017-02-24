
class Config(object):
  """Holds model hyperparams and data information.
  """
  # encoder
  # best to keep embed size the same as hidden_size for later residuals
  # embed_size = 100
  en_hidden_size = 100
  en_num_steps = 20

  # decoder
  de_hidden_size = 100
  de_num_steps = 20


  batch_size = 64
  max_epochs = 1
  early_stopping = 2
  dropout = 0.9
  lr = 0.001
  l2 = 0.001
