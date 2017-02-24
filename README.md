# seq_to_seq_mt
A sequence-to-sequence model with attention for Machine Translation

Sequence-to-Sequence Models for Machine Translation
In this assignment you will implement a complete sequence-to-sequence model with attention for Machine Translation. The implementation should be in TensorFlow. The dataset will be the IWSLT'15 English-Vietnamese Parallel Corpus available here: http://nlp.stanford.edu/projects/nmt/.

Some notes about this assignment:
You may not use the high-level sequence-sequence with attention APIs in TensorFlow. You should implement these layers yourself. (It's fine, for example, to use BasicLSTMCell or embedding_lookup, but not to use dynamic_rnn.)
Start with a simple working pipeline including data loading and preprocessing, model training and evaluation. Once you have the pipeline working you can then make the model larger and more complex. For example consider adding the attentional component after you have a working sequence-sequence model.
Start training a small model on a small subset of the data (e.g. 100 sentence pairs), and make sure you are able to fit this data well. Once this is working you can increase the data size and the model size.
We strongly encourage you to work in pairs for this assignment.

Deliverables in order:
A sequence-sequence word-level model which minimizes perplexity on at least a subset of IWSLT'15 corpus.
The above with attention, including a visualization of the alignment from the attention mechanism for a few sentence pairs.
A simple greedy decoder.
An evaluation of BLEU score for the model on the test/validation sets overall, as well as bucketed by length of the target sentence, and some examples of input+translation pairs from your model.

-----------------------
Possible extensions:
-----------------------
Making the encoder bi-directional.
A beam-search decoder.
A word-piece model instead of words.
Other ideas to improve perplexity from MT literature, e.g. architectural modifications described in the reference papers.

References:
Sequence to sequence learning with neural networks: https://arxiv.org/pdf/1409.3215.pdf
Effective approaches to attention-based machine translation: http://www.aclweb.org/anthology/D15-1166
https://arxiv.org/pdf/1609.08144.pdf
