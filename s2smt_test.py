import argparse
import logging
import time
import tensorflow as tf

from config import Config
from s2smt import S2SMTModel
from s2smt import translate_sentence


def run_translator(session, translate_model, trans_config):
    en_text = 'Welcome!'
    while en_text:
        print ' '.join(translate_sentence(
            session, translate_model, trans_config, en_text=en_text, temp=1.0))
        en_text = raw_input('> ')


def test_S2SMTModel():
    config = Config()
    trans_config = Config()
    trans_config.batch_size = trans_config.num_steps = 1

    # Create model for training
    with tf.variable_scope('RNNLM', reuse=None) as scope:
        model = S2SMTModel(config)
    # Create translate_model to use the same parameters as training result
    with tf.variable_scope('RNNLM', reuse=True) as scope:
        translate_model = S2SMTModel(trans_config)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session() as session:
        best_val_pp = float('inf')
        best_val_epoch = 0

        session.run(init)

        for epoch in xrange(config.max_epochs):
            print 'Epoch {}'.format(epoch)
            start = time.time()
            ###
            train_pp = model.run_epoch(session, model.en_train, model.de_train, train_op=model.train_step)

            valid_pp = model.run_epoch(session, model.en_dev, model.de_dev)

            print 'Training perplexity: {}'.format(train_pp)
            print 'Validation perplexity: {}'.format(valid_pp)
            if valid_pp < best_val_pp:
                best_val_pp = valid_pp
                best_val_epoch = epoch
                saver.save(session, './ptb_rnnlm.weights')
            if epoch - best_val_epoch > config.early_stopping:
                break
            print 'Total time: {}'.format(time.time() - start)

        saver.restore(session, 'ptb_rnnlm.weights')

        test_pp = model.run_epoch(session, model.en_test, model.de_test)
        print '=-=' * 5
        print 'Translator test perplexity: {}'.format(test_pp)
        print '=-=' * 5

        run_translator(session, translate_model, trans_config)


# def test_encoder():
#     t_model = S2SMTModel(Config)
#     t_model.load_data()
#     #
#     # ref_num_steps = t_model.config.en_num_steps
#     # ref_batch_size = t_model.config.batch_size
#     #
#     # ref_hidden_size = t_model.config.hidden_size
#     # ref_layer_size = t_model.config.layers
#     #
#     # output = t_model.add_embedding()
#     #assert len(t_inputs) == ref_num_steps
#
#     # 20  x  <unknown> so cannot be verified
#     # print t_inputs[0].get_shape() , "woooo"
#     #assert t_inputs[0].get_shape() == (ref_batch_size, ref_hidden_size)
#
#     t_rnn_y, f_state = t_model.add_encoding(output)
#     #assert len(t_rnn_y) == ref_num_steps
#     #assert len(f_state) == ref_layer_size
#     return t_model, output

# def test_decoder(t_model, en_output):
#
#     #ref_num_steps = t_model.config.de_num_steps
#     #ref_batch_size = t_model.config.batch_size
#
#     #ref_hidden_size = t_model.config.hidden_size
#     #ref_layer_size = t_model.config.layers
#
#     t_inputs = t_model.add_decoding(en_output, t_model.de_ref_placeholder)
#     #assert len(t_inputs) == ref_num_steps
#     return t_inputs

def test_add_model():

    t_model = S2SMTModel(Config)
    t_model.load_data()


    output = t_model.add_model()


    return output

def run_tests():
    test_add_model()
    # model, temp = test_encoder()
    # test_decoder(model, temp)
    # test_S2SMTModel()



def main():
    parser = argparse.ArgumentParser(description="Sequence to sequence machine translation model")
    parser.add_argument("-v", "--verbose", default=False, action="store_true")

    parsedArguments = parser.parse_args()
    arguments = vars(parsedArguments)
    isVerbose = arguments['verbose']

    if isVerbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run_tests()


if __name__ == '__main__':
    main()
