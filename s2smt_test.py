import argparse
import logging
import time
import tensorflow as tf
import numpy as np

from config import Config
from s2smt import S2SMTModel
from s2smt import translate_sentence


def run_translator(session, translate_model, trans_config):
    en_text = 'Welcome!'
    while en_text:
        print ' '.join(translate_sentence(
            session, translate_model, trans_config, en_text=en_text, temp=1.0))
        en_text = raw_input('> ')


def print_predictions(X, Y, predictions, srcVocab, tgtVocab):
    srcStr = ""
    for i in range(len(X)):
        srcStr += str(srcVocab.decode(X[i])) + " "
    
    print "Source sentence: \n", srcStr
    tgtStr = ""
    mini_batches = len(predictions)
    for i in range(mini_batches):
        #print "predictions ", str([x[0] for x in predictions[i]])
        tgtStr += str([tgtVocab.decode(x[0]) for x in predictions[i]])
        #for x in predictions[i]:
        #    tgtStr += tgtVocab.decode(x[0]) + " "
    
    print "\nTranslated to :", tgtStr 

    refStr = ""
    for i in range(len(Y)):
        refStr += str(tgtVocab.decode(Y[i])) + " "
    print "\nReference is :", refStr 

def test_S2SMTModel():
    config = Config()
    trans_config = Config()
    trans_config.batch_size = trans_config.num_steps = 1
    trans_config.train = False 

    # Create model for training
    with tf.variable_scope('S2SMT', reuse=None) as scope:
        model = S2SMTModel(config)
    # Create translate_model to use the same parameters as training result
    with tf.variable_scope('S2SMT', reuse=True) as scope:
        translate_model = S2SMTModel(trans_config)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


    print '\n\n\n TRAINING \n\n\n'

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
                saver.save(session, './s2smt_en_vi.weights')
            if epoch - best_val_epoch > config.early_stopping:
                break
            print 'Total time: {}'.format(time.time() - start)

        #no need to restore here!
        #saver.restore(session, 'ptb_rnnlm.weights')

        test_pp = model.run_epoch(session, model.en_test, model.de_test)
        print '\n=================================================\n'
        print '*** Translator test perplexity: {} ***'.format(test_pp)
        print '\n=================================================\n'

        for i in range(50):
            testX = model.en_test[i]
            testY = model.de_test[i]

            predictions = translate_model.predict(session, [testX])
            print_predictions(testX, testY, predictions, translate_model.src_vocab, translate_model.tgt_vocab)
            #run_translator(session, translate_model, trans_config)


# def test_encoder():
#     t_model = S2SMTModel(Config)
#     t_model.load_data()
#     #
#     # ref_num_steps = t_model.config.seq_len
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
#     #ref_num_steps = t_model.config.seq_len
#     #ref_batch_size = t_model.config.batch_size
#
#     #ref_hidden_size = t_model.config.hidden_size
#     #ref_layer_size = t_model.config.layers
#
#     t_inputs = t_model.add_decoding(en_output,)
#     #assert len(t_inputs) == ref_num_steps
#     return t_inputs

def test_add_model():

    t_model = S2SMTModel(Config)
    t_model.load_data()


    output = t_model.add_model()


    return output

def run_tests():
    #test_add_model()
    # model, temp = test_encoder()
    # test_decoder(model, temp)
    test_S2SMTModel()


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
