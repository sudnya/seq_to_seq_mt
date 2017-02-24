###############################################################################
#
# \file    DataLoader.py
# \author  Sudnya Diamos <mailsudnya@gmail.com>
# \date    Thursday Feb 23, 2017
# \brief   DataLoader for NN machine translation
#
###############################################################################

import os
import argparse
import logging
import time
import numpy as np
import math
from collections import defaultdict
from config import Config
import tensorflow as tf

logger = logging.getLogger('DataLoader')

class DataLoader():

    def __init__(self, cfg):
        train_samples = cfg.train_samples
        dev_samples   = cfg.dev_samples
        test_samples  = cfg.dev_samples
        data_type     = cfg.enc_dtype

        srcV, tgtV, srcTr, tgtTr, srcDev, tgtDev, srcTest, tgtTest = self.initialize_filenames(cfg.lang_src, cfg.lang_tgt)

        self.src_vocab = self.Vocab(srcV, "source")
        self.tgt_vocab = self.Vocab(tgtV, "target")

        self.en_vocab_size = len(self.src_vocab)
        self.de_vocab_size = len(self.tgt_vocab)
        

        #Train - src, rev_src, target (no need to reverse target)
        self.src_encoded_train = self.loadEncodings(srcTr, data_type, train_samples)
        logger.info("source training samples expected: " + str(train_samples) + " created " + str(len(self.src_encoded_train)))

        self.src_encoded_train_rev = self.loadReverseEncodings(srcTr, data_type, train_samples)
        logger.info("reversed source training samples expected: " + str(train_samples) + " created " + str(len(self.src_encoded_train_rev)))
        
        self.tgt_encoded_train = self.loadEncodings(tgtTr, data_type, train_samples)
        logger.info("target training samples expected: " + str(train_samples) + " created " + str(len(self.tgt_encoded_train)))

        
        #dev - src, rev_src, target (no need to reverse target)
        self.src_encoded_dev = self.loadEncodings(srcDev, data_type, dev_samples)
        logger.info("source dev samples expected: " + str(dev_samples) + " created " + str(len(self.src_encoded_dev)))

        self.src_encoded_dev_rev = self.loadReverseEncodings(srcDev, data_type, dev_samples)
        logger.info("reversed source training samples expected: " + str(dev_samples) + " created " + str(len(self.src_encoded_dev_rev)))

        self.tgt_encoded_dev = self.loadEncodings(tgtDev, data_type, dev_samples)
        logger.info("target dev samples expected: " + str(dev_samples) + " created " + str(len(self.tgt_encoded_dev)))


        #test - src, target - No need to rev test data
        self.src_encoded_test = self.loadEncodings(srcTest, data_type, test_samples)
        logger.info("source test samples expected: " + str(test_samples) + " created " + str(len(self.src_encoded_test)))
        
        self.src_encoded_test_rev = self.loadReverseEncodings(srcTest, data_type, test_samples)
        logger.info("reversed test training samples expected: " + str(test_samples) + " created " + str(len(self.src_encoded_test_rev)))
        
        self.tgt_encoded_test = self.loadEncodings(tgtTest, data_type, test_samples)
        logger.info("target test samples expected: " + str(test_samples) + " created " + str(len(self.tgt_encoded_test)))

    def initialize_filenames(self, src, tgt):
        srcVocab = "data/vocab." + src + ".txt"
        tgtVocab = "data/vocab." + tgt + ".txt"
        srcTrain = "data/train." + src + ".txt"
        tgtTrain = "data/train." + tgt + ".txt"
        srcDev   = "data/tst2012." + src + ".txt"
        tgtDev   = "data/tst2012." + tgt + ".txt"
        srcTest  = "data/tst2013." + src + ".txt"
        tgtTest  = "data/tst2013." + tgt + ".txt"
        return srcVocab, tgtVocab, srcTrain, tgtTrain, srcDev, tgtDev, srcTest, tgtTest



    def loadEncodings(self, trFile, data_type, subSamples=1000):
        totalSamples = 0
        encoded_train = []
        for line in open(trFile):
            if totalSamples >= subSamples:
                break
            else:
                words = line.split()
                encoded_train.append(np.array( [self.src_vocab.encode(word) for word in words], dtype=data_type))

                totalSamples += 1
        #logger.debug(encoded_train)
        #logger.info(encoded_train[0])
        logger.debug("training samples " + str(subSamples) + " matrix size " + str(encoded_train[0].shape))
        return encoded_train

    def loadReverseEncodings(self, trFile, data_type, subSamples=10000):
        totalSamples = 0
        encoded_train = []
        for line in open(trFile):
            if totalSamples >= subSamples:
                break
            else:
                words = line.split()
                words.reverse()
                encoded_train.append(np.array( [self.src_vocab.encode(word) for word in words], dtype=data_type))
                totalSamples += 1
        #logger.debug(encoded_train)
        #logger.info(encoded_train[0])
        logger.debug("reverse training samples " + str(subSamples) + " matrix size " + str(encoded_train[0].shape))
        return encoded_train



    class Vocab():
        def __init__(self, fileName, name):
            self.name = name
            self.word_to_index = {}
            self.index_to_word = {}
            self.word_freq = defaultdict(int)
            #TODO: do we need this?
            self.unknown = '<unk>'
            self.add_word(self.unknown, count=0)

            for line in open(fileName):
                for word in line.split():
                    self.add_word(word)
            logger.info(name + " vocab has " + str(len(self.word_to_index.keys())) + " uniques")


        def add_word(self, word, count=1):
            if word not in self.word_to_index:
                index = len(self.word_to_index)
                self.word_to_index[word] = index
                self.index_to_word[index] = word
                self.word_freq[word] += count
                logger.debug("Added " + word)


        def encode(self, word):
            if word not in self.word_to_index:
                word = self.unknown
            return self.word_to_index[word]

        def decode(self, index):
            return self.index_to_word[index]

        def __len__(self):
            return len(self.word_freq)

def main():
    parser = argparse.ArgumentParser(description="DataLoader")
    parser.add_argument("-v", "--verbose", default = False, action = "store_true")
    
    parsedArguments = parser.parse_args()
    arguments = vars(parsedArguments)

    isVerbose   = arguments['verbose']
    
    if isVerbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    viVocab = "data/vocab.vi.txt"
    enVocab = "data/vocab.en.txt"
    viTrain = "data/train.vi.txt"
    enTrain = "data/train.en.txt"
    viDev   = "data/tst2012.vi.txt"
    enDev   = "data/tst2012.en.txt"
    viTest  = "data/tst2013.vi.txt"
    enTest  = "data/tst2013.en.txt"

    logger.info ("Source " + viVocab + " target " + enVocab)
    cfg = Config()
    d = DataLoader(cfg)#viVocab, enVocab, viTrain, enTrain, viDev, enDev, viTest, enTest, 10, 10, 10)
    
if __name__ == '__main__':
    main()
