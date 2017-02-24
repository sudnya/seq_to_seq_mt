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


logger = logging.getLogger('DataLoader')

class DataLoader():

    def __init__(self, srcV, tgtV, srcTr, tgtTr, srcDev, tgtDev, srcTest, tgtTest, subSamples, valSamples, testSamples):
        self.vocab_source = self.Vocab(srcV, "source")
        self.vocab_target = self.Vocab(tgtV, "target")
        
        #Train - src, rev_src, target
        self.src_encoded_train = self.loadEncodings(srcTr, subSamples)
        logger.info("source training samples expected: " + str(subSamples) + " created " + str(len(self.src_encoded_train)))

        self.src_encoded_train_rev = self.loadReverseEncodings(srcTr, subSamples)
        logger.info("reversed source training samples expected: " + str(subSamples) + " created " + str(len(self.src_encoded_train_rev)))
        
        self.tgt_encoded_train = self.loadEncodings(tgtTr, subSamples)
        logger.info("target training samples expected: " + str(subSamples) + " created " + str(len(self.tgt_encoded_train)))

        
        #dev - src, rev_src, target
        self.src_encoded_dev = self.loadEncodings(srcDev, valSamples)
        logger.info("source dev samples expected: " + str(valSamples) + " created " + str(len(self.src_encoded_dev)))

        self.src_encoded_dev_rev = self.loadReverseEncodings(srcDev, subSamples)
        logger.info("reversed source training samples expected: " + str(subSamples) + " created " + str(len(self.src_encoded_dev_rev)))

        self.tgt_encoded_dev = self.loadEncodings(tgtDev, valSamples)
        logger.info("target dev samples expected: " + str(valSamples) + " created " + str(len(self.tgt_encoded_dev)))


        #test - src, target - No need to rev test data
        self.src_encoded_test = self.loadEncodings(srcTest, testSamples)
        logger.info("source test samples expected: " + str(testSamples) + " created " + str(len(self.src_encoded_test)))
        
        self.src_encoded_test_rev = self.loadReverseEncodings(srcTest, testSamples)
        logger.info("reversed test training samples expected: " + str(testSamples) + " created " + str(len(self.src_encoded_test_rev)))
        
        self.tgt_encoded_test = self.loadEncodings(tgtTest, testSamples)
        logger.info("target test samples expected: " + str(testSamples) + " created " + str(len(self.tgt_encoded_test)))



    def loadEncodings(self, trFile, subSamples=10000):
        totalSamples = 0
        encoded_train = []
        for line in open(trFile):
            if totalSamples >= subSamples:
                break
            else:
                words = line.split()
                encoded_train.append(np.array( [self.vocab_source.encode(word) for word in words], dtype=np.int32))

                totalSamples += 1
        #logger.debug(encoded_train)
        #logger.info(encoded_train[0])
        logger.debug("training samples " + str(subSamples) + " matrix size " + str(encoded_train[0].shape))
        return encoded_train

    def loadReverseEncodings(self, trFile, subSamples=10000):
        totalSamples = 0
        encoded_train = []
        for line in open(trFile):
            if totalSamples >= subSamples:
                break
            else:
                words = line.split()
                words.reverse()
                encoded_train.append(np.array( [self.vocab_source.encode(word) for word in words], dtype=np.int32))
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
    d = DataLoader(viVocab, enVocab, viTrain, enTrain, viDev, enDev, viTest, enTest, 10, 10, 10)
    
if __name__ == '__main__':
    main()
