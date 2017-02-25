###############################################################################
#
# \file    DataLoader.py
# \author  Sudnya Diamos <mailsudnya@gmail.com>
# \date    Thursday Feb 23, 2017
# \brief   DataLoader for NN machine translation
#
###############################################################################

from collections import defaultdict
import argparse
import logging
import numpy as np
from config import Config
#import matplotlib.pyplot as plt


logger = logging.getLogger('DataLoader')


class DataLoader():

    def __init__(self, cfg):
        train_samples = cfg.train_samples
        dev_samples = cfg.dev_samples
        test_samples = cfg.dev_samples
        dataType = cfg.np_raw_dtype

        srcV, tgtV, srcTr, tgtTr, srcDev, tgtDev, srcTest, tgtTest = self.__initializeFilenames__(cfg.source_lang, cfg.target_lang)

        self.src_vocab = self.Vocab(srcV, "source", cfg.vocab_max_size)
        self.tgt_vocab = self.Vocab(tgtV, "target", cfg.vocab_max_size)

        self.en_vocab_size = len(self.src_vocab)
        self.de_vocab_size = len(self.tgt_vocab)

        # Train - src, rev_src, target (no need to reverse target)
        # self.src_encoded_train = self.__loadEncodings__(srcTr, dataType, train_samples)
        # logger.info("source training samples expected: " + str(train_samples) + " created " + str(len(self.src_encoded_train)))

        self.src_encoded_train_rev = self.__loadReverseEncodings__(srcTr, dataType, train_samples)
        logger.info("reversed source training samples expected: " + str(train_samples) + " created " + str(len(self.src_encoded_train_rev)))

        self.tgt_encoded_train = self.__loadEncodings__(tgtTr, dataType, train_samples)
        logger.info("target training samples expected: " + str(train_samples) + " created " + str(len(self.tgt_encoded_train)))

        # dev - src, rev_src, target (no need to reverse target)
        # self.src_encoded_dev = self.__loadEncodings__(srcDev, dataType, dev_samples)
        # logger.info("source dev samples expected: " + str(dev_samples) + " created " + str(len(self.src_encoded_dev)))

        self.src_encoded_dev_rev = self.__loadReverseEncodings__(srcDev, dataType, dev_samples)
        logger.info("reversed source training samples expected: " + str(dev_samples) + " created " + str(len(self.src_encoded_dev_rev)))

        self.tgt_encoded_dev = self.__loadEncodings__(tgtDev, dataType, dev_samples)
        logger.info("target dev samples expected: " + str(dev_samples) + " created " + str(len(self.tgt_encoded_dev)))

        # test - src, rev_src, target (no need to reverse target)
        # self.src_encoded_test = self.__loadEncodings__(srcTest, dataType, test_samples)
        # logger.info("source test samples expected: " + str(test_samples) + " created " + str(len(self.src_encoded_test)))

        self.src_encoded_test_rev = self.__loadReverseEncodings__(srcTest, dataType, test_samples)
        logger.info("reversed test training samples expected: " + str(test_samples) + " created " + str(len(self.src_encoded_test_rev)))

        self.tgt_encoded_test = self.__loadEncodings__(tgtTest, dataType, test_samples)
        logger.info("target test samples expected: " + str(test_samples) + " created " + str(len(self.tgt_encoded_test)))

    def __initializeFilenames__(self, src, tgt):
        srcVocab = "data/vocab." + src + ".txt"
        tgtVocab = "data/vocab." + tgt + ".txt"
        srcTrain = "data/train." + src + ".txt"
        tgtTrain = "data/train." + tgt + ".txt"
        srcDev = "data/tst2012." + src + ".txt"
        tgtDev = "data/tst2012." + tgt + ".txt"
        srcTest = "data/tst2013." + src + ".txt"
        tgtTest = "data/tst2013." + tgt + ".txt"
        return srcVocab, tgtVocab, srcTrain, tgtTrain, srcDev, tgtDev, srcTest, tgtTest

    def __loadEncodings__(self, trFile, dataType, subSamples=1000):
        totalSamples = 0
        encoded_train = []
        for line in open(trFile):
            if totalSamples >= subSamples:
                break
            else:
                words = line.split()
                encoded_train.append(np.array([self.src_vocab.encode(word) for word in words], dtype=dataType))

                totalSamples += 1
        # logger.debug(encoded_train)
        # logger.info(encoded_train[0])
        logger.debug("training samples " + str(subSamples) + " matrix size " + str(encoded_train[0].shape))
        return encoded_train

    def __loadReverseEncodings__(self, trFile, dataType, subSamples=10000):
        totalSamples = 0
        encoded_train = []
        for line in open(trFile):
            if totalSamples >= subSamples:
                break
            else:
                words = line.split()
                words.reverse()
                encoded_train.append(np.array([self.src_vocab.encode(word) for word in words], dtype=dataType))
                totalSamples += 1
        # logger.debug(encoded_train)
        # logger.info(encoded_train[0])
        logger.debug("reverse training samples " + str(subSamples) + " matrix size " + str(encoded_train[0].shape))
        return encoded_train

    def plotLengths(self):
        z = [len(i) for i in self.tgt_encoded_train]

        #plt.hist(z, bins=400)
        #plt.title("Gaussian Histogram")
        #plt.xlabel("Value")
        #plt.ylabel("Frequency")
        #plt.show()

    def getStats(self):
        srcTr = {}
        tgtTr = {}

        for sample in range(len(self.src_encoded_train)):
            lenX = self.src_encoded_train[sample].shape[0]
            if lenX not in srcTr:
                srcTr[lenX] = 1
            else:
                srcTr[lenX] += 1
        logger.info("Source train stats")
        for k, v in srcTr.iteritems():
            logger.info("Sentence of length: " + str(k) + " occurs " + str(v) + " times")

        for sample in range(len(self.tgt_encoded_train)):
            lenX = self.tgt_encoded_train[sample].shape[0]
            if lenX not in tgtTr:
                tgtTr[lenX] = 1
            else:
                tgtTr[lenX] += 1
        logger.info("Source train stats")
        for k, v in tgtTr.iteritems():
            logger.info("Sentence of length: " + str(k) + " occurs " + str(v) + " times")

        # plt.hist(tgtTr)
        # plt.title("Histogram")
        # plt.xlabel("Value")
        # plt.ylabel("Frequency")

        #fig = plt.gcf()

        #plot_url = py.plot_mpl(fig, filename='mpl-basic-histogram')

    class Vocab():

        def __init__(self, fileName, name, maxVocabSize):
            self.maxVocabSize = maxVocabSize
            self.name = name
            self.word_to_index = {}
            self.index_to_word = {}
            self.word_freq = defaultdict(int)
            # TODO: do we need this?
            self.unknown = '<unk>'
            self.add_word(self.unknown, count=0)

            for line in open(fileName):
                for word in line.split():
                    self.add_word(word)
            logger.info(name + " vocab has " + str(len(self.word_to_index.keys())) + " uniques")

        def add_word(self, word, count=1):
            if len(self.word_to_index.keys()) >= self.maxVocabSize:
                logger.debug("Vocab capacity full, not adding new words")
                return

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
    parser.add_argument("-v", "--verbose", default=False, action="store_true")

    parsedArguments = parser.parse_args()
    arguments = vars(parsedArguments)

    isVerbose = arguments['verbose']

    if isVerbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    viVocab = "data/vocab.vi.txt"
    enVocab = "data/vocab.en.txt"
    viTrain = "data/train.vi.txt"
    enTrain = "data/train.en.txt"
    viDev = "data/tst2012.vi.txt"
    enDev = "data/tst2012.en.txt"
    viTest = "data/tst2013.vi.txt"
    enTest = "data/tst2013.en.txt"

    logger.info("Source " + viVocab + " target " + enVocab)
    cfg = Config()
    d = DataLoader(cfg)
    # d.getStats()
    d.plotLengths()

if __name__ == '__main__':
    main()
