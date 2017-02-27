import numpy as np


class Vocab(object):

    @classmethod
    def __init__(cls, fileName, name, maxVocabSize):
        cls.maxVocabSize = maxVocabSize
        cls.name = name
        cls.word_to_index = {}
        cls.index_to_word = {}
        cls.word_freq = defaultdict(int)
        # TODO: do we need this?
        cls.unknown = '<unk>'
        cls.add_word(cls.unknown, count=0)

        for line in open(fileName):
            for word in line.split():
                cls.add_word(word)
        #logger.info(name + " vocab has " + str(len(cls.word_to_index.keys())) + " uniques")

    @classmethod
    def add_word(cls, word, count=1):
        if len(cls.word_to_index.keys()) >= cls.maxVocabSize:
            #logger.debug("Vocab capacity full, not adding new words")
            return

        if word not in cls.word_to_index:
            index = len(cls.word_to_index)
            cls.word_to_index[word] = index
            cls.index_to_word[index] = word
            cls.word_freq[word] += count
            #logger.debug("Added " + word)

    @classmethod
    def encode(cls, word):
        if word not in cls.word_to_index:
            word = cls.unknown
        return cls.word_to_index[word]

    @classmethod
    def decode(cls, index):
        return cls.index_to_word[index]

    @classmethod
    def __len__(cls):
        return len(cls.word_freq)


class DataLoader(object):

    srcVocab = "data/vocab." + src + ".txt"
    tgtVocab = "data/vocab." + tgt + ".txt"
    srcTrain = "data/train." + src + ".txt"
    tgtTrain = "data/train." + tgt + ".txt"
    srcDev = "data/tst2012." + src + ".txt"
    tgtDev = "data/tst2012." + tgt + ".txt"
    srcTest = "data/tst2013." + src + ".txt"
    tgtTest = "data/tst2013." + tgt + ".txt"

    def __init__(self):
        pass

    @classmethod
    def load_vocab():




    @staticmethod
    def count_lines(filename):
        return np.sum([1 for line in open(filename)])
