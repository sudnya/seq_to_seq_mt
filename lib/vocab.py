from collections import defaultdict

class Vocab(object):

    @classmethod
    def __init__(cls, file_name, name, max_vocab_size):
        cls.name = name
        cls.max_vocab_size = max_vocab_size
        cls.word_to_index = {}
        cls.index_to_word = {}
        cls.word_freq = defaultdict(int)
        cls.unknown = '<unk>'
        cls.start = '<--start-->'
        cls.end = '<--end-->'
        cls.add_word(cls.unknown, count=0)
        cls.add_word(cls.end, count=0)
        cls.add_word(cls.start, count=0)
        cls.unknown_idx = cls.decode(cls.unknown)
        cls.end_idx = cls.decode(cls.end)
        cls.start_idx = cls.decode(cls.start)

        for line in open(file_name):
            for word in line.split():
                cls.add_word(word)
        #logger.info(name + " vocab has " + str(len(cls.word_to_index.keys())) + " uniques")

    @classmethod
    def add_word(cls, word, count=1):
        if len(cls.word_to_index.keys()) >= cls.max_vocab_size:
            # stop adding new words when vocab is full
            return

        if word not in cls.word_to_index:
            index = len(cls.word_to_index)
            cls.word_to_index[word] = index
            cls.index_to_word[index] = word
            cls.word_freq[word] += count

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
