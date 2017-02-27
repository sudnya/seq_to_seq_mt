
class Lines(object):

    @classmethod
    def __init__(cls, file_name, name, vocab):
        cls.name = name
        cls.vocab = vocab
        cls.lines = cls.encode(open(file_name))

    @classmethod
    def encode_line(cls, line):
        return [cls.vocab.encode(word) for word in line]

    @classmethod
    def decode_line(cls, line):
        return [cls.vocab.decode(word) for word in line]

    @classmethod
    def encode(cls, lines):
        return [cls.encode_line(line) for line in lines]

    @classmethod
    def decode(cls, lines):
        return [cls.decode_line(line) for line in lines]

    @classmethod
    def __len__(cls):
        return len(cls.lines)
