import itertools
import numpy as np
from nltk.tokenize import word_tokenize

import nltk
nltk.download('punkt')


class FileReader:
    def __init__(self, inputfile, separator='\n', vocab_size=8000):
        self.inputfile = inputfile
        self.separator = separator
        self.vocab_size = vocab_size
        self.index_to_word = []
        self.word_to_index = {}

    def paragraphs(self):
        with open(self.inputfile, "r") as f:
            paragraph = []
            for line in f:
                if line == self.separator:
                    yield paragraph
                    paragraph = []
                else:
                    paragraph.extend(word_tokenize(line, language='italian'))
            if paragraph:
                yield paragraph

    def build_indices(self):
        freqwords = nltk.FreqDist(itertools.chain(*self.paragraphs()))
        self.index_to_word = [t[0] for t in freqwords.most_common(self.vocab_size)]
        self.word_to_index = dict([(w, i) for i, w in enumerate(self.index_to_word)])

    def get_index_to_word(self):
        return self.index_to_word

    def get_word_to_index(self):
        return self.word_to_index

    def get_vocab_size(self):
        return self.vocab_size


class OneHotEncodingUtilities:
    @staticmethod
    def one_hot_encoder(index, input_dim):
        if not isinstance(index, (int, np.int32, np.int64)):
            raise TypeError

        v = np.zeros(input_dim)
        v[index] = 1
        return v

    @staticmethod
    def one_hot_decoder(v):
        return np.argmax(v)






