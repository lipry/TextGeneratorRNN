import itertools
import numpy as np
from nltk.tokenize import word_tokenize

import nltk
nltk.download('punkt')


class FileReader:
    def __init__(self, inputfile, separator='\n', vocab_size = 8000):
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
                    paragraph.extend(word_tokenize(line))
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


class OneHotEncodingUtilities:
    @staticmethod
    def one_hot_encoder(index, input_dim):
        v = np.zeros(input_dim)
        v[index] = 1
        return v

    @staticmethod
    def one_hot_decoder(v):
        return np.argmax(v)






