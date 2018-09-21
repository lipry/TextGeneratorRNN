import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize


class FileReader:
    def __init__(self, inputfile):
        self.inputfile = inputfile

    def _get_file_content(self):
        return open(self.inputfile).read()

    def tokenize(self):
        file_content = self._get_file_content()
        return word_tokenize(file_content)
