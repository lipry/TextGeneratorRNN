from src.FileReader import FileReader


class TestFileReader:

    def test_tokenization(self):
        fr = FileReader("resources/test_input_file")
        tokens = fr.tokenize()
        assert type(tokens) is list
        assert len(tokens) > 0

    def test_emptyfile_tokenization(self):
        fr = FileReader("resources/empty_file")
        tokens = fr.tokenize()
        assert type(tokens) is list
        assert len(tokens) == 0
