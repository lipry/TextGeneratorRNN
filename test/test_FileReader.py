import numpy as np

from src.FileReader import FileReader, OneHotEncodingUtilities


def test_paragraph():
    fr = FileReader("resources/test_input_file")
    p = []
    for par in fr.paragraphs():
        p.append(par)
    assert len(p) == 2
    assert p[0] == ['Meriggiare', 'pallido', 'e', 'assorto', 'presso', 'un', 'rovente', 'muro', "d'orto", ',',
                    'ascoltare', 'tra', 'i', 'pruni', 'e', 'gli', 'sterpi', 'schiocchi', 'di', 'merli', ',',
                    'frusci', 'di', 'serpi', '.', 'Nelle', 'crepe', 'del', 'suolo', 'o', 'su', 'la', 'veccia',
                    'spiar', 'le', 'file', 'di', 'rosse', 'formiche', "ch'ora", 'si', 'rompono', 'ed', 'ora',
                    "s'intrecciano", 'a', 'sommo', 'di', 'minuscole', 'biche', '.', 'Osservare', 'tra', 'frondi',
                    'il', 'palpitare', 'lontano', 'di', 'scaglie', 'di', 'mare', 'mentre', 'si', 'levano', 'tremuli',
                    'scricchi', 'di', 'cicale', 'dai', 'calvi', 'picchi', '.']
    assert p[1] == ['E', 'andando', 'nel', 'sole', 'che', 'abbaglia', 'sentire', 'con', 'triste', 'meraviglia',
                    "com'è", 'tutta', 'la', 'vita', 'e', 'il', 'suo', 'travaglio', 'in', 'questo', 'seguitare',
                    'una', 'muraglia', 'che', 'ha', 'in', 'cima', 'cocci', 'aguzzi', 'di', 'bottiglia', '.']


def test_build_indices():
    fr = FileReader("resources/test_input_file", vocab_size=20)
    fr.build_indices()
    i_w = fr.get_index_to_word()
    w_i = fr.get_word_to_index()
    assert len(i_w) == 20
    assert len(w_i) == 20
    assert i_w[1] == "."
    assert i_w[2] == "e"
    assert i_w[12] == "assorto"
    assert w_i["."] == 1
    assert w_i["e"] == 2
    assert w_i["assorto"] == 12


def test_one_hot_encoding():
    i = 7
    input_dim = 10
    one_hot = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

    v = OneHotEncodingUtilities.one_hot_encoder(i, input_dim)
    assert np.array_equal(v, one_hot)

    index = OneHotEncodingUtilities.one_hot_decoder(one_hot)
    assert index == 7

    one_hot = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    index = OneHotEncodingUtilities.one_hot_decoder(one_hot)
    assert index == 3
