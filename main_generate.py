from src.FileReader import FileReader
from src.ModelSerializer import ModelSerializer

start_token = 'START_TOK'
end_token = 'END_TOK'
serializer_file = "../models/model"
file = "../poesie/divina_commedia_canto1"

print("Reading file {}".format(file))
reader = FileReader(file)

print("Building indices...")
reader.build_indices()
word_to_index = reader.get_word_to_index()
index_to_word = reader.get_index_to_word()


def generate_stanza(network):
    stanza = [word_to_index[start_token]]
    print(stanza)
    print(word_to_index)
    while stanza[-1] != word_to_index[end_token]:
        predictions = network.prediction(stanza)
        stanza.append(predictions[-1])
        print(stanza)
    return stanza


print("Init serializer...")
serializer = ModelSerializer(serializer_file)
rnn = serializer.deserialize()

stanza = generate_stanza(rnn)
print([index_to_word[s] for s in stanza])
