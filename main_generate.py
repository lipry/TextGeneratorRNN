import argparse

from src.FileReader import FileReader
from src.ModelSerializer import ModelSerializer


parser = argparse.ArgumentParser(description="Given a trained model generate some poems")
parser.add_argument("model_file", help="Trained model to load")
parser.add_argument("text_file", help="File used for training")
parser.add_argument("-st", "--start_token", default="START_TOK")
parser.add_argument("-et", "--end_token", default="END_TOK")
args = parser.parse_args()

start_token = args.start_token
end_token = args.end_token
serializer_file = args.model_file
file = args.text_file

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

generated = generate_stanza(rnn)
print([index_to_word[s] for s in generated])
