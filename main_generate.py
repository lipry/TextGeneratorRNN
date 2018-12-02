import argparse

import numpy as np

from src.FileReader import FileReader
from src.ModelSerializer import ModelSerializer


parser = argparse.ArgumentParser(description="Given a trained model generate some poems")
parser.add_argument("model_file", help="Trained model to load")
parser.add_argument("text_file", help="File used for training")
parser.add_argument("-st", "--start_token", default="START_TOK")
parser.add_argument("-et", "--end_token", default="END_TOK")
parser.add_argument("-ut", "--unknown_token", default="UNKNOWN_TOKEN")
parser.add_argument("-cf", "--completefile", default=False)
args = parser.parse_args()

start_token = args.start_token
end_token = args.end_token
unknown_token = args.unknown_token
serializer_file = args.model_file
file = args.text_file
completefile = args.completefile

print("Reading file {}".format(file))
reader = FileReader(file)

print("Building indices...")
reader.build_indices()
word_to_index = reader.get_word_to_index()
index_to_word = reader.get_index_to_word()


def complete_sentence(network, start):
    sentence = start
    while sentence[-1] != word_to_index[end_token]:
        predictions = network.prediction(sentence)
        sentence.append(predictions[-1])
    sentence = [index_to_word[x] for x in sentence[1:-1]]
    return sentence


def generate_sentence(network):
    sentence = [word_to_index[start_token]]
    return complete_sentence(network, sentence)


print("Init serializer...")
serializer = ModelSerializer(serializer_file)
rnn = serializer.deserialize()

if completefile:
    r = FileReader(completefile)
    paragraph = [[w if w in word_to_index else reader.unknown for w in p] for p in r.paragraphs()]
    X = np.asarray([[word_to_index[word] for word in p[:(len(p) // 2)]] for p in paragraph])
    for x in X:
        print([index_to_word[y] for y in x])
        generated = complete_sentence(rnn, x)
        print(" ".join(generated))
        print("\n")
else:
    generated = complete_sentence(rnn)
    print(" ".join(generated))
    print("\n")
