import argparse
import random

import numpy as np

from src.FileReader import FileReader
from src.ModelSerializer import ModelSerializer


parser = argparse.ArgumentParser(description="Given a trained model generate some poems")
parser.add_argument("model_file", help="Trained model to load")
parser.add_argument("text_file", help="File used for training")
parser.add_argument("-st", "--start_token", default="START_TOK")
parser.add_argument("-et", "--end_token", default="END_TOK")
parser.add_argument("-ut", "--unknown_token", default="UNKNOWN_TOKEN")
args = parser.parse_args()

start_token = args.start_token
end_token = args.end_token
unknown_token = args.unknown_token
serializer_file = args.model_file
file = args.text_file

print("Reading file {}".format(file))
reader = FileReader(file)

print("Building indices...")
reader.build_indices()
word_to_index = reader.get_word_to_index()
index_to_word = reader.get_index_to_word()


def generate_sentence(network):
    sentence = [word_to_index[start_token]]
    while sentence[-1] != word_to_index[end_token]:
        predictions = network.prediction(sentence)
        sentence.append(predictions[-1])
    sentence = [index_to_word[x] for x in sentence[1:-1]]
    return sentence

#def generate_sentence(model):
#    new_sentence = [word_to_index[start_token]]

#    while not new_sentence[-1] == word_to_index[end_token]:
#        next_word_probs = model.prediction(new_sentence)
#        sampled_word = word_to_index[unknown_token]
#         while sampled_word == word_to_index[unknown_token]:
#             samples = np.random.multinomial(1, next_word_probs[-1])
#             sampled_word = np.argmax(samples)
#         new_sentence.append(sampled_word)
#     sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
#     return sentence_str


print("Init serializer...")
serializer = ModelSerializer(serializer_file)
rnn = serializer.deserialize()


for _ in range(0, 10):
    generated = generate_sentence(rnn)
    print(" ".join(generated))
    print("\n\n")
