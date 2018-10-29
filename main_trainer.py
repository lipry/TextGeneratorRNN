import argparse
import traceback

import numpy as np
import sys

from src.FileReader import FileReader
from src.ModelSerializer import ModelSerializer
from src.RNNLayer import RnnNetwork

parser = argparse.ArgumentParser(description="Train a recursive neural network")
parser.add_argument("input_file", help="The folder or document path of training data")
parser.add_argument("model_file", help="The path where save the final model or load the model if -n is true")
parser.add_argument("-n", "--new", default=1, help="If 1 the model is trained from scratch")
parser.add_argument("--neurons", default=100, help="Number of hidden neurons for every RNN layers")
parser.add_argument("--epochs", default=500, help="Number of training epochs over the dataset")
args = parser.parse_args()

file = args.input_file
serializer_file = args.model_file
new_model = bool(args.new)

print("Reading file {}".format(file))
reader = FileReader(file)

print("Building indices...")
reader.build_indices()
word_to_index = reader.get_word_to_index()

print("Building training set...")
X_train = np.asarray([[word_to_index[word] for word in stanza[:-1]] for stanza in reader.paragraphs()])
Y_train = np.asarray([[word_to_index[word] for word in stanza[1:]] for stanza in reader.paragraphs()])

print("Init serializer...")
serializer = ModelSerializer(serializer_file)

print("Init Network...")
if new_model:
    rnn = RnnNetwork(reader.vocab_size, int(args.neurons))
    serializer.set_model(rnn)
else:
    rnn = serializer.deserialize()

try:
    print("Training...")
    rnn.train(X_train, Y_train, epochs=int(args.epochs))
    print("Training ended, total_loss = {}".format(rnn.total_loss(X_train, Y_train)))
    print("Serializing network...")
    serializer.serialize()

except KeyboardInterrupt:
    serializer.serialize()
    print("Shutdown requested...exiting")
except Exception:
    traceback.print_exc(file=sys.stdout)
    sys.exit(0)
