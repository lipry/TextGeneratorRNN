import numpy as np

from src.FileReader import FileReader
from src.ModelSerializer import ModelSerializer
from src.RNNLayer import RnnNetwork

file = "../poesie/test"
serializer_file = "../models/model"
new_model = False

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
    rnn = RnnNetwork(reader.vocab_size, 10)
    serializer.set_model(rnn)
else:
    rnn = serializer.deserialize()

print("Training...")
rnn.train(X_train, Y_train)
print("Training ended, total_loss = {}".format(rnn.total_loss(X_train, Y_train)))
print("Serializing network...")
serializer.serialize()

