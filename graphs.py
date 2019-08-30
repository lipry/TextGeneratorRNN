import argparse
import matplotlib.pyplot as plt

from src.ModelSerializer import ModelSerializer

parser = argparse.ArgumentParser(description="Given a trained model generate some poems")
parser.add_argument("model_file", help="Trained model to load")
parser.add_argument("epochs", help="Number of epochs", type=int)
args = parser.parse_args()

serializer = ModelSerializer(args.model_file)

rnn = serializer.deserialize()

y_train = rnn.losses
y_test = rnn.test_losses
x = range(0, args.epochs, 5)


plt.plot(x, y_train, color='b', label="Train loss")
plt.plot(x, y_test, color='r', label="Test loss")

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim((0, max(max(y_train), max(y_test))))
plt.xlim(0, args.epochs)
plt.legend()
plt.show()
