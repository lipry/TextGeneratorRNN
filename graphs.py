import argparse
import matplotlib.pyplot as plt

from src.ModelSerializer import ModelSerializer

parser = argparse.ArgumentParser(description="Given a trained model generate some poems")
parser.add_argument("model_file", help="Trained model to load")
args = parser.parse_args()

serializer = ModelSerializer(args.model_file)

rnn = serializer.deserialize()

y = rnn.losses
x = range(0, 3  00, 5)
print(x)
print(y)

plt.plot(x, y)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim((0, 10))
plt.show()
